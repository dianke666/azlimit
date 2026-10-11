"""Property 7: 全局 60 秒窗口。

∀ 模拟运行（FakeClock 离散事件模拟，含随机游走、启动爬坡、随机 429 与 ``--tpm T``），
任一 60 秒滑动窗口内放行的 Σ(input + max_tokens) ≤ tpm_walk_high × (1 + bucket_burst_s / 60)
≤ 0.95 × quota_tpm，请求数 ≤ rpm_max × (1 + bucket_burst_s / 60)，目标 TPM 始终位于游走区间内。

**Validates: Requirements 17.3, 16.2, 16.3, 16.6, 16.7, 16.8**

本模块同时提供调度器离散事件模拟辅助（``SimClock`` / ``run_sim`` / ``FakeAzure`` /
``FakeAssembler`` / ``make_scheduler``），供其他调度器测试复用。
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from random import Random
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.corpus.index import PassageIndex
from loadtest.cost.guard import CostGuard
from loadtest.load.assembler import CorpusExhausted
from loadtest.load.azure_client import AzureResult
from loadtest.load.breaker import CircuitBreaker
from loadtest.load.models import PlannedRequest
from loadtest.load.scheduler import Scheduler
from loadtest.metrics.collector import MetricsCollector
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import CostSettings, LoadSettings, ThrottleSettings
from loadtest.throttle import ThrottleController

# ====================================================================== 离散事件模拟


class SimClock(FakeClock):
    """sleep 注册到事件堆，由 ``run_sim`` 按时间顺序唤醒（不真正等待）。"""

    def __init__(self, start: float = 0.0) -> None:
        super().__init__(start=start, sleep_fn=self._sim_sleep)
        self._heap: list[tuple[float, int, asyncio.Future]] = []
        self._seq = itertools.count()

    async def _sim_sleep(self, seconds: float) -> None:
        fut = asyncio.get_running_loop().create_future()
        heapq.heappush(self._heap, (self.now() + seconds, next(self._seq), fut))
        await fut

    async def sleep(self, seconds: float) -> None:  # 不记录 sleeps（长模拟时避免内存增长）
        await self._sim_sleep(max(0.0, float(seconds)))


async def _settle(loop: asyncio.AbstractEventLoop) -> None:
    for _ in range(100_000):
        await asyncio.sleep(0)
        if not loop._ready:  # type: ignore[attr-defined]  # 其他协程都已阻塞
            return


async def run_sim(clock: SimClock, main: Coroutine[Any, Any, Any], max_time: float = 1e7) -> Any:
    """运行 main 直到完成；所有协程阻塞时把时钟推进到最早的唤醒点。"""
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(main)
    while not task.done():
        await _settle(loop)
        if task.done():
            break
        heap = clock._heap
        while heap and heap[0][2].done():
            heapq.heappop(heap)
        if not heap:
            task.cancel()
            raise RuntimeError("simulation deadlock: no pending timers")
        t, _, fut = heapq.heappop(heap)
        if t > max_time:
            task.cancel()
            raise RuntimeError(f"simulation exceeded max_time={max_time}")
        clock.set(max(t, clock.now()))
        fut.set_result(None)
    return task.result()


# ====================================================================== 假组件

Script = Callable[[Any, float], tuple[float, Outcome, float | None]]  # (planned, now) -> (latency, outcome, retry_after)


def ok_script(latency: float = 1.0) -> Script:
    return lambda planned, now: (latency, Outcome.OK, None)


class FakeAzure:
    """按脚本返回结果的 Azure 客户端；latency 经 clock.sleep 模拟。"""

    def __init__(self, clock: SimClock, script: Script | None = None) -> None:
        self.clock = clock
        self.script = script or ok_script()
        self.sent: list[tuple[float, str]] = []

    async def send(self, planned: Any) -> AzureResult:
        now = self.clock.now()
        self.sent.append((now, planned.request_id))
        latency, outcome, retry_after = self.script(planned, now)
        await self.clock.sleep(latency)
        usage = outcome in (Outcome.OK, Outcome.CONTENT_FILTER)
        status = {
            Outcome.OK: 200, Outcome.THROTTLED: 429, Outcome.CONTENT_FILTER: 400,
            Outcome.CLIENT_ERROR: 400, Outcome.SERVER_ERROR: 500, Outcome.TIMEOUT: None,
        }[outcome]
        return AzureResult(
            request_id=planned.request_id,
            outcome=outcome,
            status=status,
            latency_ms=latency * 1000,
            prompt_tokens=planned.input_tokens if usage else None,
            completion_tokens=30 if outcome is Outcome.OK else (0 if usage else None),
            cached_tokens=0 if usage else None,
            retry_after_s=retry_after,
            headers=RateLimitHeaders(),
        )


class FakeAssembler:
    """不依赖语料的组装器：input_tokens = target（cost ≤ 6040）。"""

    def __init__(self, lang: str = "zh", fail_after: int | None = None) -> None:
        self.lang = lang
        self.fail_after = fail_after
        self._n = itertools.count(1)

    def assemble(self, target: int, rng: Random, now: float | None = None) -> PlannedRequest:
        n = next(self._n)
        if self.fail_after is not None and n > self.fail_after:
            raise CorpusExhausted(self.lang)
        return PlannedRequest(
            request_id=f"r{n}",
            lang=self.lang,  # type: ignore[arg-type]
            messages=[],
            input_tokens=int(target),
            max_tokens=40,
            passage_ids=(n,),
            task_id="t",
            target_tokens=int(target),
        )


@dataclass
class DispatchRec:
    t: float
    request_id: str
    cost: int
    is_probe: bool
    state: Any
    throttle_in_flight: int  # note_dispatch 之后的 in_flight（含本请求）
    probe_phase_start: float


@dataclass
class Harness:
    clock: SimClock
    sched: Scheduler
    throttle: ThrottleController
    breaker: CircuitBreaker
    cost_guard: CostGuard
    client: FakeAzure
    stop: asyncio.Event
    metrics: MetricsCollector
    dispatches: list[DispatchRec] = field(default_factory=list)
    ticks: list[float] = field(default_factory=list)

    async def run_for(self, duration: float, extra: list[Coroutine[Any, Any, Any]] | None = None) -> Any:
        async def stopper() -> None:
            await self.clock.sleep(duration)
            self.stop.set()

        async def main() -> Any:
            bg = [asyncio.ensure_future(stopper())]
            bg += [asyncio.ensure_future(c) for c in (extra or [])]
            try:
                return await self.sched.run()
            finally:
                for b in bg:
                    b.cancel()

        return await run_sim(self.clock, main())

    def run(self, duration: float, extra_factory: Callable[["Harness"], list] | None = None) -> Any:
        async def go() -> Any:
            return await self.run_for(duration, extra_factory(self) if extra_factory else None)

        return asyncio.run(go())


def make_scheduler(
    *,
    seed: int = 0,
    load: LoadSettings | None = None,
    script: Script | None = None,
    assembler: Any = None,
    index: PassageIndex | None = None,
    cost_guard: CostGuard | None = None,
    throttle_cfg: ThrottleSettings | None = None,
    breaker: CircuitBreaker | None = None,
    quota_rpm: int = 1000,
    clock: SimClock | None = None,
    **kw: Any,
) -> Harness:
    """在 asyncio.run 内部或外部均可调用（Scheduler 构造不需要运行中的事件循环）。"""
    clock = clock or SimClock()
    load = load or LoadSettings()
    throttle = ThrottleController(throttle_cfg or ThrottleSettings(), clock)
    breaker = breaker or CircuitBreaker()
    cg = cost_guard or CostGuard(1e9, 0.0, CostSettings().prices)
    client = FakeAzure(clock, script)
    metrics = MetricsCollector(clock, throttle=throttle)
    stop = asyncio.Event()
    h_ref: dict[str, Harness] = {}

    def on_dispatch(now: float, planned: Any, ticket: Any) -> None:
        h_ref["h"].dispatches.append(
            DispatchRec(now, planned.request_id, planned.input_tokens + planned.max_tokens, ticket.is_probe,
                        throttle.state, throttle.in_flight, throttle.probe_phase_start)
        )

    sched = Scheduler(
        load,
        assembler=assembler or FakeAssembler(),
        cost_guard=cg,
        throttle=throttle,
        breaker=breaker,
        client=client,
        index=index if index is not None else PassageIndex(),
        clock=clock,
        rng=Random(seed),
        model="gpt-4.1-nano",
        quota_rpm=quota_rpm,
        metrics=metrics,
        stop_event=stop,
        on_dispatch=on_dispatch,
        **kw,
    )
    h = Harness(clock, sched, throttle, breaker, cg, client, stop, metrics)
    h_ref["h"] = h
    orig_tick = throttle.tick

    def tick(now: float) -> None:
        h.ticks.append(now)
        orig_tick(now)

    throttle.tick = tick  # type: ignore[method-assign]
    return h


def window_max(times: list[float], weights: list[float], width: float = 60.0) -> float:
    """任一闭区间 [t, t + width] 内权重之和的最大值（times 升序）。"""
    best = 0.0
    j = 0
    acc = 0.0
    for i, t in enumerate(times):
        acc += weights[i]
        while times[j] < t - width:
            acc -= weights[j]
            j += 1
        best = max(best, acc)
    return best


# ====================================================================== Property 7


def _random_script(seed: int, p_throttle: float) -> Script:
    rng = Random(seed ^ 0x5EED)

    def script(planned: Any, now: float) -> tuple[float, Outcome, float | None]:
        lat = rng.uniform(0.3, 4.0)
        if rng.random() < p_throttle:
            return lat, Outcome.THROTTLED, rng.choice([None, 0.0, 5.0, 12.0, 30.0])
        return lat, Outcome.OK, None

    return script


@settings(max_examples=6, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**31 - 1),
    tpm=st.one_of(st.none(), st.integers(100_000, 825_000)),
    rpm_max=st.integers(12, 876),
    ramp=st.booleans(),
    p_throttle=st.sampled_from([0.0, 0.0, 0.002, 0.01]),
)
def test_global_60s_window(seed: int, tpm: int | None, rpm_max: int, ramp: bool, p_throttle: float) -> None:
    load = LoadSettings(rpm_min=min(200, rpm_max), rpm_max=rpm_max)
    quota_tpm, quota_rpm = 1_000_000, 1_000
    h = make_scheduler(
        seed=seed, load=load, script=_random_script(seed, p_throttle), quota_rpm=quota_rpm,
        tpm_override=tpm, ramp_enabled=ramp,
    )
    summary = h.run(300.0)
    assert summary.stop_reason is None
    assert h.dispatches, "expected some dispatches"

    walk_high = h.sched.walk_high
    walk_low = h.sched.walk_low
    burst = load.bucket_burst_s
    tok_bound = walk_high * (1 + burst / 60)
    req_bound = min(rpm_max, quota_rpm * 0.5) * (1 + burst / 60)
    assert tok_bound <= 0.95 * quota_tpm + 1e-6  # 16.8 校验通过的配置

    times = [d.t for d in h.dispatches]
    assert times == sorted(times)
    assert window_max(times, [d.cost for d in h.dispatches]) <= tok_bound + 1e-6
    assert window_max(times, [1.0] * len(times)) <= req_bound + 1e-6
    assert len(times) <= rpm_max * (1 + burst / 60) * (300 / 60 + 1)

    # 目标 TPM 始终位于游走区间内，且自 60s 起每 60s 游走一次（16.2 / 16.7）
    if tpm is not None:
        assert abs(walk_low - tpm * 0.875) < 1e-6 and abs(walk_high - tpm * 1.0625) < 1e-6
    else:
        assert (walk_low, walk_high) == (load.tpm_walk_low, load.tpm_walk_high)
    assert all(walk_low - 1e-9 <= v <= walk_high + 1e-9 for _, v in summary.walk_history)
    assert walk_low <= summary.final_target_tpm <= walk_high
    walk_times = [t for t, _ in summary.walk_history]
    assert walk_times and walk_times[0] >= summary.t_start + 60 - 1e-9
    assert all(b - a >= 60 - 1e-9 for a, b in zip(walk_times, walk_times[1:]))
    assert all(b - a <= 61 + 1e-9 for a, b in zip([summary.t_start + 0.0] + walk_times, walk_times))


def test_window_bound_default_saturating() -> None:
    """默认配置、全部 OK：60s 窗口逼近但不超过上界，且实际 TPM 接近目标。"""
    load = LoadSettings()
    h = make_scheduler(seed=1, load=load, script=ok_script(1.0), ramp_enabled=False)
    summary = h.run(240.0)
    times = [d.t for d in h.dispatches]
    costs = [d.cost for d in h.dispatches]
    assert window_max(times, costs) <= load.tpm_walk_high * (1 + load.bucket_burst_s / 60) + 1e-6
    assert window_max(times, [1.0] * len(times)) <= load.rpm_max * (1 + load.bucket_burst_s / 60) + 1e-6
    tpm = sum(costs) / ((summary.t_stop - summary.t_start) / 60)
    assert 600_000 <= tpm <= 920_000
