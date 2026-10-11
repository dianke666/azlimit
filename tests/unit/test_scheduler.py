"""Scheduler 与 worker 单元测试（Requirements 16.1, 16.4, 18.5, 20.2, 23.3, 23.6, 24.3, 25.5, 27.8 等）。

使用 ``tests.property.test_global_window`` 中的 FakeClock 离散事件模拟辅助。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from loadtest.bucket import DualTokenBucket
from loadtest.clock import FakeClock
from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage
from loadtest.cost.guard import CostGuard
from loadtest.load.assembler import TargetUnreachable
from loadtest.load.azure_client import AzureResult
from loadtest.load.breaker import CircuitBreaker
from loadtest.load.models import PlannedRequest
from loadtest.load.worker import ContentFilterWindow, WorkerContext, run_request
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import CostSettings, LoadSettings, ThrottleSettings
from loadtest.throttle import ThrottleController, ThrottleState
from tests.property.test_global_window import FakeAssembler, Harness, SimClock, make_scheduler, ok_script

PRICES = CostSettings().prices


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], str, str | None, str | None]] = []

    async def quarantine(self, ids, reason, detail=None, run_id=None) -> int:
        self.calls.append((list(ids), reason, detail, run_id))
        return len(list(ids))


def _passages(ids, expires_at=None) -> list[Passage]:
    return [Passage(id=i, category="c", lang="zh", token_len=600, use_count=0, expires_at=expires_at,
                    text="x", length_bucket="S") for i in ids]


def _req_no(planned: Any) -> int:
    return int(planned.request_id[1:])


def _at(h: Harness, t: float, fn) -> Any:
    async def co() -> None:
        await h.clock.sleep(t)
        fn()

    return co()


# ====================================================================== 停止原因


def test_budget_refused_first_reserve_zero_dispatch() -> None:
    h = make_scheduler(cost_guard=CostGuard(1e-5, 0.0, PRICES))
    s = h.run(60.0)
    assert s.stop_reason == "budget"
    assert s.dispatched == 0 and h.client.sent == []
    assert h.stop.is_set()
    assert s.t_stop < 5.0
    assert h.cost_guard.outstanding == 0 and h.cost_guard.reserved == 0


def test_budget_exhausted_mid_run_stops_dispatch() -> None:
    cg = CostGuard(0.005, 0.0, PRICES)  # 约十几个请求
    h = make_scheduler(cost_guard=cg, script=ok_script(0.5))
    s = h.run(600.0)
    assert s.stop_reason == "budget"
    assert 0 < s.dispatched == len(h.client.sent)
    assert max(d.t for d in h.dispatches) <= s.t_stop
    assert s.t_stop < 600.0
    assert cg.spent <= cg.cap + 1e-12 and cg.outstanding == 0


def test_corpus_exhausted_stops() -> None:
    h = make_scheduler(assembler=FakeAssembler(fail_after=7))
    s = h.run(120.0)
    assert s.stop_reason == "corpus_exhausted"
    assert s.dispatched == 7
    assert h.cost_guard.outstanding == 0


class UnreachableAbove(FakeAssembler):
    def __init__(self, limit: int | None) -> None:
        super().__init__()
        self.limit = limit
        self.targets: list[int] = []

    def assemble(self, target, rng, now=None):
        if self.limit is None or target > self.limit:
            raise TargetUnreachable("zh", target, 0)
        self.targets.append(target)
        return super().assemble(target, rng, now)


def test_target_unreachable_resamples_target() -> None:
    asm = UnreachableAbove(3000)
    h = make_scheduler(assembler=asm)
    s = h.run(30.0)
    assert s.stop_reason is None
    assert s.dispatched > 0 and s.target_unreachable > 0
    assert all(t <= 3000 for t in asm.targets)


def test_target_always_unreachable_does_not_stop_or_spin() -> None:
    h = make_scheduler(assembler=UnreachableAbove(None))
    s = h.run(20.0)
    assert s.stop_reason is None and s.dispatched == 0
    assert s.t_stop == pytest.approx(20.0)


def test_external_stop_no_dispatch_after_and_drains() -> None:
    h = make_scheduler(script=ok_script(2.0))
    s = h.run(30.0)
    assert s.stop_reason is None
    assert s.t_stop == pytest.approx(30.0)
    assert all(d.t <= 30.0 for d in h.dispatches)
    assert s.drained and s.cancelled == 0
    assert h.metrics.completed == s.dispatched
    assert h.cost_guard.outstanding == 0
    assert h.throttle.in_flight == 0
    assert h.metrics.dispatch_stop == pytest.approx(30.0)


def test_drain_timeout_cancels_and_settles_by_reservation() -> None:
    load = LoadSettings(drain_timeout_s=5.0)
    h = make_scheduler(load=load, script=ok_script(100.0))
    s = h.run(10.0)
    assert not s.drained and s.cancelled == s.dispatched > 0
    assert h.cost_guard.outstanding == 0 and h.cost_guard.spent > 0  # 未完成按预留结算
    assert h.throttle.in_flight == 0
    assert h.sched.semaphore._value == load.max_concurrency
    assert h.clock.now() == pytest.approx(15.0)


# ====================================================================== EWMA / 爬坡 / 游走


def test_ewma_updated_only_on_dispatch() -> None:
    h = make_scheduler(seed=11)
    s = h.run(30.0)
    e = 3040.0
    for d in h.dispatches:
        e += 0.05 * (d.cost - e)
    assert s.ewma_request_tokens == pytest.approx(e)


def test_ewma_unchanged_when_nothing_dispatched() -> None:
    h = make_scheduler(cost_guard=CostGuard(1e-5, 0.0, PRICES))
    s = h.run(10.0)
    assert s.ewma_request_tokens == 3040.0


def test_ramp_zero_is_loop_start_and_halves_rate() -> None:
    h = make_scheduler(seed=2, clock=SimClock(start=1000.0))
    s = h.run(40.0)
    assert s.t_start == pytest.approx(1000.0)
    first = sum(1 for d in h.dispatches if d.t < 1010.0)
    later = sum(1 for d in h.dispatches if 1010.0 <= d.t < 1040.0) / 3
    assert 0.3 <= first / later <= 0.75


def test_no_ramp_full_rate_from_start() -> None:
    h = make_scheduler(seed=2, ramp_enabled=False)
    h.run(40.0)
    first = sum(1 for d in h.dispatches if d.t < 10.0)
    later = sum(1 for d in h.dispatches if 10.0 <= d.t < 40.0) / 3
    assert first / later >= 0.75


def test_tpm_override_sets_target_and_walk_range() -> None:
    h = make_scheduler(tpm_override=400_000)
    assert h.sched.target_tpm == 400_000
    assert h.sched.walk_low == pytest.approx(350_000) and h.sched.walk_high == pytest.approx(425_000)
    s = h.run(200.0)
    assert len(s.walk_history) == 3
    assert all(350_000 <= v <= 425_000 for _, v in s.walk_history)
    # 速率随目标与档位更新（16.6）
    assert h.sched.bucket.tok_rate == pytest.approx(s.final_target_tpm / 60)
    assert h.sched.bucket.req_rate == pytest.approx(500 / 60)


# ====================================================================== 内容过滤 / 熔断


def test_content_filter_burst_pauses_30s_and_quarantines() -> None:
    def script(planned, now):
        return (1.0, Outcome.CONTENT_FILTER, None) if _req_no(planned) <= 6 else (1.0, Outcome.OK, None)

    store = FakeStore()
    index = PassageIndex(_passages(range(1, 11)))
    h = make_scheduler(script=script, store=store, index=index, run_id="run-1")
    s = h.run(90.0)
    cf = h.sched.cf_window
    assert s.cf_pauses == 1 and cf.total_hits == 6
    t6, n = cf.warnings[0]
    assert n == 6
    assert not [d for d in h.dispatches if t6 < d.t < t6 + 30.0]
    assert [d for d in h.dispatches if d.t >= t6 + 30.0]
    assert sorted(c[0][0] for c in store.calls) == [1, 2, 3, 4, 5, 6]
    assert all(c[1] == "azure_content_filter" and c[3] == "run-1" for c in store.calls)
    assert all(pid not in index for pid in range(1, 7)) and all(pid in index for pid in range(7, 11))


def test_content_filter_five_hits_no_pause() -> None:
    def script(planned, now):
        return (1.0, Outcome.CONTENT_FILTER, None) if _req_no(planned) <= 5 else (1.0, Outcome.OK, None)

    h = make_scheduler(script=script)
    s = h.run(30.0)
    assert s.cf_pauses == 0
    gaps = [b.t - a.t for a, b in zip(h.dispatches, h.dispatches[1:])]
    assert max(gaps) < 10.0


def _track_breaker(h: Harness) -> dict[str, list[float]]:
    ev: dict[str, list[float]] = {"open": [], "closed": []}
    last = ["closed"]

    def on_result(planned, res, ticket, now):
        st = h.breaker.state(now)
        if st != last[0]:
            if st == "open":
                ev["open"].append(now)
            elif st == "closed":
                ev["closed"].append(now)
            last[0] = st

    h.sched.ctx.on_result = on_result
    return ev


def test_breaker_open_blocks_dispatch_and_half_open_limits_probes() -> None:
    def script(planned, now):
        return (2.0, Outcome.SERVER_ERROR, None) if now < 5.0 else (2.0, Outcome.OK, None)

    h = make_scheduler(script=script, max_breaker_opens=None, ramp_enabled=False)
    ev = _track_breaker(h)
    h.run(150.0)
    assert len(ev["open"]) == 1 and ev["closed"]
    t_open, t_close = ev["open"][0], ev["closed"][0]
    assert not [d for d in h.dispatches if t_open <= d.t < t_open + 60.0]
    half_open = [d for d in h.dispatches if t_open + 60.0 <= d.t < t_close]
    assert 1 <= len(half_open) <= 5
    assert [d for d in h.dispatches if d.t >= t_close]  # 关闭后恢复派发


def test_third_breaker_open_stops_with_circuit_open() -> None:
    h = make_scheduler(script=lambda p, now: (1.0, Outcome.SERVER_ERROR, None))
    s = h.run(1000.0)
    assert s.stop_reason == "circuit_open"
    assert h.breaker.open_count == 3
    assert s.t_stop < 300.0


# ====================================================================== 获取期间状态变化 / 探测


class GatedBucket(DualTokenBucket):
    """open_at 之前 try_acquire 一律失败（模拟令牌不足）。"""

    def __init__(self, clock, open_at: float) -> None:
        super().__init__(800_000, 500, 5.0, 2.0, clock)
        self.open_at = open_at

    def try_acquire(self, tokens: int) -> bool:
        if self.clock.now() < self.open_at:
            return False
        return super().try_acquire(tokens)

    def _wait_s(self, cost: int) -> float:
        return 0.25


def test_hold_during_bucket_wait_releases_reservation() -> None:
    clock = SimClock()
    h = make_scheduler(clock=clock, bucket=GatedBucket(clock, open_at=5.0))
    seen: dict[str, Any] = {}

    def trigger() -> None:
        seen["reserved_before"] = h.cost_guard.outstanding
        h.throttle.on_throttle(10.0, clock.now())

    def check() -> None:
        seen["reserved_after"] = (h.cost_guard.outstanding, h.cost_guard.reserved)

    s = h.run(40.0, lambda hh: [_at(hh, 2.0, trigger), _at(hh, 2.5, check)])
    assert seen["reserved_before"] == 1  # 预留后在令牌桶处等待
    assert seen["reserved_after"] == (0, 0.0)  # 进入 HOLD → 释放预留
    assert s.released >= 1
    assert not [d for d in h.dispatches if d.t < 12.0]
    first = h.dispatches[0]
    assert first.is_probe and first.state is ThrottleState.PROBE  # PROBE 下派发的 ticket.is_probe 为真


def test_hold_during_semaphore_wait_releases_reservation() -> None:
    load = LoadSettings(max_concurrency=1)
    h = make_scheduler(load=load, script=ok_script(30.0))
    seen: dict[str, Any] = {}

    def trigger() -> None:
        seen["before"] = h.cost_guard.outstanding  # r1 在途 + r2 等待信号量
        h.throttle.on_throttle(10.0, h.clock.now())

    def check() -> None:
        seen["after"] = h.cost_guard.outstanding

    s = h.run(120.0, lambda hh: [_at(hh, 5.0, trigger), _at(hh, 5.5, check)])
    assert seen == {"before": 2, "after": 1}
    assert s.released >= 1
    assert not [d for d in h.dispatches if 5.0 <= d.t < 15.0]
    assert h.cost_guard.outstanding == 0


def test_main_loop_starting_in_hold_sends_single_probe() -> None:
    def script(planned, now):
        return 3.0, Outcome.OK, None

    h = make_scheduler(script=script)
    h.throttle.on_throttle(20.0, 0.0)  # 启动 Probe_Request 返回 THROTTLED（22.8）
    assert h.throttle.state is ThrottleState.HOLD
    h.run(120.0)
    first, second = h.dispatches[0], h.dispatches[1]
    assert first.t >= 20.0 and first.is_probe
    assert second.t >= first.t + 3.0 and not second.is_probe  # 探测结果返回前不派发
    assert h.throttle.stats.probes_sent == 1 and h.throttle.stats.probe_ok == 1


def test_probe_error_waits_5s_before_next_probe() -> None:
    def script(planned, now):
        return 0.5, (Outcome.TIMEOUT if _req_no(planned) <= 2 else Outcome.OK), None

    h = make_scheduler(script=script, max_breaker_opens=None)
    h.throttle.on_throttle(10.0, 0.0)
    h.run(60.0)
    p1, p2, p3 = h.dispatches[:3]
    assert p1.is_probe and p2.is_probe and p3.is_probe
    assert p2.t >= p1.t + 0.5 + 5.0 - 1e-9 and p3.t >= p2.t + 0.5 + 5.0 - 1e-9
    assert h.throttle.stats.probe_error == 2 and h.throttle.stats.probe_ok == 1


# ====================================================================== 补充队列 / 过期


def test_ingest_queue_and_expiry_check() -> None:
    clock = SimClock()
    wall0 = clock.wall()
    index = PassageIndex(_passages([1], expires_at=wall0 + 30.0) + _passages([2]), clock=clock)
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    h = make_scheduler(clock=clock, index=index, replenish_queue=q)
    seen: dict[str, bool] = {}

    def put() -> None:
        q.put_nowait(_passages([99])[0])

    def check_new() -> None:
        seen["ingested"] = 99 in index

    def check_29() -> None:
        seen["alive_29"] = 1 in index

    def check_90() -> None:
        seen["gone_90"] = 1 not in index

    h.run(100.0, lambda hh: [_at(hh, 5.0, put), _at(hh, 8.0, check_new), _at(hh, 29.0, check_29),
                             _at(hh, 90.0, check_90)])
    assert seen == {"ingested": True, "alive_29": True, "gone_90": True}
    assert 2 in index


# ====================================================================== worker


class RecordingGuard(CostGuard):
    def __init__(self, throttle: ThrottleController | None = None) -> None:
        super().__init__(1e9, 0.0, PRICES)
        self.th = throttle
        self.events: list[tuple[str, Any]] = []

    async def settle(self, r, usage, model=None):
        self.events.append(("settle", usage))
        return await super().settle(r, usage, model)

    async def settle_zero(self, r):
        self.events.append(("zero", self.th.state if self.th else None))
        await super().settle_zero(r)


class StubClient:
    def __init__(self, result: AzureResult | Exception) -> None:
        self.result = result

    async def send(self, planned):
        await asyncio.sleep(0)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _planned(ids=(1, 2)) -> PlannedRequest:
    return PlannedRequest(request_id="req-1", lang="zh", messages=[], input_tokens=3000, max_tokens=40,
                          passage_ids=tuple(ids), task_id="t", target_tokens=3000)


async def _run_worker(result, *, is_probe: bool = False, probe_state: bool = False):
    clock = FakeClock(start=100.0)
    th = ThrottleController(ThrottleSettings(), clock)
    if probe_state:
        th.on_throttle(10.0, 90.0)
        th.tick(100.0)
        assert th.state is ThrottleState.PROBE
    br = CircuitBreaker()
    cg = RecordingGuard(th)
    store = FakeStore()
    index = PassageIndex(_passages([1, 2, 3]))
    sem = asyncio.Semaphore(1)
    await sem.acquire()
    ctx = WorkerContext(client=StubClient(result), throttle=th, breaker=br, cost_guard=cg, index=index,
                        clock=clock, store=store, run_id="r", cf_window=ContentFilterWindow(), semaphore=sem)
    planned = _planned()
    r = await cg.reserve("gpt-4.1-nano", planned.input_tokens, planned.max_tokens)
    ticket = th.note_dispatch(clock.now(), is_probe=is_probe)
    res = await run_request(ctx, planned, r, ticket)
    return res, ctx, cg, store, sem, r


def _result(outcome: Outcome, **kw) -> AzureResult:
    return AzureResult(request_id="req-1", outcome=outcome, status=kw.pop("status", 200), latency_ms=10.0, **kw)


async def test_worker_throttled_settles_zero_after_sync_hold() -> None:
    res, ctx, cg, _, sem, _ = await _run_worker(_result(Outcome.THROTTLED, status=429, retry_after_s=15.0))
    assert cg.events == [("zero", ThrottleState.HOLD)]  # on_result 在结算 await 之前同步执行
    assert cg.spent == 0 and cg.outstanding == 0
    assert ctx.throttle.hold_until == pytest.approx(115.0)
    assert ctx.throttle.in_flight == 0 and not sem.locked()
    assert len(ctx.breaker._window) == 0  # THROTTLED 不计入熔断


async def test_worker_ok_settles_by_usage_and_records_breaker() -> None:
    res, ctx, cg, _, _, r = await _run_worker(
        _result(Outcome.OK, prompt_tokens=3000, completion_tokens=25, cached_tokens=1000,
                headers=RateLimitHeaders(remaining_tokens=10, limit_tokens=1000))
    )
    kind, usage = cg.events[0]
    assert kind == "settle" and (usage.prompt_tokens, usage.completion_tokens, usage.cached_tokens) == (3000, 25, 1000)
    assert 0 < cg.spent < r.amount
    assert len(ctx.breaker._window) == 1
    assert ctx.throttle.level == 0.75  # 主动减速（21.1）


async def test_worker_timeout_settles_by_reservation() -> None:
    res, ctx, cg, _, _, r = await _run_worker(_result(Outcome.TIMEOUT, status=None))
    assert cg.events == [("settle", None)]
    assert cg.spent == pytest.approx(r.amount)
    assert ctx.breaker.consecutive == 1


async def test_worker_client_exception_is_server_error() -> None:
    res, ctx, cg, _, sem, r = await _run_worker(RuntimeError("boom"))
    assert res.outcome is Outcome.SERVER_ERROR and res.status is None
    assert ctx.breaker.consecutive == 1
    assert cg.spent == pytest.approx(r.amount) and cg.outstanding == 0
    assert ctx.throttle.in_flight == 0 and not sem.locked()


async def test_worker_probe_content_filter_quarantines_and_allows_reprobe() -> None:
    res, ctx, cg, store, _, _ = await _run_worker(
        _result(Outcome.CONTENT_FILTER, status=400, filter_detail="hate"), is_probe=True, probe_state=True
    )
    assert store.calls == [([1, 2], "azure_content_filter", "hate", "r")]
    assert 1 not in ctx.index and 2 not in ctx.index and 3 in ctx.index
    assert ctx.cf_window.total_hits == 1
    th = ctx.throttle
    assert th.state is ThrottleState.PROBE and not th.probe_in_flight
    assert th.stats.probe_content_filter == 1
    assert th.allowed_dispatch(100.0)[0] == 1  # 可立即再探
    assert len(ctx.breaker._window) == 0  # CONTENT_FILTER 不计入熔断


async def test_worker_probe_error_counts_in_breaker() -> None:
    res, ctx, _, _, _, _ = await _run_worker(_result(Outcome.SERVER_ERROR, status=500), is_probe=True,
                                             probe_state=True)
    assert ctx.breaker.consecutive == 1
    assert ctx.throttle.probe_ready_at == pytest.approx(105.0)


async def test_worker_cancel_settles_by_reservation() -> None:
    clock = FakeClock()
    th = ThrottleController(ThrottleSettings(), clock)
    cg = RecordingGuard(th)

    class Hanging:
        async def send(self, planned):
            await asyncio.Event().wait()

    sem = asyncio.Semaphore(1)
    await sem.acquire()
    ctx = WorkerContext(client=Hanging(), throttle=th, breaker=CircuitBreaker(), cost_guard=cg,
                        index=PassageIndex(), clock=clock, semaphore=sem)
    planned = _planned()
    r = await cg.reserve("gpt-4.1-nano", 3000, 40)
    task = asyncio.ensure_future(run_request(ctx, planned, r, th.note_dispatch(0.0, False)))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cg.events == [("settle", None)] and cg.spent == pytest.approx(r.amount)
    assert th.in_flight == 0 and not sem.locked()


def test_content_filter_window_threshold() -> None:
    w = ContentFilterWindow()
    for i in range(5):
        assert not w.record(float(i))
    assert w.record(5.0) and w.pause_until == 35.0 and w.pauses == 1
    # 窗口滑出后不再触发
    w2 = ContentFilterWindow()
    for t in (0, 10, 20, 30, 40, 61):
        w2.record(float(t))
    assert w2.pauses == 0 and w2.count(61.0) == 5
