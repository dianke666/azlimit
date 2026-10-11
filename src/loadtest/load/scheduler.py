"""Scheduler：泊松到达 + 目标随机游走的 Azure 派发主循环（design.md 第 4 节、任务 15.2）。

每次循环（Requirements 4.3, 10.3, 10.5, 16.x, 17.3, 18.4, 18.5, 19.4, 20.2, 20.3, 20.6,
23.3, 23.6, 24.3, 25.5, 27.3, 27.8）::

    ingest_pending(queue, 64)（只用 get_nowait，从不等待补充任务）
    每 60s 过期检查；自 t_start + 60s 起每 60s 目标 TPM += N(0, 30k²) 并截断到游走区间
    throttle.tick(t)
    blocked(t)（HOLD / PROBE 未就绪 / 熔断 open 或 half_open 名额用尽 / 内容过滤暂停）
        → sleep_ticking(≤ 1s 步长) 后 continue
    NORMAL / RECOVER: eff_level = min(ramp(t − t_start), throttle.level)
    PROBE:            eff_level = levels[0]，跳过泊松等待，只派发 1 个 Throttle_Probe
    bucket.set_rates(target × eff, min(rpm_max, quota_rpm × 0.5) × eff)
    泊松等待 Exp(λ)，λ = min(eff_tpm / EWMA, eff_rpm) / 60
    组装（TargetUnreachable → 重采样目标长度；CorpusExhausted → stop_reason=corpus_exhausted）
    → CostGuard.reserve（None → stop_reason=budget）→ 令牌桶 → 并发信号量
    获取期间停止 / 进入阻塞状态 / PROBE 状态变化 → 释放预留与信号量、不派发
    ticket = throttle.note_dispatch(now, is_probe)；EWMA 只在实际派发后更新；spawn worker

所有等待（阻塞轮询、泊松间隔、令牌桶、信号量、drain）都拆成 ≤ ``tick_interval_s``（1s）
的小段，每段后调用 ``throttle.tick``，并在停止信号到达时立即返回。
令牌桶以 ``try_acquire`` + 分段等待实现（本调度器是唯一使用者，与 ``acquire`` 的放行
序列一致），从而在等待令牌期间同样保持 tick 间隔 ≤ 1s。
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from typing import Any

from loadtest.bucket import DualTokenBucket
from loadtest.load.assembler import CorpusExhausted, TargetUnreachable
from loadtest.load.distribution import sample_target_length
from loadtest.load.ramp import StartupRamp
from loadtest.load.worker import ContentFilterWindow, WorkerContext, run_request
from loadtest.settings import LoadSettings
from loadtest.throttle import DispatchTicket, ThrottleState

log = logging.getLogger(__name__)

EWMA_INIT = 3040.0
WALK_SIGMA = 30_000.0
WALK_INTERVAL_S = 60.0
EXPIRE_INTERVAL_S = 60.0
INGEST_MAX = 64
BUCKET_STEP_S = 0.25


@dataclass
class RunSummary:
    stop_reason: str | None  # 调度器自身决定的停止原因（budget / corpus_exhausted / circuit_open）；外部停止为 None
    dispatched: int
    probes_dispatched: int
    released: int  # 已预留但未派发（获取期间状态变化 / 停止）而退回的预留数
    t_start: float
    t_stop: float | None
    final_target_tpm: float
    ewma_request_tokens: float
    walk_history: list[tuple[float, float]] = field(default_factory=list)  # (时刻, 目标 TPM)
    drained: bool = True  # drain 期间全部在途请求正常结束
    cancelled: int = 0  # drain 超时 / 立即退出时被取消（按预留结算）的在途请求数
    target_unreachable: int = 0  # 因 TargetUnreachable 重采样目标长度的次数
    cf_pauses: int = 0  # 内容过滤暂停次数


class Scheduler:
    def __init__(
        self,
        load: LoadSettings,
        *,
        assembler: Any,
        cost_guard: Any,
        throttle: Any,
        breaker: Any,
        client: Any,
        index: Any,
        clock: Any,
        rng: Random,
        model: str,
        quota_rpm: int = 1000,
        bucket: DualTokenBucket | None = None,
        metrics: Any = None,
        store: Any = None,
        run_id: str | None = None,
        jsonl: Any = None,
        replenish_queue: asyncio.Queue | None = None,
        stop_event: asyncio.Event | None = None,
        tpm_override: float | None = None,
        ramp_enabled: bool = True,
        cf_window: ContentFilterWindow | None = None,
        length_sampler: Callable[[Random, LoadSettings], int] = sample_target_length,
        max_target_retries: int = 5,
        max_breaker_opens: int | None = 3,
        walk_sigma: float = WALK_SIGMA,
        walk_interval_s: float = WALK_INTERVAL_S,
        expire_interval_s: float = EXPIRE_INTERVAL_S,
        ewma_init: float = EWMA_INIT,
        on_dispatch: Callable[[float, Any, DispatchTicket], None] | None = None,
    ) -> None:
        self.load = load
        self.assembler = assembler
        self.cost_guard = cost_guard
        self.throttle = throttle
        self.breaker = breaker
        self.client = client
        self.index = index
        self.clock = clock
        self.rng = rng
        self.model = model
        self.quota_rpm = int(quota_rpm)
        self.bucket = bucket
        self.metrics = metrics
        self.store = store
        self.run_id = run_id
        self.jsonl = jsonl
        self.queue = replenish_queue
        self.stop = stop_event if stop_event is not None else asyncio.Event()
        self.ramp = StartupRamp(load.startup_ramp, enabled=ramp_enabled)
        self.cf_window = cf_window if cf_window is not None else ContentFilterWindow()
        self.length_sampler = length_sampler
        self.max_target_retries = max(1, int(max_target_retries))
        self.max_breaker_opens = max_breaker_opens
        self.walk_sigma = float(walk_sigma)
        self.walk_interval_s = float(walk_interval_s)
        self.expire_interval_s = float(expire_interval_s)
        self.on_dispatch = on_dispatch
        self.tick_s = float(load.tick_interval_s)

        # 目标 TPM 与游走区间（16.2 / 16.7）
        if tpm_override is not None:
            t = float(tpm_override)
            self.walk_low, self.walk_high = t * load.walk_low_ratio, t * load.walk_high_ratio
        else:
            t = float(load.target_tpm)
            self.walk_low, self.walk_high = float(load.tpm_walk_low), float(load.tpm_walk_high)
        self.target_tpm = min(max(t, self.walk_low), self.walk_high)
        self.walk_history: list[tuple[float, float]] = []

        # 运行状态
        self.ewma_request_tokens = float(ewma_init)
        self.eff_level = 0.0
        self.stop_reason: str | None = None
        self.dispatched = 0
        self.probes_dispatched = 0
        self.released = 0
        self.target_unreachable = 0
        self.cancelled = 0
        self.drained = True
        self.t_start: float | None = None
        self.t_stop: float | None = None
        self._next_walk = math.inf
        self._next_expire = math.inf
        self._tasks: set[asyncio.Task] = set()
        self._slot_freed = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self.semaphore = asyncio.Semaphore(int(load.max_concurrency))
        self.ctx = WorkerContext(
            client=client,
            throttle=throttle,
            breaker=breaker,
            cost_guard=cost_guard,
            index=index,
            clock=clock,
            metrics=metrics,
            store=store,
            run_id=run_id,
            cf_window=self.cf_window,
            jsonl=jsonl,
            semaphore=self.semaphore,
        )

    # ------------------------------------------------------------------ 时钟辅助
    def _now(self) -> float:
        c = self.clock
        return float(c.now() if hasattr(c, "now") else c())

    def _wall(self) -> float | None:
        w = getattr(self.clock, "wall", None)
        return float(w()) if callable(w) else None

    async def _wait_event(self, event: asyncio.Event, timeout: float) -> None:
        """等待 event 或 clock.sleep(timeout)，先到者返回。"""
        if event.is_set():
            return
        sleeper = asyncio.ensure_future(self.clock.sleep(max(0.0, timeout)))
        waiter = asyncio.ensure_future(event.wait())
        try:
            await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for f in (sleeper, waiter):
                if not f.done():
                    f.cancel()

    def _next_due(self, t: float) -> float:
        """节流器下一个状态变化时刻（HOLD 结束 / RECOVER 下一次升档）。

        等待分段在该时刻精确唤醒并 tick，避免每档升档延迟累积（20.5：回满 ≤ recover_full_s + 1s）。
        """
        th = self.throttle
        s = th.state
        if s is ThrottleState.HOLD:
            due = float(getattr(th, "hold_until", math.inf))
        elif s is ThrottleState.RECOVER:
            due = float(getattr(th, "last_event", -math.inf)) + float(getattr(th, "step_s", math.inf))
        else:
            return math.inf
        return due if due > t else math.inf

    def _chunk(self, t: float, end: float = math.inf) -> float:
        """下一段等待时长：≤ tick_interval_s，且不越过 end 与节流器的下一个状态变化时刻。"""
        return max(0.0, min(self.tick_s, end - t, max(self._next_due(t) - t, 1e-9)))

    async def sleep_ticking(self, duration: float) -> None:
        """把等待拆成 ≤ tick_interval_s 的小段，每段后调用 throttle.tick；停止信号到达立即返回（20.6）。"""
        end = self._now() + max(0.0, duration)
        while not self.stop.is_set():
            t = self._now()
            if t >= end:
                break
            await self._wait_event(self.stop, self._chunk(t, end))
            t = self._now()
            self.throttle.tick(t)
            self._periodic(t)

    # ------------------------------------------------------------------ 周期任务
    def _ingest(self) -> None:
        """非阻塞读取补充队列（10.3）；异常只记录，不影响调度（10.5）。"""
        if self.queue is None:
            return
        try:
            self.index.ingest_pending(self.queue, INGEST_MAX)
        except Exception:  # noqa: BLE001
            log.exception("ingest_pending failed")

    def _periodic(self, t: float) -> None:
        if t >= self._next_expire:  # 4.3：每 60s 过期检查
            try:
                wall = self._wall()
                self.index.expire_due(wall if wall is not None else None)
            except Exception:  # noqa: BLE001
                log.exception("expire_due failed")
            self._next_expire = t + self.expire_interval_s
        if t >= self._next_walk:  # 16.2：第 60s 起每 60s 游走
            step = self.rng.gauss(0.0, self.walk_sigma)
            self.target_tpm = min(max(self.target_tpm + step, self.walk_low), self.walk_high)
            self.walk_history.append((t, self.target_tpm))
            self._next_walk = t + self.walk_interval_s

    # ------------------------------------------------------------------ 状态判定
    def blocked(self, t: float) -> bool:
        """HOLD / PROBE 未就绪 / 熔断 open 或 half_open 名额用尽 / 内容过滤暂停。"""
        max_new, _ = self.throttle.allowed_dispatch(t)
        return (
            max_new == 0
            or not self.breaker.allow_dispatch(t)  # 23.3 / 23.6
            or t < self.cf_window.pause_until  # 24.3
        )

    def _probing(self) -> bool:
        return self.throttle.state is ThrottleState.PROBE

    def _abort(self, probing: bool) -> bool:
        return self.stop.is_set() or self.blocked(self._now()) or probing != self._probing()

    def _breaker_limit_reached(self) -> bool:
        if self.max_breaker_opens is None:
            return False
        if getattr(self.breaker, "open_count", 0) >= self.max_breaker_opens:
            self.stop_reason = self.stop_reason or "circuit_open"
            self.stop.set()
            return True
        return False

    def effective_rates(self, t: float) -> tuple[float, float, float, bool]:
        """(eff_level, eff_tpm, eff_rpm, probing)（18.4 / 16.3）。"""
        _, thr_level = self.throttle.allowed_dispatch(t)
        probing = self._probing()
        if probing:
            eff = thr_level if thr_level > 0 else self.throttle.levels[0]
        else:
            t0 = self.t_start if self.t_start is not None else t
            eff = min(self.ramp.factor(t - t0), self.throttle.level)
        eff_tpm = self.target_tpm * eff
        eff_rpm = min(self.load.rpm_max, self.quota_rpm * 0.5) * eff
        return eff, eff_tpm, eff_rpm, probing

    # ------------------------------------------------------------------ 组装与获取
    def _assemble(self) -> Any | None:
        wall = self._wall()
        for _ in range(self.max_target_retries):
            target = self.length_sampler(self.rng, self.load)
            try:
                return self.assembler.assemble(target, self.rng, now=wall)
            except TargetUnreachable:
                self.target_unreachable += 1  # 重新采样目标长度
                continue
            except CorpusExhausted:
                self.stop_reason = self.stop_reason or "corpus_exhausted"  # 27.8
                self.stop.set()
                return None
        return None

    async def _acquire(self, cost: int, probing: bool) -> bool:
        """令牌桶 + 并发信号量；成功时持有一个信号量名额。中途需放弃时返回 False（不持有名额）。"""
        bucket = self.bucket
        assert bucket is not None
        while True:
            if self._abort(probing):
                return False
            if bucket.try_acquire(cost):
                break
            wait = bucket._wait_s(cost) if hasattr(bucket, "_wait_s") else BUCKET_STEP_S
            await self.sleep_ticking(min(max(wait, 1e-6), BUCKET_STEP_S))
        sem = self.semaphore
        while sem.locked():
            if self._abort(probing):
                return False
            self._slot_freed.clear()
            await self._wait_event(self._slot_freed, self._chunk(self._now()))
            t = self._now()
            self.throttle.tick(t)
            self._periodic(t)
        await sem.acquire()  # 未锁定时立即返回
        if self._abort(probing):
            sem.release()
            return False
        return True

    # ------------------------------------------------------------------ 派发
    def _dispatch(self, planned: Any, reservation: Any, probing: bool) -> None:
        now = self._now()
        cost = int(planned.input_tokens) + int(planned.max_tokens)
        self.ewma_request_tokens += self.load.ewma_alpha * (cost - self.ewma_request_tokens)  # 16.1
        ticket = self.throttle.note_dispatch(now, is_probe=probing)
        self.breaker.on_dispatch(now)
        self.dispatched += 1
        if probing:
            self.probes_dispatched += 1
        if self.metrics is not None:
            self.metrics.record_dispatch(planned, now, is_probe=probing)
        if self.on_dispatch is not None:
            self.on_dispatch(now, planned, ticket)
        task = asyncio.ensure_future(run_request(self.ctx, planned, reservation, ticket))
        self._tasks.add(task)
        self._idle.clear()
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        self._slot_freed.set()
        if not self._tasks:
            self._idle.set()
        if not task.cancelled() and task.exception() is not None:
            log.error("worker task failed", exc_info=task.exception())

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    # ------------------------------------------------------------------ 主循环
    async def run(self) -> RunSummary:
        t0 = self._now()
        self.t_start = t0  # 18.5：爬坡零点 = 主循环开始
        self._next_walk = t0 + self.walk_interval_s
        self._next_expire = t0 + self.expire_interval_s
        if self.bucket is None:
            eff, eff_tpm, eff_rpm, _ = self.effective_rates(t0)
            self.bucket = DualTokenBucket(
                eff_tpm, eff_rpm, self.load.bucket_burst_s, self.load.bucket_initial_fill_s, self.clock
            )
        if self.metrics is not None:
            self.metrics.mark_dispatch_start(t0)
        try:
            await self._loop()
        finally:
            self.t_stop = self._now()
            if self.metrics is not None:
                self.metrics.mark_dispatch_stop(self.t_stop)
        await self.drain(self.load.drain_timeout_s)
        return self.summary()

    async def _loop(self) -> None:
        stop = self.stop
        bucket = self.bucket
        assert bucket is not None
        while not stop.is_set():
            t = self._now()
            self._ingest()
            self._periodic(t)
            if self._breaker_limit_reached():
                break
            self.throttle.tick(t)
            if self.blocked(t):  # 19.4 / 20.2 / 23.3 / 24.3：按 ≤ 1s 步长轮询
                wake = max(self.breaker.retry_at, self.cf_window.pause_until, t + self.tick_s)
                await self.sleep_ticking(wake - t)
                continue

            eff, eff_tpm, eff_rpm, probing = self.effective_rates(t)
            self.eff_level = eff
            bucket.set_rates(eff_tpm, eff_rpm)  # 16.6：派发前更新速率

            if not probing:  # 探测请求不做泊松等待
                lam = min(eff_tpm / self.ewma_request_tokens, eff_rpm) / 60.0
                await self.sleep_ticking(self.rng.expovariate(lam) if lam > 0 else self.tick_s)
            if stop.is_set() or self.blocked(self._now()):
                continue
            if probing != self._probing():
                continue

            planned = self._assemble()
            if planned is None:
                if stop.is_set():
                    break
                await self.sleep_ticking(self.tick_s)  # 多次 TargetUnreachable：稍后重试
                continue

            reservation = await self.cost_guard.reserve(self.model, planned.input_tokens, planned.max_tokens)
            if reservation is None:  # 25.5
                self.stop_reason = self.stop_reason or "budget"
                stop.set()
                break

            cost = int(planned.input_tokens) + int(planned.max_tokens)
            if not await self._acquire(cost, probing):
                await self.cost_guard.release(reservation)
                self.released += 1
                continue
            self._dispatch(planned, reservation, probing)

    # ------------------------------------------------------------------ 收尾
    async def drain(self, timeout: float) -> bool:
        """等待在途请求最多 timeout 秒（27.4）；超时则取消剩余请求（worker 按预留结算）。"""
        deadline = self._now() + max(0.0, timeout)
        while self._tasks:
            t = self._now()
            if t >= deadline:
                break
            await self._wait_event(self._idle, self._chunk(t, deadline))
            self.throttle.tick(self._now())
        if self._tasks:
            self.drained = False
            await self.cancel_inflight()
        return self.drained

    async def cancel_inflight(self) -> int:
        """立即取消全部在途请求（drain 超时 / 二次 SIGINT，27.6）；返回取消数。"""
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        n = sum(1 for t in tasks if t.cancelled())
        self.cancelled += n
        return n

    def summary(self) -> RunSummary:
        return RunSummary(
            stop_reason=self.stop_reason,
            dispatched=self.dispatched,
            probes_dispatched=self.probes_dispatched,
            released=self.released,
            t_start=self.t_start if self.t_start is not None else 0.0,
            t_stop=self.t_stop,
            final_target_tpm=self.target_tpm,
            ewma_request_tokens=self.ewma_request_tokens,
            walk_history=list(self.walk_history),
            drained=self.drained,
            cancelled=self.cancelled,
            target_unreachable=self.target_unreachable,
            cf_pauses=self.cf_window.pauses,
        )

    def status_extra(self) -> dict[str, Any]:
        """供控制台 / CSV 快照的调度器字段。"""
        t = self._now()
        elapsed = t - self.t_start if self.t_start is not None else 0.0
        return {
            "target_tpm": self.target_tpm,
            "effective_level": self.eff_level,
            "ramp_level": self.ramp.factor(elapsed),
            "ramp": "done" if self.ramp.done(elapsed) else "ramping",
            "in_flight": self.in_flight,
            "ewma_request_tokens": self.ewma_request_tokens,
            "queue_len": self.queue.qsize() if self.queue is not None else 0,
            "cf_hits_60s": self.cf_window.count(t),
        }


__all__ = ["RunSummary", "Scheduler", "EWMA_INIT", "WALK_SIGMA"]
