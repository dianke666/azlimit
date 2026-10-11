"""Replenisher：压测运行期间的后台语料补充（设计组件 11，Requirement 10）。

- 独立 asyncio 任务；``concurrency``（≤ 4）个 worker 共享一个 SupplyPlanner；
  使用独立的 GeminiRateLimiter（replenisher=True，≤ 30 RPM）与独立的 CorpusStore 连接（10.1）。
- 优先级由 SupplyPlanner.next_spec 给出：隔离替换 > 过期替换 > 缺口（10.7）。
  planner 在启动时、每 ``refresh_interval_s`` 秒以及无任务可发时按最新库存 ``reset``。
- 新段落入库后读取完整行构造 :class:`Passage`，``put_nowait`` 到队列；队列满则丢弃通知，
  段落已入库，下次运行加载（10.2）。
- 任何异常在任务内捕获、记录、计数，``error_backoff_s``（5s）后继续，不向调度器传播（10.4）。
- GeminiBudgetExhausted：记录 1 条告警，本次运行内所有 worker 不再调用 Gemini（10.6）。
- 停止信号置位后不再发起新的生成调用；所有等待以 ≤ 1s 分段并检查停止信号（10.8）。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from random import Random
from typing import Any, Awaitable, Callable

from loadtest.clock import Clock
from loadtest.corpus.gemini_client import GeminiBudgetExhausted
from loadtest.corpus.store import Passage, from_iso

log = logging.getLogger(__name__)

MAX_CONCURRENCY = 4  # Requirement 10.1


@dataclass
class ReplenisherStats:
    tasks: int = 0  # 发出的生成任务数
    inserted: int = 0
    rejected: int = 0
    failed: int = 0  # 生成器返回 status="error"（Gemini 调用失败，已在客户端内重试）
    errors: int = 0  # 补充任务内部捕获的异常
    dropped_notifications: int = 0  # 队列满而丢弃的通知（段落已入库）
    delivered: int = 0  # 成功放入队列的段落
    budget_exhausted: bool = False


def passage_from_row(row: Any) -> Passage:
    """``CorpusStore.get_passage`` 返回的行 → :class:`Passage`。"""
    exp = row["expires_at"]
    if isinstance(exp, str):
        expires_at: float | None = from_iso(exp)
    elif exp is None:
        expires_at = None
    else:
        expires_at = float(exp)
    keys = row.keys() if hasattr(row, "keys") else ()
    return Passage(
        id=int(row["id"]),
        category=row["category"],
        lang=row["lang"],
        token_len=int(row["token_len"]),
        use_count=int(row["use_count"] or 0),
        expires_at=expires_at,
        text=row["text"],
        length_bucket=row["length_bucket"] if "length_bucket" in keys else None,
    )


class Replenisher:
    def __init__(
        self,
        generator: Any,
        store: Any,
        planner: Any,
        *,
        clock: Clock,
        rng: Random,
        queue: asyncio.Queue | None = None,
        concurrency: int = MAX_CONCURRENCY,
        error_backoff_s: float = 5.0,
        refresh_interval_s: float = 60.0,
        idle_sleep_s: float = 10.0,
        store_lock: asyncio.Lock | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if concurrency > MAX_CONCURRENCY:
            log.warning("replenisher concurrency %d capped to %d", concurrency, MAX_CONCURRENCY)
        self.generator = generator
        self.store = store
        self.planner = planner
        self.clock = clock
        self.rng = rng
        self.queue = queue
        self.concurrency = min(int(concurrency), MAX_CONCURRENCY)
        self.error_backoff_s = float(error_backoff_s)
        self.refresh_interval_s = float(refresh_interval_s)
        self.idle_sleep_s = float(idle_sleep_s)
        # 与生成器共用同一把锁：同一 aiosqlite 连接上的读写串行化
        self.store_lock: asyncio.Lock = store_lock or getattr(generator, "store_lock", None) or asyncio.Lock()
        self.stats = ReplenisherStats()
        self._refresh_lock = asyncio.Lock()
        self._last_refresh: float | None = None
        self._grounding_loaded = False
        self._stop: asyncio.Event | None = None

    # ------------------------------------------------------------------ 属性

    @property
    def added(self) -> int:
        """本次运行新增段落数（runs.passages_added）。"""
        return self.stats.inserted

    @property
    def budget_exhausted(self) -> bool:
        return self.stats.budget_exhausted

    def _stopped(self) -> bool:
        return self.stats.budget_exhausted or (self._stop is not None and self._stop.is_set())

    # ------------------------------------------------------------------ 辅助

    async def _sleep(self, seconds: float) -> None:
        """可被停止信号打断的等待（≤ 1s 分段）。"""
        deadline = self.clock.now() + max(0.0, seconds)
        while not self._stopped():
            left = deadline - self.clock.now()
            if left <= 0:
                return
            await self.clock.sleep(min(1.0, left))

    async def _refresh(self, force: bool = False) -> None:
        async with self._refresh_lock:
            now = self.clock.now()
            if (
                not force
                and self._last_refresh is not None
                and now - self._last_refresh < self.refresh_interval_s
            ):
                return
            async with self.store_lock:
                if not self._grounding_loaded and hasattr(self.store, "grounding_calls_today"):
                    used = await self.store.grounding_calls_today()
                    if hasattr(self.planner, "set_grounding_used"):
                        self.planner.set_grounding_used(used)
                    self._grounding_loaded = True
                stats = await self.store.stats()
            self.planner.reset(stats)
            self._last_refresh = self.clock.now()

    def _notify(self, passage: Passage) -> None:
        assert self.queue is not None
        try:
            self.queue.put_nowait(passage)
            self.stats.delivered += 1
        except asyncio.QueueFull:
            self.stats.dropped_notifications += 1  # 已入库，下次运行加载

    async def _deliver(self, pid: int) -> None:
        if self.queue is None:
            return
        async with self.store_lock:
            row = await self.store.get_passage(pid)
        if row is None:
            log.warning("replenisher: inserted passage %s not found", pid)
            return
        self._notify(passage_from_row(row))

    def _on_budget(self) -> None:
        if not self.stats.budget_exhausted:
            self.stats.budget_exhausted = True
            log.warning("gemini budget exhausted; replenisher stopped")

    # ------------------------------------------------------------------ 主循环

    async def _step(self) -> None:
        await self._refresh()
        if self._stopped():
            return
        spec = self.planner.next_spec(self.rng)
        if spec is None:
            await self._sleep(self.idle_sleep_s)
            if not self._stopped():
                await self._refresh(force=True)
            return
        if self._stopped():  # 取到任务后、调用前再次检查（10.8）
            self.planner.note_rejected(spec, "cancelled")
            return
        self.stats.tasks += 1
        try:
            out = await self.generator.generate_and_store(spec)
        except GeminiBudgetExhausted:
            self._on_budget()
            return
        if out.status == "inserted":
            self.stats.inserted += 1
            if out.passage_id is not None:
                await self._deliver(out.passage_id)
        elif out.status == "rejected":
            self.stats.rejected += 1
        else:
            self.stats.failed += 1

    async def _worker(self, idx: int) -> None:
        while not self._stopped():
            try:
                await self._step()
            except asyncio.CancelledError:
                raise
            except GeminiBudgetExhausted:
                self._on_budget()
            except Exception:  # noqa: BLE001 - 10.4：任务内捕获，不终止压测
                self.stats.errors += 1
                log.exception("replenisher worker %d error", idx)
                await self._sleep(self.error_backoff_s)

    async def run(self, stop: asyncio.Event, out: asyncio.Queue | None = None) -> None:
        """运行到 ``stop`` 置位或 Gemini 预算耗尽；从不抛出普通异常。"""
        if out is not None:
            self.queue = out
        self._stop = stop
        tasks = [
            asyncio.create_task(self._worker(i), name=f"replenisher-{i}") for i in range(self.concurrency)
        ]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for r in results:
            if isinstance(r, Exception):  # pragma: no cover - worker 已捕获所有异常
                log.error("replenisher worker ended with %r", r)


# --------------------------------------------------------------------------- 组装


async def build_replenisher(
    settings: Any,
    db_path: Any,
    clock: Clock,
    *,
    run_id: str,
    rng: Random,
    queue: asyncio.Queue | None = None,
    cost_guard: Any = None,
    backend: Any = None,
) -> tuple[Replenisher, Callable[[], Awaitable[None]]]:
    """组装 Replenisher（独立 store 连接、独立限流器）。返回 ``(replenisher, closer)``；
    ``closer()`` 刷写 Gemini 账本并关闭 store 连接。"""
    from loadtest.corpus.gemini_client import GeminiClient, VertexBackend
    from loadtest.corpus.gemini_limiter import GeminiRateLimiter
    from loadtest.corpus.model_chain import ModelChain
    from loadtest.corpus.planner import SupplyPlanner
    from loadtest.corpus.safety import SafetyFilter
    from loadtest.corpus.store import CorpusStore
    from loadtest.corpus.taxonomy import TopicTaxonomy
    from loadtest.cost.guard import CostGuard
    from loadtest.corpus.warmup import CorpusGenerator

    gs = settings.gemini
    store = await CorpusStore(db_path, clock).open()
    try:
        taxonomy = TopicTaxonomy.load(grounding_categories=gs.grounding_categories)
        guard = cost_guard if cost_guard is not None else await CostGuard.gemini_from_settings(
            store, settings.cost, gs.project
        )
        chain = ModelChain(
            gs.model_chain,
            gs.failover_after_failures,
            gs.failback_after_s,
            budget_ratio_fn=guard.remaining_ratio,
            expensive=gs.expensive_models,
        )
        limiter = GeminiRateLimiter.from_settings(gs, clock, replenisher=True)
        backend = backend or VertexBackend(
            gs.project, gs.location, gs.credentials_file, thinking_levels=gs.thinking_levels
        )
        client = GeminiClient.from_settings(gs, backend, limiter, chain, guard, clock=clock, rng=rng)

        def utcnow() -> datetime:
            return datetime.fromtimestamp(clock.wall(), tz=timezone.utc)

        planner = SupplyPlanner(taxonomy, settings, utcnow=utcnow)
        safety = SafetyFilter.default(settings=settings.corpus)
        lock = asyncio.Lock()
        generator = CorpusGenerator(
            client, store, planner, taxonomy, safety, run_id=run_id, clock=clock, store_lock=lock
        )
        if queue is None:
            queue = asyncio.Queue(maxsize=gs.replenish_queue_size)
        rep = Replenisher(
            generator, store, planner,
            clock=clock, rng=rng, queue=queue,
            concurrency=gs.replenish_concurrency, store_lock=lock,
        )
    except BaseException:
        await store.close()
        raise

    async def closer() -> None:
        try:
            async with lock:
                await guard.flush(store, run_id)
        finally:
            await store.close()

    return rep, closer


__all__ = ["MAX_CONCURRENCY", "Replenisher", "ReplenisherStats", "build_replenisher", "passage_from_row"]
