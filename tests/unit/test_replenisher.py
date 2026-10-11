"""Replenisher 单元测试（Requirements 10.2, 10.4, 10.6, 10.7, 10.8）。"""

from __future__ import annotations

import asyncio
import logging
from random import Random
from types import SimpleNamespace
from typing import Any

import pytest

from loadtest.clock import FakeClock
from loadtest.corpus.gemini_client import GeminiBudgetExhausted
from loadtest.corpus.planner import SupplyPlanner
from loadtest.corpus.replenisher import Replenisher, passage_from_row
from loadtest.corpus.store import CorpusStore, Passage, PoolStats
from loadtest.corpus.taxonomy import TopicTaxonomy
from loadtest.corpus.warmup import GenOutcome
from loadtest.settings import CorpusSettings, LoadSettings

# ---------------------------------------------------------------------------
# 假组件
# ---------------------------------------------------------------------------


class FakeStore:
    def __init__(self, stats: PoolStats | None = None) -> None:
        self._stats = stats or PoolStats()
        self.rows: dict[int, dict[str, Any]] = {}
        self.stats_calls = 0

    def add(self, pid: int, spec: Any = None) -> None:
        self.rows[pid] = {
            "id": pid,
            "category": getattr(spec, "category", "cat"),
            "lang": getattr(spec, "lang", "zh"),
            "token_len": 300,
            "use_count": 0,
            "expires_at": "2030-01-01T00:00:00+00:00" if pid % 2 else None,
            "text": f"text {pid}",
            "length_bucket": getattr(spec, "length_bucket", "M"),
        }

    async def stats(self) -> PoolStats:
        self.stats_calls += 1
        return self._stats

    async def get_passage(self, pid: int) -> dict[str, Any] | None:
        return self.rows.get(pid)

    async def grounding_calls_today(self) -> int:
        return 0


class FakePlanner:
    """总是给出任务；记录 reset / 归还。"""

    def __init__(self) -> None:
        self.resets = 0
        self.issued = 0
        self.returned: list[str | None] = []

    def reset(self, stats: PoolStats) -> None:
        self.resets += 1

    def set_grounding_used(self, n: int) -> None:
        pass

    def next_spec(self, rng: Random, stats: PoolStats | None = None) -> Any:
        self.issued += 1
        return SimpleNamespace(category="cat", lang="zh", length_bucket="M", n=self.issued)

    def note_rejected(self, spec: Any, reason: str | None) -> None:
        self.returned.append(reason)


class FakeGenerator:
    """按脚本返回结果；``script(i, spec)`` 返回 GenOutcome / 异常实例 / 协程。"""

    def __init__(self, store: FakeStore, clock: FakeClock, script: Any, planner: Any = None) -> None:
        self.store = store
        self.clock = clock
        self.script = script
        self.planner = planner
        self.calls: list[tuple[float, Any]] = []
        self.store_lock = asyncio.Lock()

    async def generate_and_store(self, spec: Any) -> GenOutcome:
        i = len(self.calls)
        self.calls.append((self.clock.now(), spec))
        await asyncio.sleep(0)
        res = self.script(i, spec)
        if asyncio.iscoroutine(res):
            res = await res
        if isinstance(res, BaseException):
            raise res
        if res.status == "inserted":
            self.store.add(res.passage_id, spec)
            if self.planner is not None:
                self.planner.note_inserted(spec)
        return res


def inserted(i: int) -> GenOutcome:
    return GenOutcome("inserted", passage_id=i + 1, model="m", attempts=1)


def make(script: Any, *, queue_size: int = 1024, concurrency: int = 1, store: FakeStore | None = None,
         planner: Any = None, gen_planner: Any = None):
    clock = FakeClock()
    store = store or FakeStore()
    planner = planner or FakePlanner()
    gen = FakeGenerator(store, clock, script, planner=gen_planner)
    q: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
    rep = Replenisher(gen, store, planner, clock=clock, rng=Random(1), queue=q, concurrency=concurrency)
    return rep, gen, store, planner, clock, q


def stop_after(n: int, stop: asyncio.Event, inner: Any):
    def script(i: int, spec: Any):
        if i + 1 >= n:
            stop.set()
        return inner(i, spec)
    return script


# ---------------------------------------------------------------------------
# 10.2 队列满：段落仍入库、通知被丢弃、put_nowait 不阻塞
# ---------------------------------------------------------------------------


async def test_queue_full_drops_notification_but_keeps_passage() -> None:
    stop = asyncio.Event()
    rep, gen, store, _, _, q = make(stop_after(3, stop, lambda i, s: inserted(i)), queue_size=1)
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert len(gen.calls) == 3
    assert sorted(store.rows) == [1, 2, 3]  # 全部入库
    assert q.qsize() == 1
    first = q.get_nowait()
    assert isinstance(first, Passage) and first.id == 1
    assert rep.stats.inserted == 3 == rep.added
    assert rep.stats.delivered == 1
    assert rep.stats.dropped_notifications == 2


async def test_put_nowait_never_blocks_with_full_queue() -> None:
    """队列已满且无人消费时，补充任务仍持续推进（不在 put 上等待）。"""
    stop = asyncio.Event()
    rep, gen, _, _, _, q = make(stop_after(50, stop, lambda i, s: inserted(i)), queue_size=1, concurrency=4)
    q.put_nowait("occupied")
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert len(gen.calls) >= 50
    assert q.qsize() == 1 and q.get_nowait() == "occupied"
    assert rep.stats.dropped_notifications == rep.stats.inserted


async def test_passage_built_from_row() -> None:
    stop = asyncio.Event()
    rep, _, _, _, _, q = make(stop_after(2, stop, lambda i, s: inserted(i)))
    await rep.run(stop)
    p1, p2 = q.get_nowait(), q.get_nowait()
    assert p1.id == 1 and p1.expires_at is not None and p1.length_bucket == "M"
    assert p2.id == 2 and p2.expires_at is None
    assert p1.text == "text 1" and p1.lang == "zh" and p1.token_len == 300


async def test_passage_from_real_store_row(tmp_path) -> None:
    clock = FakeClock()
    store = await CorpusStore(tmp_path / "c.db", clock).open()
    try:
        res = await store.insert_text(
            category="tech", subtopic="t", genre="g", lang="en", token_len=820, length_bucket="M",
            text="hello world " * 40, gen_model="m", grounded=True, sources=("https://example.com",),
        )
        row = await store.get_passage(res.id)
        p = passage_from_row(row)
        assert p.id == res.id and p.category == "tech" and p.lang == "en"
        assert p.token_len == 820 and p.use_count == 0 and p.length_bucket == "M"
        assert p.expires_at is not None and p.expires_at > clock.wall()
    finally:
        await store.close()


async def test_rejected_and_failed_not_delivered() -> None:
    stop = asyncio.Event()

    def inner(i: int, s: Any) -> GenOutcome:
        return GenOutcome("rejected", "duplicate") if i % 2 == 0 else GenOutcome("error", "error")

    rep, _, _, _, _, q = make(stop_after(4, stop, inner))
    await rep.run(stop)
    assert q.empty()
    assert rep.stats.rejected == 2 and rep.stats.failed == 2 and rep.stats.inserted == 0


# ---------------------------------------------------------------------------
# 10.4 异常：捕获、记录，5s 后继续
# ---------------------------------------------------------------------------


async def test_exception_is_caught_and_retried_after_5s(caplog: pytest.LogCaptureFixture) -> None:
    stop = asyncio.Event()

    def inner(i: int, s: Any):
        if i == 0:
            return RuntimeError("boom")
        return inserted(i)

    rep, gen, _, _, _, q = make(stop_after(3, stop, inner))
    with caplog.at_level(logging.ERROR, logger="loadtest.corpus.replenisher"):
        await asyncio.wait_for(rep.run(stop), timeout=5)  # 不向外抛出
    assert rep.stats.errors == 1
    assert len(gen.calls) == 3
    assert gen.calls[1][0] - gen.calls[0][0] >= 5.0
    assert any(r.exc_info for r in caplog.records)
    assert q.qsize() == 2


async def test_store_exception_also_contained() -> None:
    stop = asyncio.Event()

    class BadStore(FakeStore):
        n = 0

        async def stats(self) -> PoolStats:
            self.n += 1
            if self.n == 1:
                raise OSError("db locked")
            return PoolStats()

    rep, gen, _, _, _, _ = make(stop_after(1, stop, lambda i, s: inserted(i)), store=BadStore())
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert rep.stats.errors == 1
    assert len(gen.calls) == 1
    assert gen.calls[0][0] >= 5.0


# ---------------------------------------------------------------------------
# 10.6 预算耗尽：1 条告警，此后零调用
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("concurrency", [1, 4])
async def test_budget_exhausted_stops_calls_with_one_warning(
    caplog: pytest.LogCaptureFixture, concurrency: int
) -> None:
    stop = asyncio.Event()

    def script(i: int, s: Any):
        return GeminiBudgetExhausted() if i >= 2 else inserted(i)

    rep, gen, _, _, _, _ = make(script, concurrency=concurrency)
    with caplog.at_level(logging.WARNING, logger="loadtest.corpus.replenisher"):
        await asyncio.wait_for(rep.run(stop), timeout=5)  # stop 从未置位，仍自行结束
    assert rep.budget_exhausted
    n = len(gen.calls)
    if concurrency == 1:
        assert n == 3
    else:
        assert n <= 2 + concurrency  # 预算拒绝时已在途的调用之外没有新调用
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "budget" in r.getMessage()]
    assert len(warns) == 1
    # 再次运行也不会再调用 Gemini
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert len(gen.calls) == n


# ---------------------------------------------------------------------------
# 10.8 停止后不再发起调用
# ---------------------------------------------------------------------------


async def test_no_calls_when_stopped_before_start() -> None:
    stop = asyncio.Event()
    stop.set()
    rep, gen, _, _, _, _ = make(lambda i, s: inserted(i), concurrency=4)
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert gen.calls == []


async def test_no_new_calls_after_stop() -> None:
    stop = asyncio.Event()
    release = asyncio.Event()

    async def slow(i: int) -> GenOutcome:
        await release.wait()
        return inserted(i)

    rep, gen, _, _, _, q = make(lambda i, s: slow(i), concurrency=4)
    task = asyncio.create_task(rep.run(stop))
    for _ in range(20):
        await asyncio.sleep(0)
    in_flight = len(gen.calls)
    assert 1 <= in_flight <= 4
    stop.set()
    release.set()
    await asyncio.wait_for(task, timeout=5)
    assert len(gen.calls) == in_flight
    assert rep.stats.inserted == in_flight  # 在途调用照常完成并入库
    assert q.qsize() == in_flight


async def test_stop_interrupts_backoff() -> None:
    stop = asyncio.Event()

    def inner(i: int, s: Any):
        stop.set()
        return RuntimeError("boom")

    rep, gen, _, _, clock, _ = make(inner)
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert len(gen.calls) == 1
    assert clock.now() < 5.0  # 停止后不再等待完整退避


async def test_idle_planner_sleeps_and_refreshes() -> None:
    stop = asyncio.Event()

    class IdlePlanner(FakePlanner):
        def next_spec(self, rng: Random, stats: PoolStats | None = None) -> Any:
            self.issued += 1
            if self.issued >= 3:
                stop.set()
            return None

    rep, gen, _, planner, clock, _ = make(lambda i, s: inserted(i), planner=IdlePlanner())
    await asyncio.wait_for(rep.run(stop), timeout=5)
    assert gen.calls == []
    assert planner.resets >= 3  # 启动 + 每次空闲后强制刷新
    assert clock.now() >= 20.0


# ---------------------------------------------------------------------------
# 10.7 替换优先级（真实 SupplyPlanner）
# ---------------------------------------------------------------------------


def make_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        load=LoadSettings(), corpus=CorpusSettings(), gemini=SimpleNamespace(grounding_daily_limit=300)
    )


async def test_quarantine_then_expired_then_deficit() -> None:
    tax = TopicTaxonomy.load()
    plain = [c.id for c in tax.categories() if not c.grounding]
    qcell = ("zh", plain[0], "M")
    ecell = ("en", plain[1], "S")
    planner = SupplyPlanner(tax, make_cfg(), target_size=100)
    store = FakeStore(PoolStats(quarantined_cells={qcell: 2}, expired_cells={ecell: 1}))
    stop = asyncio.Event()
    rep, gen, _, _, _, q = make(
        stop_after(6, stop, lambda i, s: inserted(i)), store=store, planner=planner, gen_planner=planner
    )
    await asyncio.wait_for(rep.run(stop), timeout=5)
    cells = [(s.lang, s.category, s.length_bucket) for _, s in gen.calls]
    assert cells[:2] == [qcell, qcell]
    assert cells[2] == ecell
    assert len(cells) == 6
    assert q.qsize() == 6
    assert planner.inflight == 0


async def test_refresh_does_not_repeat_replacements() -> None:
    """周期刷新后（隔离计数仍在库存统计中）不会重复生成已替换的段落。"""
    tax = TopicTaxonomy.load()
    plain = [c.id for c in tax.categories() if not c.grounding]
    qcell = ("zh", plain[0], "M")
    planner = SupplyPlanner(tax, make_cfg(), target_size=100)
    store = FakeStore(PoolStats(quarantined_cells={qcell: 1}))
    stop = asyncio.Event()
    rep, gen, _, _, _, _ = make(
        stop_after(4, stop, lambda i, s: inserted(i)), store=store, planner=planner, gen_planner=planner
    )
    rep.refresh_interval_s = 0.0  # 每步都刷新
    await asyncio.wait_for(rep.run(stop), timeout=5)
    cells = [(s.lang, s.category, s.length_bucket) for _, s in gen.calls]
    assert cells[0] == qcell
    assert store.stats_calls >= 4
    assert planner._q_remaining.get(qcell, 0) == 0
    assert planner._q_consumed[qcell] == 1  # 只发出过一次替换任务


# ---------------------------------------------------------------------------
# 组装（无网络：backend 为占位对象，不会被调用）
# ---------------------------------------------------------------------------


async def test_build_replenisher_assembles_own_store_and_limiter(tmp_path) -> None:
    from loadtest.corpus.replenisher import build_replenisher
    from loadtest.settings import CostSettings, GeminiSettings

    (tmp_path / "sa.json").write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        load=LoadSettings(),
        corpus=CorpusSettings(),
        cost=CostSettings(),
        gemini=GeminiSettings(credentials_file=tmp_path / "sa.json", project="p"),
    )
    clock = FakeClock()
    rep, closer = await build_replenisher(
        settings, tmp_path / "corpus.db", clock, run_id="r1", rng=Random(0), backend=object()
    )
    try:
        assert rep.concurrency == 4
        assert rep.queue is not None and rep.queue.maxsize == 1024
        client = rep.generator.client
        assert client.limiter.rpm_cap == 30  # 独立限流器（replenisher=True）
        assert rep.store is rep.generator.store
        assert rep.store_lock is rep.generator.store_lock
    finally:
        await closer()
    assert rep.store._db is None  # closer 关闭了独立连接
