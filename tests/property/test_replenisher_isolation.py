"""Property 23: 补充任务不阻塞调度器。

∀ 补充任务行为（任意延迟、挂起、抛异常、队列满），调度器派发序列与关闭补充任务时相同
（相同 seed、相同初始 Passage_Index），``ingest_pending`` 从不 await，补充任务异常不会终止运行。
假补充任务与真实调度器在同一事件循环（FakeClock 离散事件模拟）中运行。

**Validates: Requirements 10.4, 10.5**
"""

from __future__ import annotations

import asyncio
import inspect
from random import Random
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage
from loadtest.load.assembler import CorpusExhausted
from loadtest.load.models import PlannedRequest
from loadtest.outcome import Outcome
from tests.property.test_global_window import Harness, SimClock, make_scheduler


def _pool(n: int = 120) -> list[Passage]:
    out = []
    for i in range(1, n + 1):
        lang = "zh" if i % 5 < 3 else "en"
        tl = 500 + (i * 37) % 1000
        out.append(Passage(id=i, category="c", lang=lang, token_len=tl, use_count=i % 3, expires_at=None,
                           text="x", length_bucket="S" if tl < 800 else ("M" if tl < 1200 else "L")))
    return out


class IndexAssembler:
    """依赖 PassageIndex 选取的简化组装器：索引内容变化会改变派发序列。"""

    def __init__(self, index: PassageIndex) -> None:
        self.index = index
        self.n = 0

    def assemble(self, target: int, rng: Random, now: float | None = None) -> PlannedRequest:
        lang = "zh" if rng.random() < 0.6 else "en"
        ids: list[int] = []
        total = 0
        while total < target - 400 and len(ids) < 4:
            p = self.index.pick(lang, near=max(1, (target - total) // max(1, 4 - len(ids))),
                                exclude=set(ids), rng=rng, now=now)
            if p is None:
                break
            ids.append(p.id)
            total += p.token_len
        if not ids:
            raise CorpusExhausted(lang)
        for pid in ids:
            self.index.mark_used(pid)
        self.n += 1
        return PlannedRequest(request_id=f"r{self.n}", lang=lang, messages=[], input_tokens=min(6000, max(1, total)),  # type: ignore[arg-type]
                              max_tokens=40, passage_ids=tuple(ids), task_id="t", target_tokens=target)


def _script(seed: int):
    rng = Random(seed)

    def script(planned: Any, now: float):
        u = rng.random()
        if u < 0.02:
            return rng.uniform(0.2, 2.0), Outcome.THROTTLED, 10.0
        return rng.uniform(0.2, 3.0), Outcome.OK, None

    return script


def _harness(seed: int, queue: asyncio.Queue | None) -> Harness:
    index = PassageIndex(_pool(), per_run_max_uses=1000, clock=lambda: 0.0)
    return make_scheduler(seed=seed, script=_script(seed), assembler=IndexAssembler(index), index=index,
                          replenish_queue=queue, clock=SimClock())


def _sequence(h: Harness) -> list[tuple[float, str, int, bool]]:
    return [(d.t, d.request_id, d.cost, d.is_probe) for d in h.dispatches]


def _fake_replenisher(kind: str, h: Harness, queue: asyncio.Queue, seed: int, errors: list[int]):
    """假补充任务：不向队列交付任何新段落（只可能塞入已在索引中的重复段落）。"""
    rng = Random(seed)
    clock = h.clock
    existing = _pool()[:5]

    async def hang() -> None:
        await asyncio.Event().wait()

    async def raising() -> None:
        while True:
            try:
                await clock.sleep(rng.uniform(0.0, 3.0))
                raise RuntimeError("gemini exploded")
            except RuntimeError:
                errors[0] += 1
                await clock.sleep(5.0)  # 10.4：捕获、记录，5 秒后继续

    async def crash() -> None:
        await clock.sleep(rng.uniform(0.0, 10.0))
        raise RuntimeError("unhandled replenisher crash")

    async def slow() -> None:
        while True:
            await clock.sleep(rng.expovariate(0.2))
            for _ in range(rng.randint(0, 50)):
                await asyncio.sleep(0)  # 让出事件循环，模拟忙碌

    async def full() -> None:
        while True:
            for p in existing:  # 队列满：put_nowait 抛 QueueFull，丢弃通知
                try:
                    queue.put_nowait(p)
                except asyncio.QueueFull:
                    pass
            await clock.sleep(rng.uniform(0.1, 2.0))

    return {"hang": hang, "raise": raising, "crash": crash, "slow": slow, "full": full}[kind]()


def test_ingest_pending_is_not_a_coroutine() -> None:
    assert not inspect.iscoroutinefunction(PassageIndex.ingest_pending)


def test_sequence_comparison_is_sensitive_to_delivered_passages() -> None:
    """对照：真正交付新段落时派发序列会改变（说明上面的等价比较有区分度）。"""
    base = _harness(1, None)
    base.run(60.0)
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    for i in range(1000, 1040):
        queue.put_nowait(Passage(id=i, category="c", lang="zh", token_len=1400, use_count=0, expires_at=None,
                                 text="x", length_bucket="L"))
    h = _harness(1, queue)
    h.run(60.0)
    assert any(pid >= 1000 for pid in h.sched.index.run_uses)
    assert h.sched.index.run_uses != base.sched.index.run_uses


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1), kind=st.sampled_from(["hang", "raise", "crash", "slow", "full"]))
def test_replenisher_does_not_affect_dispatch(seed: int, kind: str) -> None:
    duration = 90.0
    base = _harness(seed, None)
    base_summary = base.run(duration)
    assert base_summary.stop_reason is None
    expected = _sequence(base)
    assert expected

    queue: asyncio.Queue = asyncio.Queue(maxsize=3)
    h = _harness(seed, queue)
    errors = [0]
    summary = h.run(duration, lambda hh: [_fake_replenisher(kind, hh, queue, seed, errors)])

    assert summary.stop_reason is None  # 补充任务异常不终止运行
    assert summary.t_stop == base_summary.t_stop
    assert _sequence(h) == expected
    assert h.sched.index.run_uses == base.sched.index.run_uses
    if kind == "raise":
        assert errors[0] >= 1
