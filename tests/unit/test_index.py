from __future__ import annotations

import asyncio
from random import Random

from loadtest.clock import FakeClock
from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage


def P(pid, lang="zh", token_len=1000, use_count=0, expires_at=None, bucket=None):
    return Passage(id=pid, category="c", lang=lang, token_len=token_len, use_count=use_count,
                   expires_at=expires_at, text=f"t{pid}", length_bucket=bucket)


def test_bucketing_from_token_len_and_explicit():
    idx = PassageIndex([P(1, token_len=600), P(2, token_len=900), P(3, token_len=1300),
                        P(4, token_len=600, bucket="L"), P(5, lang="en", token_len=1600)], clock=lambda: 0)
    assert idx.inventory() == {("zh", "S"): 1, ("zh", "M"): 1, ("zh", "L"): 2, ("en", "L"): 1}


def test_prefers_bucket_containing_near():
    idx = PassageIndex([P(1, token_len=600), P(2, token_len=900), P(3, token_len=1300)], clock=lambda: 0)
    rng = Random(1)
    assert all(idx.pick("zh", 650, set(), rng).id == 1 for _ in range(20))
    assert all(idx.pick("zh", 1000, set(), rng).id == 2 for _ in range(20))
    assert all(idx.pick("zh", 1450, set(), rng).id == 3 for _ in range(20))


def test_fallback_by_distance():
    idx = PassageIndex([P(1, token_len=600), P(2, token_len=900), P(3, token_len=1300)], clock=lambda: 0)
    rng = Random(2)
    # near=1300 → L(1) 排除 → 最近 M → 再 S
    assert idx.pick("zh", 1300, {3}, rng).id == 2
    assert idx.pick("zh", 1300, {3, 2}, rng).id == 1
    assert idx.pick("zh", 1300, {1, 2, 3}, rng) is None
    # near 在所有区间之外（< 500）→ 最近 S
    assert idx.pick("zh", 100, set(), rng).id == 1
    assert idx.pick("en", 1000, set(), rng) is None


def test_weighting_prefers_low_use_count():
    idx = PassageIndex([P(1, use_count=0), P(2, use_count=9)], per_run_max_uses=10**9, clock=lambda: 0)
    rng = Random(3)
    counts = {1: 0, 2: 0}
    for _ in range(4000):
        counts[idx.pick("zh", 1000, set(), rng).id] += 1
    # 权重 1 : 0.1 → 期望约 91% 选中 1
    assert 0.87 < counts[1] / 4000 < 0.95


def test_weighting_includes_run_uses_and_alpha():
    idx = PassageIndex([P(1), P(2)], per_run_max_uses=100, alpha=2.0, clock=lambda: 0)
    for _ in range(2):
        idx.mark_used(2)  # 权重 1/9 vs 1
    rng = Random(4)
    hits = sum(idx.pick("zh", 1000, set(), rng).id == 1 for _ in range(4000))
    assert 0.86 < hits / 4000 < 0.94  # 期望 0.9


def test_pick_does_not_mark_used_and_cap():
    idx = PassageIndex([P(1)], per_run_max_uses=2, clock=lambda: 0)
    rng = Random(0)
    idx.pick("zh", 1000, set(), rng)
    assert idx.run_uses == {}
    idx.mark_used(1)
    idx.mark_used(1)
    assert idx.pick("zh", 1000, set(), rng) is None
    assert idx.run_uses == {1: 2}


def test_remove_is_permanent_and_keeps_run_uses():
    idx = PassageIndex([P(1), P(2), P(3)], clock=lambda: 0)
    idx.mark_used(2)
    assert idx.remove(2)
    assert 2 not in idx and len(idx) == 2 and idx.size == 2
    assert not idx.ingest(P(2))
    assert idx.run_uses == {2: 1}
    rng = Random(0)
    assert all(idx.pick("zh", 1000, set(), rng).id in {1, 3} for _ in range(50))


def test_default_now_uses_clock_wall():
    clock = FakeClock(start=0.0, wall_offset=1000.0)
    idx = PassageIndex([P(1, expires_at=1005.0), P(2)], clock=clock)
    rng = Random(0)
    clock.advance(10)  # wall = 1010
    assert all(idx.pick("zh", 1000, set(), rng).id == 2 for _ in range(20))
    assert idx.expire_due() == 1 and 1 not in idx


def test_deterministic_with_same_seed():
    ps = [P(i, token_len=500 + i * 10, use_count=i % 3) for i in range(1, 80)]
    seqs = []
    for _ in range(2):
        idx = PassageIndex(ps, clock=lambda: 0)
        rng = Random(42)
        seq = []
        for k in range(100):
            p = idx.pick("zh", 500 + (k * 37) % 1000, set(), rng)
            if p is None:
                break
            idx.mark_used(p.id)
            seq.append(p.id)
        seqs.append(seq)
    assert seqs[0] == seqs[1]


def test_ingest_pending_non_blocking_and_capped():
    async def run():
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        idx = PassageIndex([], clock=lambda: 0)
        assert idx.ingest_pending(q) == 0  # 空队列立即返回
        for i in range(1, 101):
            q.put_nowait(P(i))
        q.put_nowait(P(1))  # 重复
        assert idx.ingest_pending(q) == 64
        assert idx.ingest_pending(q, max_items=64) == 36  # 37 条中 1 条重复
        assert q.empty() and len(idx) == 100

    asyncio.run(run())


def test_ingest_rejects_expired():
    idx = PassageIndex([], clock=lambda: 100.0)
    assert not idx.ingest(P(1, expires_at=100.0))
    assert idx.ingest(P(2, expires_at=100.5))
