"""Property 24: 过期段落不被选中；复用上限不被突破。

Validates: Requirements 4.2, 4.3, 4.4 (+ 9.1)
"""
from __future__ import annotations

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage

T0 = 1_700_000_000.0


@st.composite
def passage_sets(draw):
    n = draw(st.integers(min_value=0, max_value=40))
    out = []
    for i in range(n):
        expires = draw(st.one_of(st.none(), st.floats(min_value=T0 - 500, max_value=T0 + 500, allow_nan=False)))
        out.append(
            Passage(
                id=i + 1,
                category="daily_news" if expires is not None else "cooking",
                lang=draw(st.sampled_from(["zh", "en"])),
                token_len=draw(st.integers(min_value=500, max_value=1500)),
                use_count=draw(st.integers(min_value=0, max_value=10)),
                expires_at=expires,
                text=f"p{i}",
                length_bucket=draw(st.sampled_from([None, "S", "M", "L"])),
            )
        )
    return out


now_st = st.floats(min_value=T0 - 600, max_value=T0 + 600, allow_nan=False)


@settings(max_examples=200, deadline=None)
@given(passages=passage_sets(), load_now=now_st, pick_now=now_st, seed=st.integers(0, 2**32 - 1),
       near=st.integers(min_value=0, max_value=2000))
def test_pick_never_returns_expired(passages, load_now, pick_now, seed, near):
    # 构建时排除 expires_at <= load_now（Req 4.2）
    idx = PassageIndex([], per_run_max_uses=4)
    for p in passages:
        idx.ingest(p, now=load_now)
    for p in passages:
        if p.expires_at is not None and p.expires_at <= load_now:
            assert p.id not in idx

    rng = Random(seed)
    for _ in range(30):
        for lang in ("zh", "en"):
            got = idx.pick(lang, near, set(), rng, now=pick_now)
            if got is not None:
                assert got.lang == lang
                assert got.expires_at is None or got.expires_at > pick_now
                assert idx.run_uses.get(got.id, 0) < 4
                idx.mark_used(got.id)
    # 复用上限（Req 9.1）
    assert all(v <= 4 for v in idx.run_uses.values())


@settings(max_examples=200, deadline=None)
@given(passages=passage_sets(), t=now_st)
def test_expire_due_removes_all_due(passages, t):
    idx = PassageIndex(passages, clock=lambda: T0 - 10_000)  # 构建时全部未过期
    before = len(idx)
    removed = idx.expire_due(t)
    assert removed == before - len(idx)
    for p in passages:
        due = p.expires_at is not None and p.expires_at <= t
        assert (p.id in idx) == (not due)
    # 已过期移除的段落不能被重新加入
    for p in passages:
        if p.expires_at is not None and p.expires_at <= t:
            assert not idx.ingest(p, now=T0 - 10_000)


@settings(max_examples=100, deadline=None)
@given(passages=passage_sets(), seed=st.integers(0, 2**32 - 1), cap=st.integers(1, 5))
def test_reuse_cap_exhausts_then_none(passages, seed, cap):
    idx = PassageIndex(passages, per_run_max_uses=cap, clock=lambda: T0 - 10_000)
    rng = Random(seed)
    for lang in ("zh", "en"):
        n_lang = sum(1 for p in passages if p.lang == lang)
        picks = 0
        while True:
            got = idx.pick(lang, 1000, set(), rng, now=T0 - 10_000)
            if got is None:
                break
            idx.mark_used(got.id)
            picks += 1
            assert picks <= n_lang * cap
        assert picks == n_lang * cap
    assert all(v == cap for v in idx.run_uses.values())
