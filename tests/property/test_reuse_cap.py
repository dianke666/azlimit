"""Property 21: 单次运行复用上限。

∀ 模拟运行，段落出现次数 ≤ per_run_max_uses；无满足上限的候选时抛 CorpusExhausted 而不突破上限；
每次组装对所用段落（含补足片段来源）的 run_uses 恰好 +1，失败的组装不修改 run_uses。

**Validates: Requirements 9.1, 9.3, 9.5**
"""
from __future__ import annotations

from collections import Counter
from random import Random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.load.assembler import CorpusExhausted, TargetUnreachable
from tests.property.test_assembler_precision import make_assembler, make_index, synthetic_passages


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    cap=st.integers(min_value=1, max_value=4),
    pool_size=st.integers(min_value=6, max_value=24),
)
def test_reuse_cap_never_exceeded(seed: int, cap: int, pool_size: int) -> None:
    rng = Random(seed)
    pool = [p for p in synthetic_passages() if p.lang == "zh"][:pool_size] + \
           [p for p in synthetic_passages() if p.lang == "en"][:pool_size]
    idx = make_index(pool, per_run_max_uses=cap)
    asm = make_assembler(idx)
    seen: Counter[int] = Counter()
    failures = 0
    for _ in range(400):
        before = idx.run_uses
        try:
            req = asm.assemble(rng.randint(1500, 6000), rng)
        except CorpusExhausted:
            failures += 1
            assert idx.run_uses == before  # 失败不修改 run_uses
            if failures >= 40:
                break
            continue
        seen.update(req.passage_ids)
        after = idx.run_uses
        for pid in set(after) | set(before):
            delta = after.get(pid, 0) - before.get(pid, 0)
            assert delta == (1 if pid in req.passage_ids else 0)
    assert all(n <= cap for n in seen.values())
    assert dict(seen) == {k: v for k, v in idx.run_uses.items() if v}
    assert all(v <= cap for v in idx.run_uses.values())


def test_fully_exhausted_raises_corpus_exhausted() -> None:
    pool = synthetic_passages()[:30] + tuple(p for p in synthetic_passages() if p.lang == "en")[:30]
    idx = make_index(pool, per_run_max_uses=2)
    for p in pool:
        idx.mark_used(p.id)
        idx.mark_used(p.id)
    asm = make_assembler(idx)
    with pytest.raises(CorpusExhausted) as ei:
        asm.assemble(3000, Random(1))
    assert type(ei.value) is CorpusExhausted  # 无任何候选：不是 TargetUnreachable
    assert all(v == 2 for v in idx.run_uses.values())


def test_unreachable_target_does_not_break_cap() -> None:
    # 只有短段落：6000 无法用 ≤ 4 段达到
    short = tuple(p for p in synthetic_passages() if p.token_len < 800)
    idx = make_index(short)
    asm = make_assembler(idx)
    with pytest.raises(TargetUnreachable):
        asm.assemble(6000, Random(5))
    assert idx.run_uses == {}
