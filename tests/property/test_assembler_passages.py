"""Property 4: 段落约束。

∀ 组装结果，1 ≤ len(passage_ids) ≤ 4，无重复，全部段落语言 == 请求语言，不包含隔离/移除段落；
初始段落数（不含补足片段）符合预算分档：< 2000 → 1–2，[2000, 4000) → 2–3，否则 3–4。

**Validates: Requirements 14.2, 14.3, 14.9**
"""
from __future__ import annotations

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property.test_assembler_precision import make_assembler, make_index, synthetic_passages


def _tier(budget: int) -> set[int]:
    if budget < 2000:
        return {1, 2}
    if budget < 4000:
        return {2, 3}
    return {3, 4}


@settings(max_examples=60, deadline=None)
@given(
    targets=st.lists(st.integers(min_value=1500, max_value=6000), min_size=1, max_size=6),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    removed_frac=st.floats(min_value=0.0, max_value=0.3),
)
def test_passage_constraints(targets: list[int], seed: int, removed_frac: float) -> None:
    rng = Random(seed)
    pool = synthetic_passages()
    by_id = {p.id: p for p in pool}
    idx = make_index()
    removed = {p.id for p in pool if rng.random() < removed_frac}
    for pid in removed:
        idx.remove(pid)  # 模拟已隔离 / 过期
    asm = make_assembler(idx)
    for t in targets:
        req = asm.assemble(t, rng)
        ids = req.passage_ids
        assert 1 <= len(ids) <= 4
        assert len(set(ids)) == len(ids)
        assert all(by_id[i].lang == req.lang for i in ids)
        assert not (set(ids) & removed)
        assert req.initial_passage_count in _tier(req.budget_tokens)
        assert req.initial_passage_count + req.filler_count == len(ids)
        # 所有段落正文（或其裁剪片段）确实出现在 user 消息里
        user = req.messages[1]["content"]
        for i in ids[:-1]:
            assert by_id[i].text in user
