"""Property 22: 语言配比。

∀ seed，组装 N = 10,000 个请求时 zh 占比 ∈ [0.58, 0.62]、en 占比 ∈ [0.38, 0.42]；语言只会是 zh / en。

组装器的语言选择就是每次 assemble 对 rng 的第一次消费 ``choose_lang(rng, lang_ratio)``；
10,000 次比例检验直接作用于 ``choose_lang``，并验证 assemble 与之逐次一致。

**Validates: Requirements 15.1, 15.2**
"""
from __future__ import annotations

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.load.assembler import choose_lang
from loadtest.settings import LoadSettings
from tests.property.test_assembler_precision import make_assembler, make_index, synthetic_passages


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_lang_ratio_10k(seed: int) -> None:
    rng = Random(seed)
    ratio = LoadSettings().lang_ratio
    langs = [choose_lang(rng, ratio) for _ in range(10_000)]
    assert set(langs) <= {"zh", "en"}
    zh = langs.count("zh") / len(langs)
    assert 0.58 <= zh <= 0.62
    assert 0.38 <= 1 - zh <= 0.42


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_assemble_lang_matches_choose_lang(seed: int) -> None:
    asm = make_assembler(make_index(), cfg=LoadSettings())
    by_id = {p.id: p for p in synthetic_passages()}
    rng = Random(seed)
    for _ in range(15):
        target = rng.randint(1500, 3000)
        expected = choose_lang(_replay(rng.getstate()), asm.lang_ratio)
        req = asm.assemble(target, rng)
        assert req.lang == expected
        assert req.lang in {"zh", "en"}
        assert {by_id[i].lang for i in req.passage_ids} == {req.lang}  # 不混合语言


def _replay(state) -> Random:
    r = Random()
    r.setstate(state)
    return r


def test_custom_ratio_respected() -> None:
    asm = make_assembler(lang_ratio={"zh": 0.0, "en": 1.0})
    rng = Random(1)
    assert {asm.assemble(1800, rng).lang for _ in range(20)} == {"en"}
