"""Feature: azure-gpt-load-tester, Property 17: 去重

∀ 文本 x，插入 x 后 ``is_duplicate(x)`` 为真；对 x 做空白 / 标点规范化变换后仍为真；
互不相关的长文本之间几乎不会被误判为重复。
（Validates: Requirements 6.1, 6.2, 6.3）
"""
from __future__ import annotations

import random
import unicodedata

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.corpus.dedup import Deduplicator, normalize, sha256_of

PUNCT_CATS = ["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"]
WHITESPACE = [" ", "\t", "\n", "\r", "\u3000", "\u00a0", "\u2003"]

# 正文字符：中英字母、数字、空白与标点混合
body_chars = st.one_of(
    st.characters(min_codepoint=0x4E00, max_codepoint=0x9FFF),
    st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    st.sampled_from(WHITESPACE),
    st.sampled_from(list("，。！？；：、“”‘’（）《》,.!?;:'\"()-[]")),
)
texts = st.text(alphabet=body_chars, max_size=400)
noise = st.one_of(
    st.sampled_from(WHITESPACE),
    st.characters(categories=PUNCT_CATS),
)


def _is_noise(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch).startswith("P")


@st.composite
def text_and_variant(draw: st.DrawFn) -> tuple[str, str]:
    """返回 (x, x')：x' 删除 x 中部分空白 / 标点，并在随机位置插入新的空白 / 标点。"""
    x = draw(texts)
    out: list[str] = []
    for ch in x:
        if _is_noise(ch) and draw(st.booleans()):
            continue  # 删除
        out.append(ch)
    inserts = draw(st.lists(st.tuples(st.integers(0, len(out)), noise), max_size=40))
    for pos, ch in sorted(inserts, key=lambda t: t[0], reverse=True):
        out.insert(pos, ch)
    return x, "".join(out)


@settings(deadline=None, max_examples=200)
@given(texts)
def test_added_text_is_duplicate(x: str) -> None:
    d = Deduplicator()
    d.add(x)
    assert d.is_duplicate(x)
    m = d.check(x)
    assert m is not None and m.kind == "exact" and m.distance == 0


@settings(deadline=None, max_examples=200)
@given(text_and_variant(), st.sampled_from([None, "zh", "en"]))
def test_whitespace_punctuation_variant_is_duplicate(pair: tuple[str, str], lang: str | None) -> None:
    x, variant = pair
    assert normalize(x) == normalize(variant)
    assert sha256_of(x) == sha256_of(variant)
    d = Deduplicator()
    d.add(x, lang=lang)  # type: ignore[arg-type]
    assert d.is_duplicate(variant)
    assert d.is_duplicate(variant, lang=lang)  # type: ignore[arg-type]


@settings(deadline=None, max_examples=100)
@given(st.lists(texts, min_size=1, max_size=8), st.data())
def test_any_inserted_text_stays_duplicate(xs: list[str], data: st.DataObject) -> None:
    """后续插入其他文本不影响先前文本的重复判定。"""
    d = Deduplicator()
    for i, x in enumerate(xs):
        d.add(x, passage_id=i)
    x = data.draw(st.sampled_from(xs))
    assert d.is_duplicate(x)


# --------------------------------------------------------------------------- 低误报（健全性）


def _random_en(rng: random.Random, vocab: list[str], n_words: int) -> str:
    words = [rng.choice(vocab) for _ in range(n_words)]
    sents = [" ".join(words[i : i + 12]).capitalize() + "." for i in range(0, n_words, 12)]
    return " ".join(sents)


def _random_zh(rng: random.Random, n_chars: int) -> str:
    chars = [chr(rng.randint(0x4E00, 0x9FA5)) for _ in range(n_chars)]
    return "，".join("".join(chars[i : i + 15]) for i in range(0, n_chars, 15)) + "。"


@settings(deadline=None, max_examples=20)
@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_unrelated_long_texts_rarely_duplicate(seed: int) -> None:
    rng = random.Random(seed)
    vocab = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9))) for _ in range(2000)]
    docs = [_random_en(rng, vocab, 300) for _ in range(30)] + [_random_zh(rng, 600) for _ in range(30)]
    d = Deduplicator()
    false_positives = 0
    for i, doc in enumerate(docs):
        if d.is_duplicate(doc):
            false_positives += 1
        d.add(doc, passage_id=i)
    # 随机 64 位指纹汉明距离 ≤ 3 的概率约 2.4e-15，60 篇文本应几乎为 0
    assert false_positives <= 1
