"""Task 1.5：token 计数与裁剪（Requirements 14.1, 14.8）。"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loadtest import tokens
from loadtest.tokens import (
    REPLY_PRIMING_TOKENS,
    TOKENS_PER_MESSAGE,
    TOKENS_PER_NAME,
    TokenCounter,
)


@pytest.fixture(scope="module")
def tc() -> TokenCounter:
    return TokenCounter()


ZH_SENTENCES = [
    "今天天气很好。",
    "我们去公园散步吧！",
    "你吃过午饭了吗？",
    "市场价格持续上涨；",
    "这本书非常有意思。",
    "他说：“明天见。”",
]
EN_SENTENCES = [
    "The cat sat on the mat. ",
    "Prices rose again today! ",
    "Did you see the game? ",
    "Pi is 3.14. ",
    'She said "see you tomorrow." ',
]
ZH_END = tuple("。！？；") + ("”",)
EN_END = (".", "!", "?", '."')


def _build(sentences: list[str], n: int) -> str:
    return "".join(sentences[i % len(sentences)] for i in range(n))


# ---------------------------------------------------------------------------
# 句子边界裁剪
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [10, 17, 33, 50, 81, 120, 199])
def test_trim_zh_sentence_boundary(tc, budget):
    text = _build(ZH_SENTENCES, 80)
    out = tc.trim_to_tokens(text, budget)
    assert tc.count(out) <= budget
    assert text.startswith(out)
    assert out.endswith(ZH_END), out[-10:]
    # 句子很短，边界必然落在 slack 内
    assert tc.count(out) >= budget - 8


@pytest.mark.parametrize("budget", [10, 17, 33, 50, 81, 120, 199])
def test_trim_en_sentence_boundary(tc, budget):
    text = _build(EN_SENTENCES, 80)
    out = tc.trim_to_tokens(text, budget)
    assert tc.count(out) <= budget
    assert text.startswith(out)
    assert out.endswith(EN_END), out[-20:]
    assert not out.endswith("3.")  # 不在小数点处切
    assert tc.count(out) >= budget - 8


def test_trim_mixed_language(tc):
    text = "".join(z + e for z, e in zip(ZH_SENTENCES * 10, EN_SENTENCES * 12))
    for budget in range(15, 200, 13):
        out = tc.trim_to_tokens(text, budget)
        assert tc.count(out) <= budget
        assert out.endswith(ZH_END + EN_END)


def test_trim_falls_back_to_hard_cut_without_boundary(tc):
    text = "这是一段没有任何句末标点的很长的文字" * 50
    out = tc.trim_to_tokens(text, 100)
    n = tc.count(out)
    assert n <= 100
    assert n >= 95  # 硬截断贴近预算
    assert text.startswith(out)


def test_trim_hard_cut_when_boundary_too_far(tc):
    # 唯一句子边界在开头附近（超出 slack），应按 token 截断而非退回到开头
    text = "开头一句。" + "后面是一段很长的没有标点的文字" * 50
    out = tc.trim_to_tokens(text, 80, slack=8)
    assert tc.count(out) <= 80
    assert tc.count(out) >= 72
    assert len(out) > len("开头一句。")


def test_trim_within_budget_unchanged(tc):
    text = "短文本。Short text."
    assert tc.trim_to_tokens(text, 1000) == text
    assert tc.trim_to_tokens(text, tc.count(text)) == text


def test_trim_non_positive_budget(tc):
    assert tc.trim_to_tokens("任何文本", 0) == ""
    assert tc.trim_to_tokens("any text", -5) == ""
    assert tc.trim_to_tokens("", 10) == ""


def test_module_level_trim_matches_instance(tc):
    text = _build(ZH_SENTENCES, 30)
    assert tokens.trim_to_tokens(text, 40) == tc.trim_to_tokens(text, 40)
    assert tokens.trim_to_tokens(text, 40, tc) == tc.trim_to_tokens(text, 40)


# ---------------------------------------------------------------------------
# 多字节字符
# ---------------------------------------------------------------------------


MULTIBYTE_TEXTS = [
    "𠀀𠀁𠀂𠀃𪚥𪚥龘龘靐靐" * 20,  # 生僻字 / 扩展区（一个字符常占多个 token）
    "👨‍👩‍👧‍👦🎉🔥🙂🇨🇳🇺🇸" * 20,  # emoji / ZWJ 序列 / 国旗
    "中文English混合😀テスト한국어" * 20,
]


@pytest.mark.parametrize("text", MULTIBYTE_TEXTS)
def test_trim_never_splits_multibyte(tc, text):
    total = tc.count(text)
    for budget in range(1, min(total, 120)):
        out = tc.trim_to_tokens(text, budget)
        assert tc.count(out) <= budget
        assert "\ufffd" not in out
        assert text.startswith(out)  # 结果是原文的字符前缀 ⇒ 不含半个字符
        out.encode("utf-8")  # 合法 UTF-8


def test_decode_drops_partial_bytes(tc):
    toks = tc.encode("𪚥")
    if len(toks) > 1:
        assert tc.decode(toks[:1]) in ("", "𪚥")
        assert "\ufffd" not in tc.decode(toks[:1])


# ---------------------------------------------------------------------------
# hypothesis：随机文本与预算
# ---------------------------------------------------------------------------


_alphabet = st.one_of(
    st.characters(codec="utf-8"),
    st.sampled_from(list("。！？；.!? \n”）中文字符😀𪚥")),
)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=st.text(alphabet=_alphabet, max_size=400), budget=st.integers(min_value=-2, max_value=200))
def test_trim_budget_property(text, budget):
    tc = tokens.default_counter()
    out = tc.trim_to_tokens(text, budget)
    assert tc.count(out) <= max(budget, 0)
    assert text.startswith(out)
    if 0 < budget and tc.count(text) <= budget:
        assert out == text


@settings(max_examples=200, deadline=None)
@given(
    sentences=st.lists(
        st.sampled_from(ZH_SENTENCES + EN_SENTENCES), min_size=5, max_size=60
    ),
    budget=st.integers(min_value=12, max_value=300),
)
def test_trim_sentence_property(sentences, budget):
    tc = tokens.default_counter()
    text = "".join(sentences)
    out = tc.trim_to_tokens(text, budget)
    assert tc.count(out) <= budget
    assert text.startswith(out)
    if tc.count(text) > budget:
        # 每句 ≤ 8 token，句子边界总在 slack 内 ⇒ 以句末标点结束
        assert out.rstrip().endswith(ZH_END + EN_END)


# ---------------------------------------------------------------------------
# 消息开销计数
# ---------------------------------------------------------------------------


def test_sentence_fixtures_are_short(tc):
    # test_trim_sentence_property 依赖每句 token 数 ≤ slack(8)
    for s in ZH_SENTENCES + EN_SENTENCES:
        assert tc.count(s) <= 8, s


def test_constants():
    assert TOKENS_PER_MESSAGE == 3
    assert REPLY_PRIMING_TOKENS == 3
    assert TOKENS_PER_NAME == 1


def test_count_basic(tc):
    assert tc.count("") == 0
    assert tc.count("hello") >= 1
    # 特殊 token 字面量按普通文本计数，不抛异常
    assert tc.count("<|endoftext|>") > 1


def test_count_messages_hand_computed(tc):
    system = "You are a helpful assistant. 你是一个助手。"
    user = "请总结以下段落。Summarize the passage."
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    expected = (
        REPLY_PRIMING_TOKENS
        + 2 * TOKENS_PER_MESSAGE
        + tc.count("system")
        + tc.count(system)
        + tc.count("user")
        + tc.count(user)
    )
    assert tc.count_messages(msgs) == expected
    assert tc.count_messages_full(msgs) == expected
    assert tokens.count_messages(msgs) == expected
    assert tokens.count_messages_full(msgs) == expected


def test_count_messages_with_name(tc):
    msgs = [{"role": "user", "name": "alice", "content": "hi"}]
    expected = (
        REPLY_PRIMING_TOKENS
        + TOKENS_PER_MESSAGE
        + tc.count("user")
        + tc.count("alice")
        + TOKENS_PER_NAME
        + tc.count("hi")
    )
    assert tc.count_messages(msgs) == expected


def test_count_messages_empty_and_none(tc):
    assert tc.count_messages([]) == REPLY_PRIMING_TOKENS
    msgs = [{"role": "user", "content": "x", "name": None}]
    assert tc.count_messages(msgs) == REPLY_PRIMING_TOKENS + TOKENS_PER_MESSAGE + tc.count("user") + tc.count("x")


def test_count_messages_triplet_form(tc):
    system, prefix, suffix = "系统提示词。", "请阅读：\n", "\n请用一句话回答。"
    expected = (
        REPLY_PRIMING_TOKENS
        + TOKENS_PER_MESSAGE
        + tc.count("system")
        + tc.count(system)
        + TOKENS_PER_MESSAGE
        + tc.count("user")
        + tc.count(prefix)
        + tc.count(suffix)
    )
    assert tc.count_messages(system, prefix, suffix) == expected
    assert tokens.count_messages(system, prefix, suffix) == expected
    # 空 system 不计 system 消息
    assert tc.count_messages("", prefix, suffix) == (
        REPLY_PRIMING_TOKENS + TOKENS_PER_MESSAGE + tc.count("user") + tc.count(prefix) + tc.count(suffix)
    )
    # 与完整消息复算的差异只来自前缀/段落/后缀拼接处的 BPE 合并，应很小
    passage = "这是一段测试段落。" * 30
    full = tc.count_messages_full(
        [{"role": "system", "content": system}, {"role": "user", "content": prefix + passage + suffix}]
    )
    assert abs(full - (tc.count_messages(system, prefix, suffix) + tc.count(passage))) <= 20


def test_count_messages_stable(tc):
    msgs = [{"role": "system", "content": "s" * 50}, {"role": "user", "content": "中文" * 100}]
    first = tc.count_messages(msgs)
    assert all(tc.count_messages(msgs) == first for _ in range(5))
    assert TokenCounter().count_messages(msgs) == first


def test_count_messages_bad_input(tc):
    with pytest.raises(TypeError):
        tc.count_messages([{"role": "user", "content": 123}])
    with pytest.raises(TypeError):
        tc.count_messages()
    with pytest.raises(TypeError):
        tc.count_messages("a", 1, "b")


@settings(max_examples=100, deadline=None)
@given(
    contents=st.lists(st.text(max_size=80), min_size=0, max_size=6),
)
def test_count_messages_additive_property(contents):
    tc = tokens.default_counter()
    msgs = [{"role": "user", "content": c} for c in contents]
    expected = REPLY_PRIMING_TOKENS + sum(TOKENS_PER_MESSAGE + tc.count("user") + tc.count(c) for c in contents)
    assert tc.count_messages_full(msgs) == expected
