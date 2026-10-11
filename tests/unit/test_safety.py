"""SafetyFilter 单元测试：每条规则的通过 / 拒绝样例，以及多规则同时失败时只返回首个原因。

（Requirements 5.1–5.9, 5.12）
"""
from __future__ import annotations

import enum
import sys
import types

import pytest

from loadtest.corpus.safety import FilterResult, SafetyFilter, char_ratios, contains_date
from loadtest.corpus.taxonomy import TopicSpec
from loadtest.tokens import count

ZH_SENTENCES = [
    "城市公共交通的规划需要兼顾效率与公平，既要缩短通勤时间，也要照顾偏远社区的出行需求。",
    "研究人员发现，合理的绿地布局能够显著降低夏季街区的地表温度，并改善居民的睡眠质量。",
    "在数字化转型过程中，许多中小企业首先面临的问题并不是技术，而是流程梳理与人员培训。",
    "图书馆近年来逐渐从借阅场所转变为社区学习中心，举办讲座、展览和亲子阅读活动。",
    "农业生产中引入滴灌技术后，用水量明显下降，作物产量却保持稳定甚至略有提高。",
    "对于初学者来说，养成每天固定时间练习的习惯，往往比一次性投入大量时间更加有效。",
    "博物馆通过线上导览和互动展项，让更多观众能够跨越距离限制，了解珍贵的历史文物。",
    "气象部门提醒，季节交替时昼夜温差较大，市民应根据天气变化及时增减衣物。",
    "一项长期跟踪调查显示，规律的体育锻炼与更好的情绪状态和专注能力之间存在关联。",
    "在团队协作中，清晰的分工和及时的沟通能够减少重复劳动，提高整体工作效率。",
    "新能源汽车的普及带动了充电设施建设，也对城市电网的调度能力提出了更高要求。",
    "传统手工艺在年轻设计师的参与下焕发新生，许多作品兼具实用价值与审美趣味。",
    "教育专家认为，培养学生提出问题的能力，与传授知识本身同样重要，甚至更为关键。",
    "随着人口结构变化，社区养老服务逐渐受到重视，日间照料中心在各地陆续建立。",
    "旅游目的地的可持续发展，需要在吸引游客与保护生态环境之间找到合适的平衡点。",
    "科普写作的难点在于既要保证内容准确，又要让普通读者读得懂、愿意读下去。",
    "垃圾分类政策实施以后，回收资源的利用率有所提升，居民的环保意识也在逐步增强。",
]

EN_SENTENCES = [
    "Urban transit planning must balance efficiency with fairness, shortening commutes while serving remote neighborhoods.",
    "Researchers found that well placed green spaces can noticeably lower summer street temperatures and improve sleep.",
    "During digital transformation, many small firms struggle less with technology than with process design and training.",
    "Public libraries have gradually evolved from lending rooms into community learning centers with talks and exhibitions.",
    "After farms adopted drip irrigation, water use dropped sharply while crop yields stayed stable or even rose slightly.",
    "For beginners, practicing at a fixed time every day is often more effective than occasional marathon sessions.",
    "Museums now offer online tours and interactive displays, letting distant visitors explore rare historical artifacts.",
    "Forecasters remind residents that temperature swings are large when seasons change, so layered clothing helps.",
    "A long term survey suggests that regular exercise is associated with better mood and stronger concentration.",
    "In team projects, a clear division of labor and timely communication reduce duplicated effort and wasted hours.",
    "The spread of electric vehicles has accelerated charger construction and placed new demands on the power grid.",
    "Traditional crafts are finding new life as young designers create pieces that are both practical and beautiful.",
    "Education experts argue that teaching students to ask good questions matters as much as transferring knowledge.",
    "As populations age, community elder care services receive more attention and day centers open across regions.",
    "Sustainable tourism requires a careful balance between attracting visitors and protecting fragile ecosystems.",
    "The hard part of science writing is staying accurate while keeping ordinary readers interested until the end.",
    "Since sorting rules took effect, recycling rates have improved and residents have become more environmentally aware.",
]

PER_PARA = 5


def make_passage(lang: str, target: int, *, extra: str = "") -> str:
    """生成 token 数 ≥ target 的多段落正文；段落互不相同（17 为素数，相邻段落的句子窗口不同）。

    ``extra`` 会作为额外句子追加到第一段末尾。
    """
    pool = ZH_SENTENCES if lang == "zh" else EN_SENTENCES
    sep = "" if lang == "zh" else " "
    paras: list[str] = []
    p = 0
    while True:
        sents = [pool[(p * 3 + j) % len(pool)] for j in range(PER_PARA)]
        paras.append(sep.join(sents))
        p += 1
        body = "\n\n".join([paras[0] + (sep + extra if extra else "")] + paras[1:])
        if len(paras) >= 3 and count(body) >= target:
            return body


def spec(lang: str = "zh", grounding: bool = False) -> TopicSpec:
    return TopicSpec(
        category="daily_news" if grounding else "technology",
        subtopic="城市交通" if lang == "zh" else "urban transit",
        genre="说明文",
        lang=lang,  # type: ignore[arg-type]
        grounding=grounding,
        length_bucket="M",
        target_len=1000,
    )


BLOCK = ["炸药", "赌博", "bomb", "casino", "hate speech"]


@pytest.fixture(scope="module")
def sf() -> SafetyFilter:
    return SafetyFilter(BLOCK)


@pytest.fixture(scope="module")
def zh_text() -> str:
    t = make_passage("zh", 700)
    assert 500 <= count(t) <= 1500
    return t


@pytest.fixture(scope="module")
def en_text() -> str:
    t = make_passage("en", 700)
    assert 500 <= count(t) <= 1500
    return t


# --------------------------------------------------------------------------- 通过


def test_clean_passages_pass(sf: SafetyFilter, zh_text: str, en_text: str) -> None:
    for lang, text in (("zh", zh_text), ("en", en_text)):
        r = sf.check(text, spec(lang), "STOP")
        assert r == FilterResult(True, None, count(text), sf.bucket_of(count(text)))


# --------------------------------------------------------------------------- finish_reason


def test_safety_finish_reason(sf: SafetyFilter, zh_text: str) -> None:
    assert sf.check(zh_text, spec(), "SAFETY").reason == "safety_block"
    assert sf.check(zh_text, spec(), "safety").reason == "safety_block"


@pytest.mark.parametrize("empty", ["", "   \n\n  ", None])
def test_empty_text_is_safety_block(sf: SafetyFilter, empty: str | None) -> None:
    r = sf.check(empty, spec(), "STOP")
    assert not r.ok and r.reason == "safety_block"


@pytest.mark.parametrize("fr", ["MAX_TOKENS", "RECITATION", "OTHER", "length", None, ""])
def test_non_stop_finish_is_structure(sf: SafetyFilter, zh_text: str, fr: object) -> None:
    assert sf.check(zh_text, spec(), fr).reason == "structure"


class FinishReason(enum.Enum):
    FINISH_REASON_UNSPECIFIED = 0
    STOP = 1
    MAX_TOKENS = 2
    SAFETY = 3


@pytest.mark.parametrize(
    ("fr", "expected"),
    [
        ("stop", None),
        ("Stop", None),
        ("FinishReason.STOP", None),
        (FinishReason.STOP, None),
        (FinishReason.SAFETY, "safety_block"),
        (FinishReason.MAX_TOKENS, "structure"),
    ],
)
def test_finish_reason_forms(sf: SafetyFilter, zh_text: str, fr: object, expected: str | None) -> None:
    assert sf.check(zh_text, spec(), fr).reason == expected


# --------------------------------------------------------------------------- length


def test_too_short_and_too_long(sf: SafetyFilter) -> None:
    short = make_passage("zh", 100)
    assert count(short) < 500
    r = sf.check(short, spec(), "STOP")
    assert (r.ok, r.reason, r.token_len, r.length_bucket) == (False, "length", count(short), None)

    long_ = make_passage("en", 1600)
    assert count(long_) > 1500
    assert sf.check(long_, spec("en"), "STOP").reason == "length"


@pytest.mark.parametrize(
    ("n", "reason", "bucket"),
    [
        (499, "length", None),
        (500, None, "S"),
        (799, None, "S"),
        (800, None, "M"),
        (1199, None, "M"),
        (1200, None, "L"),
        (1500, None, "L"),
        (1501, "length", None),
    ],
)
def test_length_boundaries_with_given_token_len(
    sf: SafetyFilter, zh_text: str, n: int, reason: str | None, bucket: str | None
) -> None:
    r = sf.check(zh_text, spec(), "STOP", token_len=n)
    assert (r.reason, r.length_bucket, r.token_len) == (reason, bucket, n)


@pytest.mark.parametrize(("target", "bucket"), [(600, "S"), (1000, "M"), (1300, "L")])
def test_real_texts_bucketed(sf: SafetyFilter, target: int, bucket: str) -> None:
    text = make_passage("en", target)
    n = count(text)
    lo, hi = sf.buckets[bucket]
    assert lo <= n <= hi, n  # 生成器步长小于桶宽，落在目标桶内
    r = sf.check(text, spec("en"), "STOP")
    assert r.ok and r.length_bucket == bucket and r.token_len == n


def test_custom_thresholds_and_buckets(zh_text: str) -> None:
    f = SafetyFilter([], min_tokens=10, max_tokens=50, buckets={"S": (10, 20), "M": (21, 35), "L": (36, 50)})
    assert f.check(zh_text, spec(), "STOP", token_len=30).length_bucket == "M"
    assert f.check(zh_text, spec(), "STOP", token_len=51).reason == "length"


# --------------------------------------------------------------------------- language


def test_language_mismatch(sf: SafetyFilter, zh_text: str, en_text: str) -> None:
    assert sf.check(en_text, spec("zh"), "STOP").reason == "language"
    assert sf.check(zh_text, spec("en"), "STOP").reason == "language"


def test_en_cjk_threshold(sf: SafetyFilter, en_text: str) -> None:
    nonspace = sum(1 for c in en_text if not c.isspace())
    # 约 1% CJK：通过
    few = en_text + " " + "城" * (nonspace // 100)
    assert char_ratios(few)[0] < 0.02
    assert sf.check(few, spec("en"), "STOP").ok
    # 约 3% CJK：拒绝
    many = en_text + " " + "城" * (nonspace * 3 // 100)
    assert char_ratios(many)[0] >= 0.02
    assert sf.check(many, spec("en"), "STOP").reason == "language"


def test_en_alpha_threshold(sf: SafetyFilter, en_text: str) -> None:
    nonspace = sum(1 for c in en_text if not c.isspace())
    digits = "\n" + " ".join(["1234567890"] * (nonspace // 30))  # 约 +33% 数字 → 字母 < 80%
    assert char_ratios(en_text + digits)[1] < 0.80
    assert sf.check(en_text + digits, spec("en"), "STOP").reason == "language"


def test_zh_cjk_threshold(sf: SafetyFilter, zh_text: str) -> None:
    cjk, _ = char_ratios(zh_text)
    assert cjk >= 0.60
    nonspace = sum(1 for c in zh_text if not c.isspace())
    mixed = zh_text + "\n" + "abcdefghij" * (nonspace // 10)  # 约 +100% 拉丁字母 → CJK < 60%
    assert char_ratios(mixed)[0] < 0.60
    assert sf.check(mixed, spec("zh"), "STOP", token_len=900).reason == "language"


# --------------------------------------------------------------------------- refusal


@pytest.mark.parametrize(
    "phrase",
    ["作为AI，我只能提供一般性的信息。", "作为一个人工智能，我不具备个人观点。", "我无法确认这一点。", "抱歉，我不能继续撰写。"],
)
def test_zh_refusal(sf: SafetyFilter, phrase: str) -> None:
    text = make_passage("zh", 700, extra=phrase)
    assert sf.check(text, spec("zh"), "STOP").reason == "refusal"


@pytest.mark.parametrize(
    "phrase",
    [
        "I can't help with that request.",
        "I cannot verify those figures.",
        "I'm sorry, but this topic is outside my scope.",
        "I’m sorry, but this topic is outside my scope.",
        "As an AI, I do not hold opinions.",
        "As an AI language model, I must note several caveats.",
    ],
)
def test_en_refusal(sf: SafetyFilter, phrase: str) -> None:
    text = make_passage("en", 700, extra=phrase)
    assert sf.check(text, spec("en"), "STOP").reason == "refusal"


# --------------------------------------------------------------------------- blocklist


def test_zh_blocklist_substring(sf: SafetyFilter) -> None:
    text = make_passage("zh", 700, extra="有人在网上宣传非法赌博网站。")
    assert sf.check(text, spec("zh"), "STOP").reason == "blocklist"


@pytest.mark.parametrize("phrase", ["A Bomb scare closed the station.", "The CASINO opened late.", "Hate Speech laws vary."])
def test_en_blocklist_word_boundary_case_insensitive(sf: SafetyFilter, phrase: str) -> None:
    text = make_passage("en", 700, extra=phrase)
    assert sf.check(text, spec("en"), "STOP").reason == "blocklist"


def test_en_blocklist_does_not_match_inside_words(sf: SafetyFilter) -> None:
    text = make_passage("en", 700, extra="His bombastic speech about casinos2 and bombs was dull.")
    # bombastic / bombs / casinos2 均不是独立单词 bomb / casino
    assert sf.check(text, spec("en"), "STOP").ok


def test_empty_blocklist_allows_everything(zh_text: str) -> None:
    assert SafetyFilter([]).check(zh_text + "炸药", spec(), "STOP").ok


# --------------------------------------------------------------------------- structure


def test_too_few_paragraphs(sf: SafetyFilter, zh_text: str) -> None:
    paras = [p for p in zh_text.split("\n\n") if p]
    two = paras[0] + "\n\n" + "".join(paras[1:])
    assert sf.check(two, spec(), "STOP").reason == "structure"
    one = "".join(paras)
    assert sf.check(one, spec(), "STOP").reason == "structure"


def test_single_newlines_separate_paragraphs(sf: SafetyFilter, zh_text: str) -> None:
    assert sf.check(zh_text.replace("\n\n", "\n"), spec(), "STOP").ok


@pytest.mark.parametrize("heading", ["# 标题", "## 小节", "  ### 缩进标题"])
def test_markdown_heading(sf: SafetyFilter, zh_text: str, heading: str) -> None:
    assert sf.check(heading + "\n" + zh_text, spec(), "STOP").reason == "structure"


def test_duplicate_paragraph_after_normalization(sf: SafetyFilter, zh_text: str) -> None:
    first = zh_text.split("\n\n")[0]
    variant = first.replace("，", ", ").replace("。", "")  # 仅空白 / 标点不同
    text = zh_text + "\n\n" + variant
    assert sf.check(text, spec(), "STOP", token_len=1000).reason == "structure"


# --------------------------------------------------------------------------- grounding


def test_grounding_requires_source_and_date(sf: SafetyFilter) -> None:
    dated = make_passage("zh", 700, extra="据二〇二六年报道，该政策已于2026年3月5日正式实施。")
    nodate = make_passage("zh", 700)
    g = spec("zh", grounding=True)
    assert sf.check(dated, g, "STOP", sources=["https://example.com/a"]).ok
    assert sf.check(dated, g, "STOP", sources=[]).reason == "grounding"
    assert sf.check(nodate, g, "STOP", sources=["https://example.com/a"]).reason == "grounding"
    # 非 grounded 规格不检查来源与日期
    assert sf.check(nodate, spec("zh"), "STOP").ok


def test_grounding_en(sf: SafetyFilter) -> None:
    dated = make_passage("en", 700, extra="The council voted on March 3, 2026.")
    assert sf.check(dated, spec("en", True), "STOP", sources=({"uri": "x"},)).ok


@pytest.mark.parametrize(
    "s",
    [
        "2026-10-11",
        "2026/3/5",
        "2026年3月5日",
        "2026 年 10 月 11 日",
        "3月5日",
        "12月31号",
        "March 3",
        "on Sept. 14th",
        "3 March 2026",
        "21st of October",
    ],
)
def test_contains_date_positive(s: str) -> None:
    assert contains_date(f"xx {s} yy")


@pytest.mark.parametrize("s", ["2026年", "13月5日", "version 2026-13-40", "Mayor 3", "March", "第3月"])
def test_contains_date_negative(s: str) -> None:
    assert not contains_date(s)


# --------------------------------------------------------------------------- 顺序：只返回首个原因


def test_first_failure_only(sf: SafetyFilter) -> None:
    bad_en = "I'm sorry, but the casino story is short.\n\nSecond."  # en、短、拒答、黑名单、段落不足
    # SAFETY 优先于一切
    assert sf.check(bad_en, spec("zh", True), "SAFETY").reason == "safety_block"
    # 非 STOP 优先于长度
    assert sf.check(bad_en, spec("zh", True), "MAX_TOKENS").reason == "structure"
    # 长度优先于语言 / 拒答 / 黑名单
    assert sf.check(bad_en, spec("zh", True), "STOP").reason == "length"

    long_en = make_passage("en", 700, extra="I'm sorry, but the casino story is incomplete.")
    # 语言优先于拒答 / 黑名单 / grounding
    assert sf.check(long_en, spec("zh", True), "STOP").reason == "language"
    # 拒答优先于黑名单 / grounding
    assert sf.check(long_en, spec("en", True), "STOP").reason == "refusal"

    block_only = make_passage("en", 700, extra="The casino closed.")
    two_paras = block_only.split("\n\n")[0] + "\n\n" + " ".join(block_only.split("\n\n")[1:])
    # 黑名单优先于结构 / grounding
    assert sf.check(two_paras, spec("en", True), "STOP").reason == "blocklist"
    # 结构优先于 grounding
    clean_two = make_passage("en", 700)
    clean_two = clean_two.split("\n\n")[0] + "\n\n" + " ".join(clean_two.split("\n\n")[1:])
    assert sf.check(clean_two, spec("en", True), "STOP").reason == "structure"


# --------------------------------------------------------------------------- default()


def test_default_uses_lazy_load_blocklist(monkeypatch: pytest.MonkeyPatch, zh_text: str) -> None:
    calls: list[object] = []

    def fake_load_blocklist(path=None):  # noqa: ANN001, ANN202
        calls.append(path)
        return frozenset({"交通"})

    mod = types.ModuleType("loadtest.corpus.prompts")
    mod.load_blocklist = fake_load_blocklist  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "loadtest.corpus.prompts", mod)

    f = SafetyFilter.default()
    assert calls == [None]
    assert f.blocklist == frozenset({"交通"})
    assert f.check(zh_text, spec(), "STOP").reason == "blocklist"


def test_default_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from loadtest.settings import CorpusSettings

    mod = types.ModuleType("loadtest.corpus.prompts")
    mod.load_blocklist = lambda path=None: frozenset()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "loadtest.corpus.prompts", mod)
    cs = CorpusSettings(passage_min_tokens=600, passage_max_tokens=1400)
    f = SafetyFilter.default(settings=cs)
    assert (f.min_tokens, f.max_tokens) == (600, 1400)
