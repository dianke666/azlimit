"""TopicTaxonomy / TaskCatalog / prompts / blocklist 单元测试。

Requirements 2.1–2.4, 2.6, 3.3, 14.7。
"""

from __future__ import annotations

import copy
from datetime import date
from importlib import resources
from random import Random

import pytest
import yaml

from loadtest.corpus import prompts
from loadtest.corpus.prompts import build_prompt, load_blocklist, target_chars, target_words
from loadtest.corpus.taxonomy import (
    DEFAULT_LENGTH_BUCKETS,
    GENRES,
    TaxonomyError,
    TopicSpec,
    TopicTaxonomy,
)
from loadtest.load.tasks import REQUIRED_SUFFIX, TaskCatalog, TaskCatalogError

GROUNDING_IDS = {"daily_news", "daily_economy", "sports_events"}


@pytest.fixture(scope="module")
def tax() -> TopicTaxonomy:
    return TopicTaxonomy.load()


def _cat(cid: str, n: int = 10, *, grounding: bool = False, zh: str | None = None) -> dict:
    return {
        "id": cid,
        "zh": zh or f"类别{cid}",
        "en": f"Category {cid}",
        "weight": 1.0,
        "grounding": grounding,
        "subtopics": [{"zh": f"{cid}子话题{i}", "en": f"{cid} subtopic {i}"} for i in range(n)],
    }


# ====================================================================== taxonomy

def test_taxonomy_size(tax: TopicTaxonomy) -> None:
    cats = tax.categories()
    assert len(cats) >= 14
    for c in cats:
        assert 10 <= len(c.subtopics) <= 30, c.id
        assert set(c.genres) <= set(GENRES)


def test_only_designated_categories_are_grounding(tax: TopicTaxonomy) -> None:
    assert {c.id for c in tax.categories() if c.grounding} == GROUNDING_IDS


def test_small_valid_taxonomy_loads_without_core() -> None:
    t = TopicTaxonomy({"categories": [_cat("a"), _cat("b", 30)]}, require_core=False)
    assert [c.id for c in t.categories()] == ["a", "b"]


def test_invalid_taxonomy_lists_all_bad_categories() -> None:
    data = {
        "categories": [
            _cat("ok"),
            _cat("too_few", 9),
            _cat("too_many", 31),
            _cat("fake_grounding", grounding=True),   # 非 grounding 类别被标记为 grounding
            _cat("daily_news", grounding=True),       # 合法 grounding 类别
        ]
    }
    with pytest.raises(TaxonomyError) as ei:
        TopicTaxonomy(data, require_core=False)
    assert set(ei.value.categories) == {"too_few", "too_many", "fake_grounding"}


def test_require_core_rejects_missing_categories(tax: TopicTaxonomy) -> None:
    raw = yaml.safe_load(resources.files("loadtest.corpus").joinpath("taxonomy.yaml").read_text(encoding="utf-8"))
    data = copy.deepcopy(raw)
    removed = [c["zh"] for c in data["categories"] if c["zh"] in ("厨艺", "自然")]
    assert len(removed) == 2
    data["categories"] = [c for c in data["categories"] if c["zh"] not in removed]
    # 同时让一个现有类别的子话题数不合规
    data["categories"][0]["subtopics"] = data["categories"][0]["subtopics"][:5]
    bad_label = data["categories"][0]["id"]
    with pytest.raises(TaxonomyError) as ei:
        TopicTaxonomy(data, require_core=True)
    assert set(removed) | {bad_label} <= set(ei.value.categories)


def test_load_from_file_rejects(tmp_path) -> None:
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump({"categories": [_cat("x", 3)]}, allow_unicode=True), encoding="utf-8")
    with pytest.raises(TaxonomyError) as ei:
        TopicTaxonomy.load(p, require_core=False)
    assert ei.value.categories == ["x"]


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_sample_is_valid(tax: TopicTaxonomy, lang: str) -> None:
    rng = Random(42)
    seen_genres: set[str] = set()
    by_id = {c.id: c for c in tax.categories()}
    for _ in range(3000):
        s = tax.sample(rng, None, lang)
        assert isinstance(s, TopicSpec)
        assert s.lang == lang
        assert s.length_bucket in ("S", "M", "L")
        lo, hi = DEFAULT_LENGTH_BUCKETS[s.length_bucket]
        assert lo <= s.target_len <= hi
        assert s.genre in GENRES
        assert s.genre in by_id[s.category].genres
        assert s.grounding == (s.category in GROUNDING_IDS)
        names = {st.name(lang) for st in by_id[s.category].subtopics}
        assert s.subtopic in names
        seen_genres.add(s.genre)
    assert seen_genres == set(GENRES)


def test_sample_fixed_bucket_and_no_grounding(tax: TopicTaxonomy) -> None:
    rng = Random(7)
    for b in ("S", "M", "L"):
        for _ in range(300):
            s = tax.sample(rng, None, "zh", length_bucket=b, allow_grounding=False)
            assert s.length_bucket == b
            assert not s.grounding
            assert s.category not in GROUNDING_IDS
    with pytest.raises(ValueError):
        tax.sample(rng, None, "zh", category="daily_news", allow_grounding=False)
    with pytest.raises(ValueError):
        tax.sample(rng, None, "fr")  # type: ignore[arg-type]


def test_penalize_reduces_sampling() -> None:
    t = TopicTaxonomy({"categories": [_cat("a")]}, require_core=False)
    target = "a子话题0"
    t.penalize(target, 0.01)
    assert t.subtopic_weight(target) == pytest.approx(0.01)
    rng = Random(1)
    n = 5000
    hits = sum(t.sample(rng, None, "zh").subtopic == target for _ in range(n))
    # 未降权时期望 1/10；降权后期望 0.01/9.01 ≈ 0.11%
    assert hits / n < 0.01
    # 英文名同样可降权
    t.penalize("a subtopic 1", 0.5)
    assert t.subtopic_weight("a子话题1") == pytest.approx(0.5)
    with pytest.raises(KeyError):
        t.penalize("不存在")


# ====================================================================== tasks

def test_task_catalog_counts_and_suffix() -> None:
    cat = TaskCatalog.load()
    for lang in ("zh", "en"):
        assert cat.count(lang) >= 30
        for t in cat.for_lang(lang):
            assert t.lang == lang
            assert t.instruction.endswith(REQUIRED_SUFFIX[lang])
            assert t.system_prompt
    rng = Random(0)
    assert cat.sample(rng, "en").lang == "en"


def _tasks_dict(n: int = 30) -> dict:
    return {
        lang: {
            "system_prompt": "sys",
            "suffix": REQUIRED_SUFFIX[lang],
            "tasks": [{"id": f"{lang}{i}", "text": f"task {i}"} for i in range(n)],
        }
        for lang in ("zh", "en")
    }


def test_task_catalog_from_dict_valid() -> None:
    cat = TaskCatalog.from_dict(_tasks_dict())
    assert cat.count("zh") == 30 and cat.count("en") == 30


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["zh"]["tasks"].pop(),                         # 少于 30 条
        lambda d: d["en"].__setitem__("suffix", "Answer briefly."),  # 后缀错误
        lambda d: d.pop("en"),                                     # 缺语言段
        lambda d: d["zh"]["tasks"].append({"id": "zh0", "text": "dup"}),  # 重复 id
        lambda d: d["zh"].__setitem__("system_prompt", ""),        # 空 system_prompt
    ],
)
def test_task_catalog_from_dict_invalid(mutate) -> None:
    d = _tasks_dict()
    mutate(d)
    with pytest.raises(TaskCatalogError):
        TaskCatalog.from_dict(d)


def test_task_catalog_rejects_non_mapping() -> None:
    with pytest.raises(TaskCatalogError):
        TaskCatalog.from_dict([1, 2, 3])


# ====================================================================== prompts

def _spec(tax: TopicTaxonomy, lang: str, *, grounding: bool, target_len: int = 1000) -> TopicSpec:
    cat = next(c for c in tax.categories() if c.grounding == grounding)
    return TopicSpec(
        category=cat.id,
        subtopic=cat.subtopics[0].name(lang),
        genre=cat.genres[0],
        lang=lang,  # type: ignore[arg-type]
        grounding=grounding,
        length_bucket="M",
        target_len=target_len,
    )


@pytest.mark.parametrize("n", [500, 650, 799, 1000, 1350, 1500])
def test_length_conversion(tax: TopicTaxonomy, n: int) -> None:
    s = _spec(tax, "zh", grounding=False, target_len=n)
    assert target_chars(s) == (round(n * 1.0), round(n * 1.3))
    assert target_words(s) == round(n * 0.75)


def test_zh_prompt(tax: TopicTaxonomy) -> None:
    s = _spec(tax, "zh", grounding=False, target_len=1000)
    p = build_prompt(s, tax)
    assert "简体中文" in p
    assert "1000–1300" in p
    assert tax.category_label(s.category, "zh") in p
    assert s.subtopic in p
    assert tax.genre_label(s.genre, "zh") in p
    for kw in ("3–6 个自然段", "空一行", "公认事实", "不要编造", "Markdown", "表格", "加粗", "作为AI",
               "政治", "宗教", "暴力", "色情", "违法", "医疗诊断", "投资建议", "争议"):
        assert kw in p, kw
    assert "检索" not in p and "日期" not in p


def test_en_prompt(tax: TopicTaxonomy) -> None:
    s = _spec(tax, "en", grounding=False, target_len=1000)
    p = build_prompt(s, tax)
    assert "in English" in p
    assert "About 750 words" in p
    assert tax.category_label(s.category, "en") in p
    assert s.subtopic in p
    assert tax.genre_label(s.genre, "en") in p
    for kw in ("3–6 paragraphs", "blank line", "well-established facts", "never invent", "Markdown",
               "tables", "bold", "as an AI", "politics", "religion", "violence", "sexual",
               "illegal", "medical diagnosis", "investment advice", "controversial"):
        assert kw in p, kw
    assert "简体中文" not in p
    assert "news reports" not in p


def test_zh_en_templates_differ(tax: TopicTaxonomy) -> None:
    zh = build_prompt(_spec(tax, "zh", grounding=False), tax)
    en = build_prompt(_spec(tax, "en", grounding=False), tax)
    assert zh != en
    assert not any("\u4e00" <= ch <= "\u9fff" for ch in en.replace(_spec(tax, "en", grounding=False).subtopic, ""))


def test_grounding_block(tax: TopicTaxonomy) -> None:
    today = date(2025, 3, 7)
    zh = build_prompt(_spec(tax, "zh", grounding=True), tax, today=today)
    assert "2025-03-07" in zh and "3月7日" in zh
    for kw in ("检索到的最近几天的真实报道", "排除", "政治", "注明", "日期", "YYYY-MM-DD", "X月X日", "不作任何投资建议"):
        assert kw in zh, kw
    en = build_prompt(_spec(tax, "en", grounding=True), tax, today=today)
    assert "2025-03-07" in en and "March 7" in en
    for kw in ("news reports from the past few days", "exclude anything political", "date of each event",
               "YYYY-MM-DD", "Month D", "no investment advice"):
        assert kw in en, kw


def test_grounding_default_today(tax: TopicTaxonomy) -> None:
    p = build_prompt(_spec(tax, "en", grounding=True), tax)
    assert "Today is 20" in p


# ====================================================================== blocklist

def test_blocklist_loads() -> None:
    words = load_blocklist()
    assert len(words) >= 30
    assert all(w and not w.startswith("#") and w == w.strip() and w == w.lower() for w in words)
    # 中英文都有
    assert any(any("\u4e00" <= ch <= "\u9fff" for ch in w) for w in words)
    assert any(w.isascii() for w in words)
    # 不过度拦截常用词
    for common in ("选择", "比赛", "中药", "股票", "crisis", "heroine", "trumpet", "election"):
        assert not any(w in common for w in words), common


def test_blocklist_custom_file(tmp_path) -> None:
    p = tmp_path / "b.txt"
    p.write_text("# comment\n\n  Foo Bar  \n违禁词\n   # indented comment\n", encoding="utf-8")
    assert load_blocklist(p) == frozenset({"foo bar", "违禁词"})


def test_module_exports() -> None:
    assert set(prompts.__all__) == {"build_prompt", "load_blocklist", "target_chars", "target_words"}
