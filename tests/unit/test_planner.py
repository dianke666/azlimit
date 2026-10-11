"""SupplyPlanner 单元测试：缺口总和、逐级取整、优先级、grounding 上限、降权、过滤。"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from random import Random
from types import SimpleNamespace

import pytest

from loadtest.corpus.planner import (
    BUCKET_SHARES,
    BUCKETS,
    SupplyPlanner,
    check_for_run,
    largest_remainder,
)
from loadtest.corpus.store import PoolStats
from loadtest.corpus.taxonomy import TopicTaxonomy
from loadtest.settings import CorpusSettings, LoadSettings


def make_cfg(limit: int = 300, **corpus) -> SimpleNamespace:
    return SimpleNamespace(
        load=LoadSettings(),
        corpus=CorpusSettings(**corpus),
        gemini=SimpleNamespace(grounding_daily_limit=limit),
    )


@pytest.fixture
def tax() -> TopicTaxonomy:
    return TopicTaxonomy.load()


class Now:
    def __init__(self, t: datetime) -> None:
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def grounding_ids(tax: TopicTaxonomy) -> set[str]:
    return {c.id for c in tax.categories() if c.grounding}


def plain_ids(tax: TopicTaxonomy) -> list[str]:
    return [c.id for c in tax.categories() if not c.grounding]


def stats_from_cells(cells: dict[tuple[str, str, str], int], **kw) -> PoolStats:
    by_lang: Counter = Counter()
    by_bucket: Counter = Counter()
    by_cat: Counter = Counter()
    for (l, c, b), n in cells.items():
        by_lang[l] += n
        by_bucket[(l, b)] += n
        by_cat[(l, c)] += n
    return PoolStats(
        active=sum(cells.values()),
        by_lang=dict(by_lang),
        by_bucket=dict(by_bucket),
        by_category=dict(by_cat),
        by_cell=dict(cells),
        **kw,
    )


# ---------------------------------------------------------------------------
# 取整与缺口
# ---------------------------------------------------------------------------


def test_largest_remainder_sums_exactly() -> None:
    rng = Random(1)
    for _ in range(500):
        total = rng.randint(0, 50_000)
        ws = {i: rng.random() for i in range(rng.randint(1, 20))}
        out = largest_remainder(total, ws)
        assert sum(out.values()) == total
        s = sum(ws.values())
        for k, v in out.items():
            assert abs(v - total * ws[k] / s) < 1.0 + 1e-9


def test_cell_targets_each_level_sums_to_parent(tax: TopicTaxonomy) -> None:
    p = SupplyPlanner(tax, make_cfg())
    cat_w = tax.target_weights()
    for target in (0, 1, 7, 999, 8000, 15_000, 12_345):
        t = p.cell_targets(target)
        assert sum(t.values()) == target
        lang_t = largest_remainder(target, p.lang_ratio)
        for lang, lt in lang_t.items():
            assert sum(n for (l, _, _), n in t.items() if l == lang) == lt
            cat_t = largest_remainder(lt, cat_w)
            for cid, ct in cat_t.items():
                cell = {b: t[(lang, cid, b)] for b in BUCKETS}
                assert sum(cell.values()) == ct
                assert cell == largest_remainder(ct, BUCKET_SHARES)


def test_plan_sum_equals_gap(tax: TopicTaxonomy) -> None:
    p = SupplyPlanner(tax, make_cfg())
    cells = list(p.cell_targets(15_000))
    rng = Random(7)
    for _ in range(200):
        have = {c: rng.choice([0, 0, rng.randint(0, 200), rng.randint(0, 800)]) for c in cells}
        st = stats_from_cells(have)
        target = rng.choice([0, 500, 8000, 15_000, 30_000])
        plan = p.plan(st, target)
        assert sum(d.n for d in plan) == max(0, target - st.active)
        targets = p.cell_targets(target)
        for d in plan:
            assert d.n > 0
            # 缩减后不超过原 cell 缺口（stats 与 active 一致时）
            assert d.n <= max(0, targets[(d.lang, d.category, d.bucket)] - have[(d.lang, d.category, d.bucket)])


def test_plan_fills_when_active_exceeds_cells(tax: TopicTaxonomy) -> None:
    """by_cell 不完整（active 中含未知类别）时 Σn 仍等于 target − active。"""
    p = SupplyPlanner(tax, make_cfg())
    st = PoolStats(active=3000, by_lang={"zh": 3000}, by_cell={})
    assert sum(d.n for d in p.plan(st, 8000)) == 5000
    st2 = PoolStats(active=9000)
    assert p.plan(st2, 8000) == []


# ---------------------------------------------------------------------------
# next_spec 优先级与回报
# ---------------------------------------------------------------------------


def test_priority_quarantine_then_expired_then_gap(tax: TopicTaxonomy) -> None:
    p = SupplyPlanner(tax, make_cfg(), target_size=100)
    plain = plain_ids(tax)
    g = sorted(grounding_ids(tax))[0]
    qcell = ("en", plain[0], "L")
    ecell = ("zh", g, "S")
    st = PoolStats(quarantined_cells={qcell: 2}, expired_cells={ecell: 3})
    rng = Random(3)
    specs = [p.next_spec(rng, st if i == 0 else None) for i in range(10)]
    cells = [(s.lang, s.category, s.length_bucket) for s in specs]
    assert cells[:2] == [qcell, qcell]
    assert cells[2:5] == [ecell, ecell, ecell]
    # 剩余为普通缺口；replacement 队列已占用完
    assert p.inflight == 10
    assert p.remaining_total == 100 - 5


def test_quarantine_refund_on_rejection(tax: TopicTaxonomy) -> None:
    p = SupplyPlanner(tax, make_cfg(), target_size=50)
    qcell = ("zh", plain_ids(tax)[1], "M")
    st = PoolStats(quarantined_cells={qcell: 1})
    rng = Random(0)
    s1 = p.next_spec(rng, st)
    assert (s1.lang, s1.category, s1.length_bucket) == qcell
    p.note_rejected(s1, "length")
    s2 = p.next_spec(rng)
    assert (s2.lang, s2.category, s2.length_bucket) == qcell
    p.note_inserted(s2)
    # 替换完成后，reset 不再重复替换
    s3 = p.next_spec(rng, st)
    assert s3 is not None
    p.note_rejected(s3, None)
    assert p._q_remaining.get(qcell, 0) == 0


def test_gap_weighting_follows_deficit(tax: TopicTaxonomy) -> None:
    """仅一个 cell 有缺口时全部任务落在该 cell；拒绝会归还缺口。"""
    p = SupplyPlanner(tax, make_cfg(), target_size=1000)
    targets = p.cell_targets()
    only = ("zh", plain_ids(tax)[0], "M")
    have = dict(targets)
    have[only] = targets[only] - 5
    st = stats_from_cells(have)
    rng = Random(11)
    specs = [p.next_spec(rng, st if i == 0 else None) for i in range(5)]
    assert all((s.lang, s.category, s.length_bucket) == only for s in specs)
    assert p.next_spec(rng) is None
    p.note_rejected(specs[0], "language")
    again = p.next_spec(rng)
    assert again is not None and (again.lang, again.category, again.length_bucket) == only
    for s in specs[1:] + [again]:
        p.note_inserted(s)
    assert p.next_spec(rng) is None and p.inflight == 0


def test_language_gap_first(tax: TopicTaxonomy) -> None:
    """zh 已满、en 全空时任务全部为 en。"""
    p = SupplyPlanner(tax, make_cfg(), target_size=2000)
    have = {c: n for c, n in p.cell_targets().items() if c[0] == "zh"}
    st = stats_from_cells(have)
    rng = Random(5)
    langs = {p.next_spec(rng, st if i == 0 else None).lang for i in range(200)}
    assert langs == {"en"}


def test_category_sampling_proportional(tax: TopicTaxonomy) -> None:
    p = SupplyPlanner(tax, make_cfg(), target_size=100_000)
    a, b = plain_ids(tax)[:2]
    targets = p.cell_targets()
    have = dict(targets)
    for bk in BUCKETS:
        have[("en", a, bk)] = targets[("en", a, bk)] - 300
        have[("en", b, bk)] = targets[("en", b, bk)] - 100
    st = stats_from_cells(have)
    rng = Random(9)
    first = Counter(p.next_spec(rng, st if i == 0 else None).category for i in range(400))
    # 前 400 条全部来自 a/b，且 a 约为 b 的 3 倍
    assert set(first) <= {a, b}
    assert 2.0 < first[a] / max(1, first[b]) < 4.5


# ---------------------------------------------------------------------------
# grounding 上限
# ---------------------------------------------------------------------------


def test_grounding_limit_removes_grounding_categories_until_next_utc_day(tax: TopicTaxonomy) -> None:
    now = Now(datetime(2024, 5, 1, 23, 0, tzinfo=timezone.utc))
    p = SupplyPlanner(tax, make_cfg(limit=5), target_size=20_000, utcnow=now)
    gids = grounding_ids(tax)
    assert gids
    p.set_grounding_used(5, date(2024, 5, 1))
    assert not p.grounding_available()
    rng = Random(2)
    st = PoolStats()
    specs = [p.next_spec(rng, st if i == 0 else None) for i in range(300)]
    assert all(s.category not in gids and not s.grounding for s in specs)

    now.t += timedelta(hours=2)  # 跨过 UTC 00:00
    assert p.grounding_available() and p.grounding_used == 0
    more = [p.next_spec(rng) for _ in range(600)]
    grounded = [s for s in more if s.grounding]
    # 在途 grounding 任务计入上限：最多发出 5 条
    assert 0 < len(grounded) <= 5
    assert all(s.category in gids for s in grounded)


def test_grounding_only_filter_returns_none_at_limit(tax: TopicTaxonomy) -> None:
    now = Now(datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc))
    gids = sorted(grounding_ids(tax))
    p = SupplyPlanner(tax, make_cfg(limit=2), target_size=1000, categories=gids, utcnow=now)
    rng = Random(4)
    s1 = p.next_spec(rng, PoolStats())
    s2 = p.next_spec(rng)
    assert s1.grounding and s2.grounding
    assert p.next_spec(rng) is None  # 在途 2 条已占满上限
    p.note_grounding_call()
    p.note_grounding_call()
    p.note_inserted(s1)
    p.note_inserted(s2)
    assert p.grounding_used == 2
    assert p.next_spec(rng) is None  # 不以非 grounding 方式生成


# ---------------------------------------------------------------------------
# 子话题降权
# ---------------------------------------------------------------------------


def _spec_for(p: SupplyPlanner, rng: Random, sub: str | None = None):
    s = p.next_spec(rng)
    while sub is not None and s.subtopic != sub:
        p.note_rejected(s, None)
        s = p.next_spec(rng)
    return s


def test_penalize_after_20_samples_over_30pct(tax: TopicTaxonomy) -> None:
    cid = plain_ids(tax)[0]
    p = SupplyPlanner(tax, make_cfg(), target_size=10_000, categories=[cid], langs=["zh"])
    rng = Random(8)
    p.reset(PoolStats())
    s = p.next_spec(rng)
    sub = s.subtopic
    p.note_rejected(s, None)  # 释放首条（不计入样本）
    w0 = tax.subtopic_weight(sub)
    # 13 次成功 + 6 次拒绝 → 19 样本，未达 20
    for _ in range(13):
        p.note_inserted(_spec_for(p, rng, sub))
    for _ in range(6):
        p.note_rejected(_spec_for(p, rng, sub), "refusal")
    assert tax.subtopic_weight(sub) == w0
    # 第 20 个样本为拒绝：7/20 = 35% > 30% → 降权一次并清零
    p.note_rejected(_spec_for(p, rng, sub), "blocklist")
    assert tax.subtopic_weight(sub) == pytest.approx(w0 * 0.5)
    assert p.subtopic_stats(sub) == (0, 0)


def test_no_penalty_at_exactly_30pct_and_non_content_reasons(tax: TopicTaxonomy) -> None:
    cid = plain_ids(tax)[0]
    p = SupplyPlanner(tax, make_cfg(), target_size=10_000, categories=[cid], langs=["en"])
    rng = Random(12)
    p.reset(PoolStats())
    s = p.next_spec(rng)
    sub = s.subtopic
    p.note_rejected(s, None)
    w0 = tax.subtopic_weight(sub)
    for _ in range(14):
        p.note_inserted(_spec_for(p, rng, sub))
    for _ in range(6):
        p.note_rejected(_spec_for(p, rng, sub), "duplicate")
    for _ in range(10):
        p.note_rejected(_spec_for(p, rng, sub), "error")  # 非内容原因不计入
    assert p.subtopic_stats(sub) == (20, 6)
    assert tax.subtopic_weight(sub) == w0


# ---------------------------------------------------------------------------
# 过滤与数量上限
# ---------------------------------------------------------------------------


def test_category_lang_filter_and_limit(tax: TopicTaxonomy) -> None:
    cats = plain_ids(tax)[:2]
    p = SupplyPlanner(tax, make_cfg(), categories=cats, langs=["en"], limit=7)
    # 库存已满：仍按 generate 指定数量给出任务
    full = stats_from_cells(SupplyPlanner(tax, make_cfg()).cell_targets(15_000))
    rng = Random(6)
    out = []
    while (s := p.next_spec(rng, full if not out else None)) is not None:
        out.append(s)
        assert len(out) <= 7
    assert len(out) == 7
    assert all(s.category in cats and s.lang == "en" for s in out)
    p.note_rejected(out[0], "length")
    s = p.next_spec(rng)
    assert s is not None
    assert p.next_spec(rng) is None
    for x in out[1:] + [s]:
        p.note_inserted(x)
    assert p.inserted == 7 and p.next_spec(rng) is None


def test_unknown_filter_rejected(tax: TopicTaxonomy) -> None:
    with pytest.raises(ValueError):
        SupplyPlanner(tax, make_cfg(), categories=["no_such_cat"])
    with pytest.raises(ValueError):
        SupplyPlanner(tax, make_cfg(), langs=["fr"])


def test_no_gap_returns_none(tax: TopicTaxonomy) -> None:
    p = SupplyPlanner(tax, make_cfg(), target_size=8000)
    full = stats_from_cells(p.cell_targets())
    assert p.plan(full) == []
    assert p.next_spec(Random(0), full) is None


# ---------------------------------------------------------------------------
# check_for_run 示例
# ---------------------------------------------------------------------------


def test_check_for_run_default_required_and_suggestion() -> None:
    cfg = make_cfg()
    chk = check_for_run(PoolStats(), 1200, cfg)
    assert chk.verdict == "refuse"
    assert chk.required == {"zh": 2369, "en": 1579}
    assert chk.suggested_command == "loadtest warmup --target 8000"
    assert any("warmup" in m for m in chk.messages)


def test_check_for_run_ok_and_warn(tax: TopicTaxonomy) -> None:
    cfg = make_cfg()
    p = SupplyPlanner(tax, cfg)
    healthy = stats_from_cells(p.cell_targets(9000))
    chk = p.check_for_run(healthy, 1200)
    assert chk.verdict == "ok" and chk.messages == [] and chk.suggested_command is None

    small = stats_from_cells(p.cell_targets(6000))
    chk = check_for_run(small, 1200, cfg)
    assert chk.verdict == "warn"
    assert chk.suggested_command and chk.suggested_command.startswith("loadtest warmup --target ")
    # tpm 覆盖：更高 TPM 使所需库存上升而拒绝
    assert check_for_run(small, 1200, cfg, tpm=5_000_000).verdict == "refuse"
