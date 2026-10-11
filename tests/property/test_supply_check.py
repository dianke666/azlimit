"""Property 25: 供应检查（判定部分）。

**Validates: Requirements 8.1, 8.2, 8.4, 8.5**

∀ 库存统计与时长：refuse ⟺ active < min_runnable ∨ ∃ lang: available[lang] < required[lang]；
未拒绝时 warn ⟺ active < healthy_size ∨ 语言占比偏离 > tolerance ∨ 某语言某桶占比 < min_bucket_share；
否则 ok。required[lang] = ceil(target_tpm × duration_s / 60 / 3040 × 2.5 × lang_ratio × (1 + margin) / max_uses)。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.corpus.planner import check_for_run
from loadtest.corpus.store import PoolStats
from loadtest.settings import CorpusSettings, LoadSettings

BUCKETS = ("S", "M", "L")


@st.composite
def pool_stats(draw) -> PoolStats:
    by_bucket: dict[tuple[str, str], int] = {}
    by_lang: dict[str, int] = {}
    for lang in ("zh", "en"):
        scale = draw(st.sampled_from([0, 50, 1000, 4000, 10_000]))
        for b in BUCKETS:
            n = draw(st.integers(0, scale))
            by_bucket[(lang, b)] = n
        by_lang[lang] = sum(by_bucket[(lang, b)] for b in BUCKETS)
    return PoolStats(active=sum(by_lang.values()), by_lang=by_lang, by_bucket=by_bucket)


@st.composite
def configs(draw) -> SimpleNamespace:
    zh = draw(st.sampled_from([0.0, 0.3, 0.5, 0.6, 0.7, 1.0]))
    load = LoadSettings(
        target_tpm=draw(st.integers(1, 2_000_000)),
        lang_ratio={"zh": zh, "en": 1.0 - zh},
    )
    corpus = CorpusSettings(
        min_runnable=draw(st.integers(0, 5000)),
        healthy_size=draw(st.integers(0, 20_000)),
        per_run_max_uses=draw(st.integers(1, 8)),
        supply_margin=draw(st.sampled_from([0.0, 0.1, 0.2, 0.5])),
        lang_share_tolerance=draw(st.sampled_from([0.0, 0.05, 0.1])),
        min_bucket_share=draw(st.sampled_from([0.0, 0.15, 0.3])),
    )
    return SimpleNamespace(load=load, corpus=corpus)


def expected_required(cfg, duration_s: int, tpm: float) -> dict[str, int]:
    exp_requests = tpm * duration_s / 60 / 3040
    return {
        lang: math.ceil(exp_requests * 2.5 * share * (1 + cfg.corpus.supply_margin) / cfg.corpus.per_run_max_uses)
        for lang, share in cfg.load.lang_ratio.items()
    }


def expected_verdict(stats: PoolStats, cfg, required: dict[str, int]) -> str:
    c = cfg.corpus
    total = stats.active
    if total < c.min_runnable or any(stats.by_lang.get(l, 0) < n for l, n in required.items()):
        return "refuse"
    for lang, share in cfg.load.lang_ratio.items():
        have = stats.by_lang.get(lang, 0)
        if abs(have / total - share) > c.lang_share_tolerance:
            return "warn"
        if have > 0 and any(stats.by_bucket.get((lang, b), 0) / have < c.min_bucket_share for b in BUCKETS):
            return "warn"
    if total < c.healthy_size:
        return "warn"
    return "ok"


@settings(max_examples=500, deadline=None)
@given(
    stats=pool_stats(),
    cfg=configs(),
    duration_s=st.integers(1, 7200),
    tpm_override=st.one_of(st.none(), st.integers(1, 3_000_000)),
)
def test_supply_check_verdict(stats: PoolStats, cfg, duration_s: int, tpm_override) -> None:
    tpm = cfg.load.target_tpm if tpm_override is None else tpm_override
    chk = check_for_run(stats, duration_s, cfg, tpm=tpm_override)

    required = expected_required(cfg, duration_s, tpm)
    assert chk.required == required
    assert chk.available == {l: stats.by_lang.get(l, 0) for l in ("zh", "en")}

    exp = expected_verdict(stats, cfg, required)
    assert chk.verdict == exp
    refuse = stats.active < cfg.corpus.min_runnable or any(
        chk.available[l] < chk.required[l] for l in chk.required
    )
    assert (chk.verdict == "refuse") == refuse
    if exp == "ok":
        assert chk.messages == [] and chk.suggested_command is None
    else:
        assert chk.messages
        assert chk.suggested_command is not None and chk.suggested_command.startswith("loadtest warmup --target ")
        n = int(chk.suggested_command.rsplit(" ", 1)[1])
        # 按建议目标拆分后每种语言都满足所需库存
        for lang, share in cfg.load.lang_ratio.items():
            if share > 0:
                assert n * share >= required[lang] - 1e-6


@settings(max_examples=200, deadline=None)
@given(stats=pool_stats(), cfg=configs(), duration_s=st.integers(1, 7200))
def test_warn_lists_every_condition(stats: PoolStats, cfg, duration_s: int) -> None:
    """warn 时每条成立的告警条件都对应一条消息（另加 1 条 warmup 建议）。"""
    chk = check_for_run(stats, duration_s, cfg)
    if chk.verdict != "warn":
        return
    c = cfg.corpus
    total = stats.active
    n = int(total < c.healthy_size)
    for lang, share in cfg.load.lang_ratio.items():
        have = stats.by_lang.get(lang, 0)
        n += int(abs(have / total - share) > c.lang_share_tolerance)
        if have > 0:
            n += sum(stats.by_bucket.get((lang, b), 0) / have < c.min_bucket_share for b in BUCKETS)
    assert len(chk.messages) == n + 1
