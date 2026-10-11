"""SupplyPlanner：按 语言 × 类别 × 长度桶 计算语料缺口、输出下一条生成任务，以及运行前供应检查。

- ``plan()``：目标规模按 lang_ratio → 类别权重 → S/M/L = 30/40/30 逐级拆分，每级用最大余数法取整，
  使子目标之和恰好等于上级目标；cell 缺口 = max(0, 目标 − 现有)，再调整使 Σn = max(0, 目标 − active)
  （Requirement 7.1, 7.2）。
- ``next_spec()``：优先级 隔离替换 > 过期替换 > 语言缺口 > 类别缺口 > 长度桶缺口；同级内按剩余缺口
  加权采样（Requirement 7.3, 10.7）。内部维护剩余缺口，生成结果通过 ``note_inserted`` /
  ``note_rejected`` 回报；已发出但未回报的任务视为占用缺口，拒绝时归还。
- grounding：当日（UTC）grounding 次数 + 在途 grounding 任务 ≥ ``grounding_daily_limit`` 时，
  grounding 类别从采样中移除到下一个 UTC 日，且不以非 grounding 方式生成这些类别（Requirement 3.5）。
- 子话题样本 ≥ 20 且拒绝率 > 30% 时 ``taxonomy.penalize(subtopic, 0.5)``，随后清零计数（Requirement 5.11）。
- ``check_for_run()``：运行前供应检查（Requirement 8.1, 8.2, 8.4, 8.5, 8.7）。
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from random import Random
from typing import Any, Callable, Hashable, Iterable, Literal, Mapping, TypeVar

from loadtest.corpus.store import PoolStats
from loadtest.corpus.taxonomy import TopicSpec, TopicTaxonomy

BUCKETS: tuple[str, ...] = ("S", "M", "L")
BUCKET_SHARES: dict[str, float] = {"S": 0.30, "M": 0.40, "L": 0.30}
AVG_INPUT_TOKENS = 3040          # 平均每请求输入 token（预期请求数换算）
PASSAGES_PER_REQUEST = 2.5       # 平均每请求段落数

PENALTY_MIN_SAMPLES = 20
PENALTY_REJECT_RATE = 0.30
PENALTY_FACTOR = 0.5
# 非内容原因（调用失败、预算、取消）：归还缺口，但不计入子话题拒绝率
NON_CONTENT_REASONS: frozenset[str] = frozenset({"error", "budget", "timeout", "cancelled"})

Cell = tuple[str, str, str]  # (lang, category, bucket)
K = TypeVar("K", bound=Hashable)


@dataclass(frozen=True)
class Deficit:
    lang: str
    category: str
    bucket: str
    n: int


@dataclass(frozen=True)
class SupplyCheck:
    verdict: Literal["ok", "warn", "refuse"]
    required: dict[str, int]                   # 本次运行各语言所需最少库存
    available: dict[str, int]                  # 各语言当前 active 未过期库存
    bucket_shares: dict[tuple[str, str], float]  # (lang, bucket) → 占该语言库存比例
    messages: list[str] = field(default_factory=list)
    suggested_command: str | None = None


# ---------------------------------------------------------------------------
# 取整
# ---------------------------------------------------------------------------


def largest_remainder(total: int, weights: Mapping[K, float]) -> dict[K, int]:
    """最大余数法：把非负整数 total 按 weights 拆分为整数，结果之和恰好为 total。

    权重全为 0（或为空以外的退化情况）时均分。并列按 weights 的迭代顺序决定，结果确定。
    """
    keys = list(weights)
    if total < 0:
        raise ValueError("total 必须 >= 0")
    if not keys:
        if total:
            raise ValueError("weights 为空但 total > 0")
        return {}
    ws = [max(0.0, float(weights[k])) for k in keys]
    s = sum(ws)
    if s <= 0:
        ws = [1.0] * len(keys)
        s = float(len(keys))
    quotas = [total * w / s for w in ws]
    base = [int(math.floor(q)) for q in quotas]
    rest = total - sum(base)
    order = sorted(range(len(keys)), key=lambda i: (-(quotas[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return {k: base[i] for i, k in enumerate(keys)}


# ---------------------------------------------------------------------------
# 运行前供应检查
# ---------------------------------------------------------------------------


def required_supply(tpm: float, duration_s: float, lang_ratio: Mapping[str, float],
                    supply_margin: float, per_run_max_uses: int) -> dict[str, int]:
    """所需库存 = ceil(预期请求数 × 2.5 × lang_ratio × (1 + supply_margin) / per_run_max_uses)。"""
    exp_requests = tpm * duration_s / 60 / AVG_INPUT_TOKENS
    exp_uses = exp_requests * PASSAGES_PER_REQUEST
    return {
        lang: math.ceil(exp_uses * share * (1 + supply_margin) / per_run_max_uses)
        for lang, share in lang_ratio.items()
    }


def check_for_run(stats: PoolStats, duration_s: float, cfg: Any, tpm: float | None = None) -> SupplyCheck:
    """运行前供应检查。stats 只统计 status='active' 且未过期的段落。

    refuse ⟺ active < min_runnable ∨ ∃ lang: available[lang] < required[lang]
    warn   ⟺ 未拒绝 ∧ (active < healthy_size ∨ |占比 − 目标| > tolerance ∨ 某语言某桶占比 < min_bucket_share)
    """
    corpus = cfg.corpus
    ratio: Mapping[str, float] = cfg.load.lang_ratio
    tpm_v = cfg.load.target_tpm if tpm is None else tpm
    required = required_supply(tpm_v, duration_s, ratio, corpus.supply_margin, corpus.per_run_max_uses)
    available = {lang: int(stats.by_lang.get(lang, 0)) for lang in ratio}
    total = int(stats.active)
    bucket_shares = {
        (lang, b): (stats.by_bucket.get((lang, b), 0) / available[lang]) if available[lang] > 0 else 0.0
        for lang in ratio
        for b in BUCKETS
    }
    suggest_n = _suggested_target(required, ratio, corpus.healthy_size, corpus.min_runnable)
    command = f"loadtest warmup --target {suggest_n}"

    short = {lang: required[lang] - available[lang] for lang in ratio if available[lang] < required[lang]}
    if total < corpus.min_runnable or short:
        msgs: list[str] = []
        if total < corpus.min_runnable:
            msgs.append(f"语料池 active 总量 {total} 低于最低可运行量 {corpus.min_runnable}")
        for lang, gap in short.items():
            msgs.append(f"{lang} 库存 {available[lang]} 低于本次运行所需 {required[lang]}（缺 {gap}）")
        msgs.append(f"运行 `{command}` 补齐缺口")
        return SupplyCheck("refuse", required, available, bucket_shares, msgs, command)

    warns: list[str] = []
    if total < corpus.healthy_size:
        warns.append(f"池规模 {total} 低于健康值 {corpus.healthy_size}")
    for lang, share in ratio.items():
        actual = available[lang] / total if total > 0 else 0.0
        if abs(actual - share) > corpus.lang_share_tolerance:
            warns.append(f"{lang} 占比 {actual:.1%} 偏离目标 {share:.0%} 超过 {corpus.lang_share_tolerance:.0%}")
        if available[lang] <= 0:
            continue
        for b in BUCKETS:
            if bucket_shares[(lang, b)] < corpus.min_bucket_share:
                warns.append(
                    f"{lang}/{b} 长度桶占比 {bucket_shares[(lang, b)]:.1%} 低于 {corpus.min_bucket_share:.0%}"
                )
    if warns:
        warns.append(f"建议运行 `{command}`")
        return SupplyCheck("warn", required, available, bucket_shares, warns, command)
    return SupplyCheck("ok", required, available, bucket_shares, [], None)


def _suggested_target(required: Mapping[str, int], ratio: Mapping[str, float],
                      healthy_size: int, min_runnable: int) -> int:
    """建议的 warmup 目标：不低于健康规模，且按 lang_ratio 拆分后每种语言 ≥ 所需库存。"""
    n = max(healthy_size, min_runnable)
    for lang, req in required.items():
        share = ratio.get(lang, 0.0)
        if share > 0:
            n = max(n, math.ceil(req / share))
    return n


# ---------------------------------------------------------------------------
# SupplyPlanner
# ---------------------------------------------------------------------------


class SupplyPlanner:
    """缺口计算与生成任务输出；供 warmup、generate、Replenisher 共用。

    cfg 需提供 ``load.lang_ratio``、``corpus.target_size`` 等；``gemini.grounding_daily_limit`` 可选（默认 600，单位为计费搜索查询数；在途任务按 1 次估计）。
    categories / langs：只生成这些类别 / 语言（generate --category / --lang）。
    limit：本实例最多入库（含在途）的段落数（generate --count）。
    指定 categories / langs / limit 任一项时，缺口填满后仍按目标配比继续给出任务，直到 limit。
    """

    def __init__(
        self,
        taxonomy: TopicTaxonomy,
        cfg: Any,
        *,
        target_size: int | None = None,
        categories: Iterable[str] | None = None,
        langs: Iterable[str] | None = None,
        limit: int | None = None,
        utcnow: Callable[[], datetime] | None = None,
    ) -> None:
        self.taxonomy = taxonomy
        self.cfg = cfg
        self.lang_ratio: dict[str, float] = dict(cfg.load.lang_ratio)
        self.target_size = int(cfg.corpus.target_size if target_size is None else target_size)
        gemini = getattr(cfg, "gemini", None)
        self.grounding_daily_limit = int(getattr(gemini, "grounding_daily_limit", 600))

        known = {c.id for c in taxonomy.categories()}
        self.categories: frozenset[str] | None = None
        if categories is not None:
            cats = frozenset(categories)
            unknown = sorted(cats - known)
            if unknown:
                raise ValueError(f"未知类别: {', '.join(unknown)}")
            self.categories = cats
        self.langs: frozenset[str] | None = None
        if langs is not None:
            ls = frozenset(langs)
            bad = sorted(ls - set(self.lang_ratio))
            if bad:
                raise ValueError(f"未知语言: {', '.join(bad)}")
            self.langs = ls
        if limit is not None and limit < 0:
            raise ValueError("limit 必须 >= 0")
        self.limit = limit
        self._explicit = categories is not None or langs is not None or limit is not None
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))

        self._initialized = False
        self._remaining: dict[Cell, int] = {}
        self._q_remaining: dict[Cell, int] = {}
        self._e_remaining: dict[Cell, int] = {}
        self._q_consumed: dict[Cell, int] = defaultdict(int)
        self._e_consumed: dict[Cell, int] = defaultdict(int)
        self._pending: dict[TopicSpec, list[str]] = defaultdict(list)
        self._inflight = 0
        self._inserted = 0
        self._g_inflight = 0
        self._g_used = 0
        self._g_day: date = self._today()
        self._sub_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [samples, rejects]

    # ------------------------------------------------------------------ 目标与缺口

    def _cells(self) -> list[Cell]:
        return [(lang, c.id, b) for lang in self.lang_ratio for c in self.taxonomy.categories() for b in BUCKETS]

    def cell_targets(self, target_size: int | None = None) -> dict[Cell, int]:
        """逐级最大余数取整的 cell 目标；每级子目标之和等于上级目标。"""
        target = self.target_size if target_size is None else int(target_size)
        target = max(0, target)
        cat_w = self.taxonomy.target_weights()
        out: dict[Cell, int] = {}
        for lang, lang_t in largest_remainder(target, self.lang_ratio).items():
            for cid, cat_t in largest_remainder(lang_t, cat_w).items():
                for b, n in largest_remainder(cat_t, BUCKET_SHARES).items():
                    out[(lang, cid, b)] = n
        return out

    def plan(self, stats: PoolStats, target_size: int | None = None) -> list[Deficit]:
        """Σ n == max(0, target − stats.active)；只返回 n > 0 的 cell。"""
        target = self.target_size if target_size is None else int(target_size)
        need = max(0, target - int(stats.active))
        targets = self.cell_targets(target)
        raw = {c: max(0, t - int(stats.by_cell.get(c, 0))) for c, t in targets.items()}
        d_sum = sum(raw.values())
        if need == 0:
            final = {c: 0 for c in raw}
        elif d_sum >= need:
            # 按缺口比例缩减（缩减后每格不超过原缺口）
            final = largest_remainder(need, raw) if d_sum > need else dict(raw)
        else:
            # 现有统计少于 active（如类别不在分类树中）：差额按 cell 目标配比补足
            extra = largest_remainder(need - d_sum, targets)
            final = {c: raw[c] + extra[c] for c in raw}
        return [Deficit(l, cid, b, n) for (l, cid, b), n in final.items() if n > 0]

    def reset(self, stats: PoolStats) -> None:
        """按最新库存重算剩余缺口与替换队列；在途任务继续占用缺口。"""
        self._remaining = {(d.lang, d.category, d.bucket): d.n for d in self.plan(stats)}
        self._q_remaining = {
            c: max(0, int(n) - self._q_consumed[c]) for c, n in stats.quarantined_cells.items()
        }
        self._e_remaining = {
            c: max(0, int(n) - self._e_consumed[c]) for c, n in stats.expired_cells.items()
        }
        for spec, kinds in self._pending.items():
            cell = (spec.lang, spec.category, spec.length_bucket)
            for k in kinds:
                if k == "d" and self._remaining.get(cell, 0) > 0:
                    self._remaining[cell] -= 1
        self._initialized = True

    @property
    def remaining_total(self) -> int:
        """剩余缺口（不含替换队列与在途任务）。"""
        return sum(self._remaining.values())

    @property
    def inflight(self) -> int:
        return self._inflight

    @property
    def inserted(self) -> int:
        return self._inserted

    # ------------------------------------------------------------------ grounding

    def _today(self) -> date:
        now = self._utcnow()
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc)
        return now.date()

    def _roll_day(self) -> None:
        today = self._today()
        if today != self._g_day:
            self._g_day = today
            self._g_used = 0

    def set_grounding_used(self, n: int, day: date | None = None) -> None:
        """从数据库恢复当日（UTC）已用 grounding 次数。"""
        self._g_day = self._today() if day is None else day
        self._g_used = int(n)
        self._roll_day()

    def note_grounding_call(self) -> None:
        self._roll_day()
        self._g_used += 1

    @property
    def grounding_used(self) -> int:
        self._roll_day()
        return self._g_used

    def grounding_available(self) -> bool:
        """当日 grounding 已用 + 在途 < 上限。"""
        self._roll_day()
        return self._g_used + self._g_inflight < self.grounding_daily_limit

    # ------------------------------------------------------------------ 任务输出

    def _eligible(self, cell: Cell, allow_grounding: bool) -> bool:
        lang, cid, _ = cell
        if self.langs is not None and lang not in self.langs:
            return False
        if self.categories is not None and cid not in self.categories:
            return False
        try:
            cat = self.taxonomy.category(cid)
        except KeyError:
            return False
        return allow_grounding or not cat.grounding

    @staticmethod
    def _choose(rng: Random, weights: Mapping[K, float]) -> K:
        keys = list(weights)
        return rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]

    def _choose_hier(self, rng: Random, cells: Mapping[Cell, float]) -> Cell:
        """语言 → 类别 → 长度桶 逐级按缺口加权选择。"""
        by_lang: dict[str, float] = defaultdict(float)
        for (l, _, _), w in cells.items():
            by_lang[l] += w
        lang = self._choose(rng, by_lang)
        by_cat: dict[str, float] = defaultdict(float)
        for (l, cid, _), w in cells.items():
            if l == lang:
                by_cat[cid] += w
        cid = self._choose(rng, by_cat)
        by_b = {b: w for (l, c, b), w in cells.items() if l == lang and c == cid}
        b = self._choose(rng, by_b)
        return (lang, cid, b)

    def next_spec(self, rng: Random, stats: PoolStats | None = None) -> TopicSpec | None:
        """返回下一条生成任务；无缺口（或达到 limit / 无可用类别）时返回 None。"""
        if stats is not None:
            self.reset(stats)
        elif not self._initialized:
            self.reset(PoolStats())
        if self.limit is not None and self._inserted + self._inflight >= self.limit:
            return None
        allow_g = self.grounding_available()

        for kind, pool, consumed in (
            ("q", self._q_remaining, self._q_consumed),
            ("e", self._e_remaining, self._e_consumed),
        ):
            cand = {c: float(n) for c, n in pool.items() if n > 0 and self._eligible(c, allow_g)}
            if cand:
                cell = self._choose_hier(rng, cand)
                pool[cell] -= 1
                consumed[cell] += 1
                return self._issue(rng, cell, kind, allow_g)

        cand = {c: float(n) for c, n in self._remaining.items() if n > 0 and self._eligible(c, allow_g)}
        if cand:
            cell = self._choose_hier(rng, cand)
            self._remaining[cell] -= 1
            return self._issue(rng, cell, "d", allow_g)

        if self._explicit:
            cand = {
                c: float(t) for c, t in self.cell_targets().items() if self._eligible(c, allow_g)
            }
            if not cand:
                return None
            if sum(cand.values()) <= 0:
                cand = {c: 1.0 for c in cand}
            return self._issue(rng, self._choose_hier(rng, cand), "x", allow_g)
        return None

    def _issue(self, rng: Random, cell: Cell, kind: str, allow_g: bool) -> TopicSpec:
        lang, cid, b = cell
        spec = self.taxonomy.sample(
            rng, None, lang, length_bucket=b, category=cid, allow_grounding=allow_g  # type: ignore[arg-type]
        )
        self._pending[spec].append(kind)
        self._inflight += 1
        if spec.grounding:
            self._g_inflight += 1
        return spec

    def _finish(self, spec: TopicSpec) -> str | None:
        kinds = self._pending.get(spec)
        if not kinds:
            return None
        kind = kinds.pop()
        if not kinds:
            del self._pending[spec]
        self._inflight -= 1
        if spec.grounding:
            self._g_inflight -= 1
        return kind

    # ------------------------------------------------------------------ 结果回报

    def note_inserted(self, spec: TopicSpec) -> None:
        kind = self._finish(spec)
        self._inserted += 1
        if kind in ("q", "e"):
            # 替代段落同时填补该 cell 的缺口
            cell = (spec.lang, spec.category, spec.length_bucket)
            if self._remaining.get(cell, 0) > 0:
                self._remaining[cell] -= 1
        self._record_subtopic(spec.subtopic, rejected=False)

    def note_rejected(self, spec: TopicSpec, reason: str | None) -> None:
        """生成失败或被拒：归还占用的缺口；内容类原因计入子话题拒绝率。"""
        kind = self._finish(spec)
        cell = (spec.lang, spec.category, spec.length_bucket)
        if kind == "d":
            self._remaining[cell] = self._remaining.get(cell, 0) + 1
        elif kind == "q":
            self._q_remaining[cell] = self._q_remaining.get(cell, 0) + 1
            self._q_consumed[cell] -= 1
        elif kind == "e":
            self._e_remaining[cell] = self._e_remaining.get(cell, 0) + 1
            self._e_consumed[cell] -= 1
        if reason is not None and reason not in NON_CONTENT_REASONS:
            self._record_subtopic(spec.subtopic, rejected=True)

    def _record_subtopic(self, subtopic: str, *, rejected: bool) -> None:
        st = self._sub_stats[subtopic]
        st[0] += 1
        if rejected:
            st[1] += 1
        if st[0] >= PENALTY_MIN_SAMPLES and st[1] / st[0] > PENALTY_REJECT_RATE:
            try:
                self.taxonomy.penalize(subtopic, PENALTY_FACTOR)
            except KeyError:
                pass
            st[0] = st[1] = 0

    def subtopic_stats(self, subtopic: str) -> tuple[int, int]:
        """(样本数, 拒绝数)，自上次降权后累计。"""
        st = self._sub_stats.get(subtopic)
        return (st[0], st[1]) if st else (0, 0)

    # ------------------------------------------------------------------ 供应检查

    def check_for_run(self, stats: PoolStats, duration_s: float, cfg: Any | None = None,
                      tpm: float | None = None) -> SupplyCheck:
        return check_for_run(stats, duration_s, cfg or self.cfg, tpm)


__all__ = [
    "BUCKETS",
    "BUCKET_SHARES",
    "Deficit",
    "SupplyCheck",
    "SupplyPlanner",
    "check_for_run",
    "largest_remainder",
    "required_supply",
]
