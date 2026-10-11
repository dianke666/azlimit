"""话题分类树（TopicTaxonomy）：大类 → 子话题 → 体裁。

- 数据来自包内 `taxonomy.yaml`（可通过 `TopicTaxonomy.load(path)` 指定其他文件）。
- 加载时校验：子话题数 ∈ [10, 30]、grounding 只允许出现在 `grounding_categories`、
  体裁属于 8 种之一、权重 > 0、子话题不含敏感词等；任一不合规时抛出 `TaxonomyError`，
  并列出全部不合规的类别名（Requirement 2.6）。
- `sample()` 返回单一语言的 `TopicSpec`，长度桶 ∈ {S, M, L}，target_len 落在桶区间内。
- `penalize()` 降低某个子话题的采样权重（例如 Gemini 安全拦截后）。
- `coverage()` 计算各类别当前占比 / 目标权重。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from random import Random
from typing import Any, Iterable, Literal, Mapping

import yaml

Lang = Literal["zh", "en"]
LengthBucket = Literal["S", "M", "L"]

LANGS: tuple[str, ...] = ("zh", "en")

# 8 种体裁（规范名为中文），以及用于英文提示词的对应名称
GENRES: tuple[str, ...] = ("说明文", "评论", "教程", "案例分析", "摘要", "对话", "书评/影评", "新闻综述")
# 类别未声明 genres 时的缺省体裁：不含"新闻综述"（避免非 grounding 类别编造新闻）与"书评/影评"
DEFAULT_GENRES: tuple[str, ...] = ("说明文", "评论", "教程", "案例分析", "摘要", "对话")
GENRE_EN: dict[str, str] = {
    "说明文": "expository article",
    "评论": "commentary",
    "教程": "tutorial",
    "案例分析": "case study",
    "摘要": "summary",
    "对话": "dialogue",
    "书评/影评": "book or film review",
    "新闻综述": "news roundup",
}

DEFAULT_GROUNDING_CATEGORIES: tuple[str, ...] = ("daily_news", "daily_economy", "sports_events")
DEFAULT_LENGTH_BUCKETS: dict[str, tuple[int, int]] = {"S": (500, 799), "M": (800, 1199), "L": (1200, 1500)}
# 长度桶默认配比 S/M/L = 30% / 40% / 30%（与 SupplyPlanner 一致）
DEFAULT_BUCKET_WEIGHTS: dict[str, float] = {"S": 0.30, "M": 0.40, "L": 0.30}

# Requirement 2.1 指定的 13 个大类（按中文名校验）
REQUIRED_CATEGORY_NAMES: tuple[str, ...] = (
    "影视技巧", "多媒体技巧", "中国小说", "人物传记", "中外文学欣赏", "学术论文",
    "每日新闻", "每日经济", "体育赛事", "厨艺", "自然", "电商", "运动",
)
MIN_CATEGORIES = 14
MIN_SUBTOPICS = 10
MAX_SUBTOPICS = 30

# 子话题敏感词（Requirement 2.5）：命中即视为该类别不合规
_SENSITIVE_ZH: tuple[str, ...] = (
    "政治", "政党", "选举", "宗教", "教派", "暴力", "恐怖", "色情", "赌博", "博彩", "毒品",
    "违法", "犯罪", "诊断", "处方", "投资建议", "荐股", "炒股", "争议",
)
_SENSITIVE_EN = re.compile(
    r"\b(politic\w*|election\w*|religio\w*|violen\w*|terror\w*|porn\w*|sex\w*|gambl\w*|"
    r"narcotic\w*|illegal|crime|criminal|diagnos\w*|prescription\w*|investment advice|"
    r"stock pick\w*|controvers\w*)\b",
    re.IGNORECASE,
)

_PENALTY_FLOOR = 1e-3  # 子话题权重下限，避免被完全移出采样


class TaxonomyError(ValueError):
    """taxonomy.yaml 不合规。`categories` 为全部不合规的类别名。"""

    def __init__(self, categories: list[str], details: list[str]):
        self.categories = categories
        self.details = details
        msg = "taxonomy 校验失败，不合规类别: " + ", ".join(categories)
        if details:
            msg += "\n" + "\n".join(f"  - {d}" for d in details)
        super().__init__(msg)


@dataclass(frozen=True)
class TopicSpec:
    category: str              # 类别 id，例如 "daily_news"
    subtopic: str              # 子话题名称（与 lang 同语言）
    genre: str                 # 8 种体裁之一（中文规范名）
    lang: Lang
    grounding: bool
    length_bucket: LengthBucket
    target_len: int            # 按长度桶取值：S≈650 / M≈1000 / L≈1350 token


@dataclass(frozen=True)
class Subtopic:
    zh: str
    en: str

    def name(self, lang: str) -> str:
        return self.zh if lang == "zh" else self.en


@dataclass(frozen=True)
class Category:
    id: str
    zh: str
    en: str
    weight: float
    grounding: bool
    genres: tuple[str, ...]
    subtopics: tuple[Subtopic, ...] = field(default_factory=tuple)

    def name(self, lang: str) -> str:
        return self.zh if lang == "zh" else self.en


def _sensitive(text: str) -> bool:
    return any(w in text for w in _SENSITIVE_ZH) or bool(_SENSITIVE_EN.search(text))


class TopicTaxonomy:
    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        grounding_categories: Iterable[str] = DEFAULT_GROUNDING_CATEGORIES,
        length_buckets: Mapping[str, tuple[int, int]] | None = None,
        bucket_weights: Mapping[str, float] | None = None,
        require_core: bool = True,
    ) -> None:
        self._grounding_categories = frozenset(grounding_categories)
        self._buckets: dict[str, tuple[int, int]] = dict(length_buckets or DEFAULT_LENGTH_BUCKETS)
        if set(self._buckets) != {"S", "M", "L"}:
            raise ValueError("length_buckets 的键必须为 S, M, L")
        for b, (lo, hi) in self._buckets.items():
            if lo > hi:
                raise ValueError(f"length_buckets['{b}'] 下界大于上界")
        self._bucket_weights = dict(bucket_weights or DEFAULT_BUCKET_WEIGHTS)

        self._categories: list[Category] = self._parse_and_validate(data, require_core)
        self._by_id: dict[str, Category] = {c.id: c for c in self._categories}
        # 子话题采样权重，键 = (类别 id, 子话题中文名)
        self._sub_weights: dict[tuple[str, str], float] = {
            (c.id, s.zh): 1.0 for c in self._categories for s in c.subtopics
        }
        # 名称 → 键（中英文名均可用于 penalize）
        self._sub_index: dict[str, list[tuple[str, str]]] = {}
        for c in self._categories:
            for s in c.subtopics:
                for n in {s.zh, s.en}:
                    self._sub_index.setdefault(n, []).append((c.id, s.zh))

    # ------------------------------------------------------------------ 加载

    @classmethod
    def load(cls, path: str | Path | None = None, **kwargs: Any) -> "TopicTaxonomy":
        """从 YAML 加载；path 为空时使用包内 taxonomy.yaml。"""
        if path is None:
            text = resources.files("loadtest.corpus").joinpath("taxonomy.yaml").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise TaxonomyError(["<root>"], ["顶层必须是映射"])
        return cls(data, **kwargs)

    def _parse_and_validate(self, data: Mapping[str, Any], require_core: bool) -> list[Category]:
        raw_cats = data.get("categories")
        if not isinstance(raw_cats, list) or not raw_cats:
            raise TaxonomyError(["<root>"], ["categories 必须是非空列表"])

        bad: list[str] = []
        details: list[str] = []

        def flag(name: str, reason: str) -> None:
            if name not in bad:
                bad.append(name)
            details.append(f"{name}: {reason}")

        cats: list[Category] = []
        seen_ids: set[str] = set()
        for i, rc in enumerate(raw_cats):
            if not isinstance(rc, Mapping):
                flag(f"<#{i}>", "类别必须是映射")
                continue
            cid = str(rc.get("id") or "").strip()
            zh = str(rc.get("zh") or "").strip()
            en = str(rc.get("en") or "").strip()
            label = cid or zh or f"<#{i}>"
            if not cid or not zh or not en:
                flag(label, "缺少 id / zh / en")
            if cid in seen_ids:
                flag(label, "类别 id 重复")
            seen_ids.add(cid)

            weight = rc.get("weight", 1.0)
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                flag(label, f"weight 必须 > 0（当前 {weight!r}）")
                weight = 1.0

            grounding = rc.get("grounding", False)
            if not isinstance(grounding, bool):
                flag(label, "grounding 必须是布尔值")
                grounding = bool(grounding)
            if grounding and cid not in self._grounding_categories:
                flag(label, "非 grounding 类别被标记为需要 grounding")

            genres_raw = rc.get("genres") or list(DEFAULT_GENRES)
            genres = tuple(str(g) for g in genres_raw) if isinstance(genres_raw, list) else ()
            if not genres or any(g not in GENRES for g in genres):
                flag(label, f"genres 必须是 {list(GENRES)} 的非空子集")
                genres = tuple(g for g in genres if g in GENRES) or GENRES

            subs_raw = rc.get("subtopics")
            subs: list[Subtopic] = []
            if not isinstance(subs_raw, list):
                subs_raw = []
            for s in subs_raw:
                if isinstance(s, Mapping) and s.get("zh") and s.get("en"):
                    subs.append(Subtopic(zh=str(s["zh"]).strip(), en=str(s["en"]).strip()))
                else:
                    flag(label, f"子话题缺少 zh/en: {s!r}")
            if not (MIN_SUBTOPICS <= len(subs_raw) <= MAX_SUBTOPICS):
                flag(label, f"子话题数 {len(subs_raw)} 不在 {MIN_SUBTOPICS}–{MAX_SUBTOPICS} 内")
            if len({s.zh for s in subs}) != len(subs):
                flag(label, "子话题重复")
            for s in subs:
                if _sensitive(s.zh) or _sensitive(s.en):
                    flag(label, f"子话题含敏感内容: {s.zh} / {s.en}")

            cats.append(Category(cid, zh, en, float(weight), grounding, genres, tuple(subs)))

        if require_core:
            names = {c.zh for c in cats}
            missing = [n for n in REQUIRED_CATEGORY_NAMES if n not in names]
            if missing:
                for n in missing:
                    flag(n, "缺少必需大类")
            if len(cats) < MIN_CATEGORIES:
                flag("<root>", f"大类数 {len(cats)} 少于 {MIN_CATEGORIES}")
            elif len(cats) <= len(REQUIRED_CATEGORY_NAMES) or not (names - set(REQUIRED_CATEGORY_NAMES)):
                flag("<root>", "至少需要 1 个扩展类别")

        if bad:
            raise TaxonomyError(bad, details)
        return cats

    # ------------------------------------------------------------------ 查询

    def categories(self) -> list[Category]:
        return list(self._categories)

    def category(self, cid: str) -> Category:
        return self._by_id[cid]

    def target_weights(self) -> dict[str, float]:
        """归一化后的类别目标权重。"""
        total = sum(c.weight for c in self._categories)
        return {c.id: c.weight / total for c in self._categories}

    def category_label(self, cid: str, lang: str) -> str:
        return self._by_id[cid].name(lang)

    @staticmethod
    def genre_label(genre: str, lang: str) -> str:
        return genre if lang == "zh" else GENRE_EN[genre]

    @property
    def length_buckets(self) -> dict[str, tuple[int, int]]:
        return dict(self._buckets)

    def subtopic_weight(self, subtopic: str) -> float:
        keys = self._sub_index.get(subtopic)
        if not keys:
            raise KeyError(subtopic)
        return min(self._sub_weights[k] for k in keys)

    # ------------------------------------------------------------------ 采样

    def sample(
        self,
        rng: Random,
        weights: Mapping[str, float] | None,
        lang: Lang,
        *,
        length_bucket: LengthBucket | None = None,
        category: str | None = None,
        allow_grounding: bool = True,
    ) -> TopicSpec:
        """按类别权重采样一个 TopicSpec。

        weights: 类别 id → 权重（SupplyPlanner 按缺口给出）；为空时使用 YAML 目标权重。
        length_bucket / category: 指定时固定该维度。
        allow_grounding: False 时排除 grounding 类别（例如当日 grounding 次数已达上限）。
        """
        if lang not in LANGS:
            raise ValueError(f"lang 必须是 zh 或 en，当前 {lang!r}")

        if category is not None:
            cat = self._by_id[category]
            if cat.grounding and not allow_grounding:
                raise ValueError(f"类别 {category} 需要 grounding，但当前不允许")
        else:
            base = weights if weights else {c.id: c.weight for c in self._categories}
            cands = [
                (c, float(base.get(c.id, 0.0)))
                for c in self._categories
                if allow_grounding or not c.grounding
            ]
            cands = [(c, w) for c, w in cands if w > 0]
            if not cands:
                raise ValueError("没有可采样的类别（权重全为 0 或均被排除）")
            cat = rng.choices([c for c, _ in cands], weights=[w for _, w in cands], k=1)[0]

        sub = rng.choices(
            cat.subtopics, weights=[self._sub_weights[(cat.id, s.zh)] for s in cat.subtopics], k=1
        )[0]
        genre = rng.choice(cat.genres)

        if length_bucket is None:
            bs = list(self._bucket_weights)
            length_bucket = rng.choices(bs, weights=[self._bucket_weights[b] for b in bs], k=1)[0]
        elif length_bucket not in self._buckets:
            raise ValueError(f"length_bucket 必须是 S/M/L，当前 {length_bucket!r}")

        return TopicSpec(
            category=cat.id,
            subtopic=sub.name(lang),
            genre=genre,
            lang=lang,
            grounding=cat.grounding and cat.id in self._grounding_categories,
            length_bucket=length_bucket,  # type: ignore[arg-type]
            target_len=self._target_len(rng, length_bucket),
        )

    def _target_len(self, rng: Random, bucket: str) -> int:
        """以桶中点为中心（S≈650 / M≈1000 / L≈1350），±1/6 桶宽抖动，始终落在桶区间内。"""
        lo, hi = self._buckets[bucket]
        center = (lo + hi) // 2 + ((lo + hi) % 2)
        span = (hi - lo) // 6
        return max(lo, min(hi, rng.randint(center - span, center + span)))

    # ------------------------------------------------------------------ 调整与统计

    def penalize(self, subtopic: str, factor: float = 0.5) -> None:
        """将子话题（中文或英文名）的采样权重乘以 factor（0 < factor ≤ 1），下限 1e-3。"""
        if not (0 < factor <= 1):
            raise ValueError("factor 必须在 (0, 1] 内")
        keys = self._sub_index.get(subtopic)
        if not keys:
            raise KeyError(subtopic)
        for k in keys:
            self._sub_weights[k] = max(_PENALTY_FLOOR, self._sub_weights[k] * factor)

    def coverage(self, counts: Mapping[str, int]) -> dict[str, float]:
        """各类别 当前占比 / 目标权重；1.0 表示恰好达标，库存为空时全部为 0。"""
        targets = self.target_weights()
        total = sum(max(0, int(counts.get(cid, 0))) for cid in targets)
        if total == 0:
            return {cid: 0.0 for cid in targets}
        return {cid: (max(0, int(counts.get(cid, 0))) / total) / t for cid, t in targets.items()}


__all__ = [
    "Category",
    "DEFAULT_GROUNDING_CATEGORIES",
    "GENRES",
    "GENRE_EN",
    "Subtopic",
    "TaxonomyError",
    "TopicSpec",
    "TopicTaxonomy",
]
