"""Deduplicator：SHA256 精确匹配 + 自实现 64 位 SimHash（汉明距离 ≤ 3）。

- 精确：删除全部空白字符与 Unicode 标点（类别 P*）后的 UTF-8 文本取 SHA256（Requirement 6.2）。
- 近似：64 位 SimHash；中文按字符 3-gram、英文按小写词 3-shingle 取特征，特征哈希为 blake2b-64，
  按出现次数加权；汉明距离 ≤ 3 视为重复（Requirement 6.3）。
- 索引：鸽巢原理，64 位分成 4 个 16 位 band，距离 ≤ 3 的两个指纹至少有一个 band 完全相同；
  只需比对同 band 的候选。CorpusStore 在 SQLite 中对 4 个 band 建表达式索引，
  本模块的 :class:`Deduplicator` 是等价的内存实现（属性测试 / 批内去重使用）。
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Literal

HAMMING_THRESHOLD = 3
SIMHASH_BITS = 64
BAND_BITS = 16
NUM_BANDS = SIMHASH_BITS // BAND_BITS  # 4 = HAMMING_THRESHOLD + 1，满足鸽巢条件
_MASK64 = (1 << 64) - 1
_BAND_MASK = (1 << BAND_BITS) - 1

ZH_NGRAM = 3
EN_SHINGLE = 3

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


# --------------------------------------------------------------------------- 规范化与精确哈希


def normalize(text: str) -> str:
    """删除全部空白字符与 Unicode 标点（general category 以 P 开头）。"""
    return "".join(
        ch for ch in text if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def sha256_of(text: str) -> str:
    """规范化文本的 SHA256 十六进制摘要。"""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- SimHash


def _feature_hash(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


def _detect_lang(text: str) -> Literal["zh", "en"]:
    letters = sum(1 for ch in text if ch.isalpha())
    if not letters:
        return "en"
    cjk = len(_CJK_RE.findall(text))
    return "zh" if cjk * 2 >= letters else "en"


def _ngrams(units: list[str], n: int, sep: str) -> Iterator[str]:
    if len(units) < n:
        if units:
            yield sep.join(units)
        return
    for i in range(len(units) - n + 1):
        yield sep.join(units[i : i + n])


def features(text: str, lang: Literal["zh", "en"] | None = None) -> Counter[str]:
    """SimHash 特征：中文为规范化文本的字符 3-gram；英文为小写词 3-shingle。"""
    lang = lang or _detect_lang(text)
    if lang == "zh":
        return Counter(_ngrams(list(normalize(text)), ZH_NGRAM, ""))
    words = [w.lower() for w in _WORD_RE.findall(text)]
    return Counter(_ngrams(words, EN_SHINGLE, " "))


def simhash64(text: str, lang: Literal["zh", "en"] | None = None) -> int:
    """64 位 SimHash（无符号整数）。``lang`` 为空时按 CJK 字符占比自动判断。"""
    feats = features(text, lang)
    if not feats:
        return 0
    v = [0] * SIMHASH_BITS
    for feat, weight in feats.items():
        h = _feature_hash(feat)
        for i in range(SIMHASH_BITS):
            if (h >> i) & 1:
                v[i] += weight
            else:
                v[i] -= weight
    out = 0
    for i, s in enumerate(v):
        if s > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    """两个 64 位指纹（有符号或无符号均可）的汉明距离。"""
    return ((a ^ b) & _MASK64).bit_count()


def bands(h: int) -> tuple[int, ...]:
    """把 64 位指纹拆成 4 个 16 位 band（band i = (h >> 16i) & 0xFFFF）。"""
    h &= _MASK64
    return tuple((h >> (BAND_BITS * i)) & _BAND_MASK for i in range(NUM_BANDS))


# --------------------------------------------------------------------------- Deduplicator


@dataclass(frozen=True)
class Fingerprint:
    sha256: str
    simhash: int  # 无符号 64 位

    @classmethod
    def of(cls, text: str, lang: Literal["zh", "en"] | None = None) -> "Fingerprint":
        return cls(sha256_of(text), simhash64(text, lang))


@dataclass(frozen=True)
class DupMatch:
    kind: Literal["exact", "near"]
    distance: int
    passage_id: int | None = None


class Deduplicator:
    """内存去重索引（SHA256 集合 + 4 band SimHash 表）。

    ``CorpusStore.insert_text`` 直接在 SQLite 上执行等价查询（覆盖所有状态段落、跨连接一致）；
    本类用于无数据库场景（属性测试、批内去重）或由 :meth:`CorpusStore.load_deduplicator` 预加载。
    """

    def __init__(self, threshold: int = HAMMING_THRESHOLD) -> None:
        if threshold >= NUM_BANDS:
            raise ValueError(f"threshold must be < {NUM_BANDS} for band lookup to be exact")
        self.threshold = threshold
        self._sha: dict[str, int | None] = {}
        self._bands: list[dict[int, list[tuple[int, int | None]]]] = [{} for _ in range(NUM_BANDS)]

    def __len__(self) -> int:
        return len(self._sha)

    def add_fingerprint(self, sha256: str, simhash: int, passage_id: int | None = None) -> None:
        simhash &= _MASK64
        self._sha.setdefault(sha256, passage_id)
        for i, b in enumerate(bands(simhash)):
            self._bands[i].setdefault(b, []).append((simhash, passage_id))

    def add(self, text: str, passage_id: int | None = None, lang: Literal["zh", "en"] | None = None) -> Fingerprint:
        fp = Fingerprint.of(text, lang)
        self.add_fingerprint(fp.sha256, fp.simhash, passage_id)
        return fp

    def extend(self, items: Iterable[tuple[str, int, int | None]]) -> None:
        for sha, sh, pid in items:
            self.add_fingerprint(sha, sh, pid)

    def match_fingerprint(self, sha256: str, simhash: int) -> DupMatch | None:
        if sha256 in self._sha:
            return DupMatch("exact", 0, self._sha[sha256])
        best: DupMatch | None = None
        for i, b in enumerate(bands(simhash)):
            for cand, pid in self._bands[i].get(b, ()):
                d = hamming(simhash, cand)
                if d <= self.threshold and (best is None or d < best.distance):
                    best = DupMatch("near", d, pid)
        return best

    def check(self, text: str, lang: Literal["zh", "en"] | None = None) -> DupMatch | None:
        return self.match_fingerprint(sha256_of(text), simhash64(text, lang))

    def is_duplicate(self, text: str, lang: Literal["zh", "en"] | None = None) -> bool:
        return self.check(text, lang) is not None


__all__ = [
    "BAND_BITS",
    "Deduplicator",
    "DupMatch",
    "Fingerprint",
    "HAMMING_THRESHOLD",
    "NUM_BANDS",
    "bands",
    "features",
    "hamming",
    "normalize",
    "sha256_of",
    "simhash64",
]
