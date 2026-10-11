"""SafetyFilter：语料入库前的逐条质量 / 合规校验（Requirement 5）。

校验顺序（Requirement 5.12）：
    finish_reason → length → language → refusal → blocklist → structure → grounding
在第一个失败处停止，只返回该首个原因。去重（``duplicate``）由 CorpusStore 在其后执行。

- finish_reason：``SAFETY`` 或正文为空 → ``safety_block``（调用方负责把子话题权重 × 0.5）；
  既不是 STOP 也不是 SAFETY → ``structure``。接受字符串（大小写不敏感，允许 ``FinishReason.STOP``
  形式）或带 ``.name`` 的枚举对象。
- length：o200k_base token 数 ∉ [min_tokens, max_tokens] → ``length``；通过则归入 S/M/L。
- language：分母为非空白字符；zh 要求 CJK ≥ 60%；en 要求 [A-Za-z] ≥ 80% 且 CJK < 2%。
- refusal：拒答 / 免责模板正则。
- blocklist：含 CJK 的词按子串匹配；其余按单词边界、大小写不敏感匹配。
- structure：段落 = 以换行分隔的非空文本块；< 3 段、存在以 ``#`` 开头的行、
  或两个段落经 :func:`loadtest.corpus.dedup.normalize` 后相同 → ``structure``。
- grounding：``spec.grounding`` 时需 ≥ 1 个来源且正文含日期 → ``grounding``。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from loadtest.corpus.dedup import normalize

if TYPE_CHECKING:  # pragma: no cover
    from loadtest.corpus.taxonomy import TopicSpec
    from loadtest.settings import CorpusSettings
    from loadtest.tokens import TokenCounter

Reason = Literal["safety_block", "structure", "length", "language", "refusal", "blocklist", "grounding"]

DEFAULT_BUCKETS: dict[str, tuple[int, int]] = {"S": (500, 799), "M": (800, 1199), "L": (1200, 1500)}

ZH_CJK_MIN = 0.60
EN_ALPHA_MIN = 0.80
EN_CJK_MAX = 0.02  # en 要求 CJK 占比 < 2%

# CJK 统一表意文字（基本区、扩展 A、兼容区、扩展 B 及以后）
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")
_HAS_CJK_RE = _CJK_RE

# --------------------------------------------------------------------------- 拒答 / 免责模板

_APOS = r"['’]"
_REFUSAL_PATTERNS: tuple[str, ...] = (
    # 中文
    r"作为\s*(?:一[个名款种]\s*)?(?:AI|ＡＩ|人工智能|语言模型|大语言模型|大型语言模型|AI\s*助手|AI\s*模型)",
    r"我(?:是|只是)\s*(?:一[个名款种]\s*)?(?:AI|人工智能|语言模型|大语言模型)",
    r"我无法",
    r"我不能(?:提供|回答|协助|帮助|满足|完成|生成)",
    r"抱歉\s*[，,、]?\s*我(?:不能|无法|没办法)",
    r"对不起\s*[，,、]?\s*我(?:不能|无法|没办法)",
    r"很抱歉\s*[，,、]?\s*(?:我|但)",
    # 英文
    r"\bI\s+can" + _APOS + r"?t\b",
    r"\bI\s+cannot\b",
    r"\bI\s+am\s+unable\s+to\b",
    r"\bI" + _APOS + r"m\s+unable\s+to\b",
    r"\bI" + _APOS + r"m\s+sorry\s*,?\s+but\b",
    r"\bI\s+am\s+sorry\s*,?\s+but\b",
    r"\bI\s+apologi[sz]e\s*,?\s+but\b",
    r"\bAs\s+an?\s+(?:AI|artificial\s+intelligence|language\s+model|large\s+language\s+model|AI\s+language\s+model)\b",
    r"\bI\s+(?:am|" + _APOS + r"m)\s+(?:just\s+)?an?\s+(?:AI|language\s+model)\b",
)
REFUSAL_RE = re.compile("|".join(f"(?:{p})" for p in _REFUSAL_PATTERNS), re.IGNORECASE)

# --------------------------------------------------------------------------- 日期

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_MON_NUM = r"(?:0?[1-9]|1[0-2])"
_ORD = r"(?:st|nd|rd|th)?"
_DATE_PATTERNS: tuple[str, ...] = (
    # 2026-10-11 / 2026/10/11 / 2026.10.11
    rf"(?<!\d)\d{{4}}\s*[-/.]\s*{_MON_NUM}\s*[-/.]\s*{_DAY}(?!\d)",
    # 2026年3月5日 / 3月5日 / 3月5号（年份可选）
    rf"(?:\d{{4}}\s*年\s*)?(?<!\d){_MON_NUM}\s*月\s*{_DAY}\s*[日号]",
    # March 3 / March 3rd, 2026 / Mar. 3
    rf"\b{_MONTHS}\s+{_DAY}{_ORD}\b",
    # 3 March / 3rd of March 2026
    rf"\b{_DAY}{_ORD}\s+(?:of\s+)?{_MONTHS}(?![A-Za-z])",
)
DATE_RE = re.compile("|".join(f"(?:{p})" for p in _DATE_PATTERNS), re.IGNORECASE)


def contains_date(text: str) -> bool:
    """正文是否含"年-月-日"或"月-日"形式的日期（中英文常见写法）。"""
    return DATE_RE.search(text) is not None


# --------------------------------------------------------------------------- 结果


@dataclass(frozen=True)
class FilterResult:
    ok: bool
    reason: str | None
    token_len: int
    length_bucket: str | None = None


# --------------------------------------------------------------------------- 辅助


def _finish_name(finish_reason: Any) -> str:
    """把 finish_reason 规范成大写名称：支持枚举（``.name``）、``FinishReason.STOP`` 字符串等。"""
    if finish_reason is None:
        return ""
    name = getattr(finish_reason, "name", None)
    s = name if isinstance(name, str) else str(finish_reason)
    return s.strip().rsplit(".", 1)[-1].upper()


def char_ratios(text: str) -> tuple[float, float]:
    """返回 (CJK 占比, ASCII 字母占比)，分母为非空白字符数；无非空白字符时为 (0, 0)。"""
    total = cjk = alpha = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            alpha += 1
        elif _CJK_RE.match(ch):
            cjk += 1
    if total == 0:
        return 0.0, 0.0
    return cjk / total, alpha / total


def language_ok(text: str, lang: str) -> bool:
    cjk, alpha = char_ratios(text)
    if lang == "zh":
        return cjk >= ZH_CJK_MIN
    if lang == "en":
        return alpha >= EN_ALPHA_MIN and cjk < EN_CJK_MAX
    raise ValueError(f"unsupported lang: {lang!r}")


def paragraphs(text: str) -> list[str]:
    """以换行分隔的非空文本块（去除首尾空白）。"""
    return [p.strip() for p in text.splitlines() if p.strip()]


def _compile_blocklist(words: Iterable[str]) -> tuple[tuple[str, ...], re.Pattern[str] | None]:
    zh: set[str] = set()
    en: set[str] = set()
    for w in words:
        w = w.strip()
        if not w:
            continue
        (zh if _HAS_CJK_RE.search(w) else en).add(w)
    en_re = None
    if en:
        # 长词优先，避免短词抢先匹配；边界为非字母数字（单词边界）
        alts = "|".join(re.escape(w) for w in sorted(en, key=len, reverse=True))
        en_re = re.compile(rf"(?<![0-9A-Za-z_])(?:{alts})(?![0-9A-Za-z_])", re.IGNORECASE)
    return tuple(sorted(zh)), en_re


# --------------------------------------------------------------------------- SafetyFilter


class SafetyFilter:
    """按 Requirement 5.12 的顺序执行校验，首个失败即返回。"""

    def __init__(
        self,
        blocklist: Iterable[str],
        *,
        min_tokens: int = 500,
        max_tokens: int = 1500,
        buckets: Mapping[str, tuple[int, int]] | None = None,
        counter: TokenCounter | None = None,
    ) -> None:
        if min_tokens > max_tokens:
            raise ValueError("min_tokens must be <= max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.buckets: dict[str, tuple[int, int]] = dict(buckets or DEFAULT_BUCKETS)
        if not self.buckets:
            raise ValueError("buckets must not be empty")
        self._counter = counter
        self.blocklist: frozenset[str] = frozenset(w.strip() for w in blocklist if w and w.strip())
        self._zh_block, self._en_block_re = _compile_blocklist(self.blocklist)

    # ---- 构造 ---------------------------------------------------------------
    @classmethod
    def default(
        cls,
        path: Any = None,
        *,
        settings: CorpusSettings | None = None,
        counter: TokenCounter | None = None,
    ) -> SafetyFilter:
        """使用包内 ``blocklist.txt``（或 ``path``）构造；可选按 CorpusSettings 设置长度阈值。"""
        from loadtest.corpus.prompts import load_blocklist  # 延迟导入，避免循环依赖

        kwargs: dict[str, Any] = {"counter": counter}
        if settings is not None:
            kwargs.update(
                min_tokens=settings.passage_min_tokens,
                max_tokens=settings.passage_max_tokens,
                buckets=settings.length_buckets,
            )
        return cls(load_blocklist(path), **kwargs)

    @property
    def counter(self) -> TokenCounter:
        if self._counter is None:
            from loadtest.tokens import default_counter

            self._counter = default_counter()
        return self._counter

    # ---- 单项规则（亦供测试 / 调用方单独使用） --------------------------------
    def bucket_of(self, token_len: int) -> str:
        for name, (lo, hi) in self.buckets.items():
            if lo <= token_len <= hi:
                return name
        # 桶之间有空隙时归入距离最近的桶
        return min(
            self.buckets,
            key=lambda b: min(abs(token_len - self.buckets[b][0]), abs(token_len - self.buckets[b][1])),
        )

    def is_refusal(self, text: str) -> bool:
        return REFUSAL_RE.search(text) is not None

    def blocklist_hit(self, text: str) -> str | None:
        for w in self._zh_block:
            if w in text:
                return w
        if self._en_block_re is not None:
            m = self._en_block_re.search(text)
            if m:
                return m.group(0)
        return None

    @staticmethod
    def structure_ok(text: str) -> bool:
        paras = paragraphs(text)
        if len(paras) < 3:
            return False
        if any(p.startswith(("#", "＃")) for p in paras):
            return False
        seen: set[str] = set()
        for p in paras:
            key = normalize(p)
            if key in seen:
                return False
            seen.add(key)
        return True

    @staticmethod
    def grounding_ok(text: str, sources: Sequence[Any] | Iterable[Any]) -> bool:
        return len(list(sources or ())) >= 1 and contains_date(text)

    # ---- 主入口 -------------------------------------------------------------
    def check(
        self,
        text: str | None,
        spec: TopicSpec,
        finish_reason: Any,
        sources: Sequence[Any] | Iterable[Any] = (),
        token_len: int | None = None,
    ) -> FilterResult:
        text = text or ""
        given_len = token_len if token_len is not None else 0

        # 1. finish_reason / 空正文
        fr = _finish_name(finish_reason)
        if fr == "SAFETY" or not text.strip():
            return FilterResult(False, "safety_block", given_len)
        if fr != "STOP":
            return FilterResult(False, "structure", given_len)

        # 2. length
        n = token_len if token_len is not None else self.counter.count(text)
        if not (self.min_tokens <= n <= self.max_tokens):
            return FilterResult(False, "length", n)
        bucket = self.bucket_of(n)

        # 3. language
        if not language_ok(text, spec.lang):
            return FilterResult(False, "language", n, bucket)
        # 4. refusal
        if self.is_refusal(text):
            return FilterResult(False, "refusal", n, bucket)
        # 5. blocklist
        if self.blocklist_hit(text) is not None:
            return FilterResult(False, "blocklist", n, bucket)
        # 6. structure
        if not self.structure_ok(text):
            return FilterResult(False, "structure", n, bucket)
        # 7. grounding
        if spec.grounding and not self.grounding_ok(text, sources):
            return FilterResult(False, "grounding", n, bucket)
        return FilterResult(True, None, n, bucket)


__all__ = [
    "DATE_RE",
    "DEFAULT_BUCKETS",
    "FilterResult",
    "REFUSAL_RE",
    "SafetyFilter",
    "char_ratios",
    "contains_date",
    "language_ok",
    "paragraphs",
]
