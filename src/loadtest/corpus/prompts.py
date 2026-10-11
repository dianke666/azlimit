"""Gemini 生成提示词模板（Requirement 3.2 / 3.3 / 3.4）与黑名单加载（Requirement 5.7）。

- `build_prompt(spec, taxonomy)` 按 spec.lang 选用中文或英文模板；
  grounding 类别额外附加"仅依据检索到的近期报道、注明事件日期"的要求。
- 长度换算（o200k_base）：中文约 1.0–1.3 字/token，英文约 0.75 词/token。
  这里只是给模型的目标长度提示，入库判定以实测 token 数为准。
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from importlib import resources
from pathlib import Path

from .taxonomy import TopicSpec, TopicTaxonomy

ZH_CHARS_PER_TOKEN: tuple[float, float] = (1.0, 1.3)
EN_WORDS_PER_TOKEN = 0.75
# 不依赖 locale 的英文月份名
_MONTHS_EN: tuple[str, ...] = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


# ---------------------------------------------------------------- 黑名单

def load_blocklist(path: str | Path | None = None) -> frozenset[str]:
    """加载黑名单：每行一个关键词，忽略空行与 # 注释行，统一转小写。"""
    if path is None:
        text = resources.files("loadtest.corpus").joinpath("blocklist.txt").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    words: set[str] = set()
    for line in text.splitlines():
        w = line.strip()
        if not w or w.startswith("#"):
            continue
        words.add(w.lower())
    return frozenset(words)


# ---------------------------------------------------------------- 长度换算

def target_chars(spec: TopicSpec) -> tuple[int, int]:
    """中文目标字数区间 = target_len × [1.0, 1.3]。"""
    lo, hi = ZH_CHARS_PER_TOKEN
    return round(spec.target_len * lo), round(spec.target_len * hi)


def target_words(spec: TopicSpec) -> int:
    """英文目标词数 = target_len × 0.75。"""
    return round(spec.target_len * EN_WORDS_PER_TOKEN)


# ---------------------------------------------------------------- 模板

_ZH_TEMPLATE = (
    "你是一名{category}领域的资深撰稿人。请用简体中文写一篇关于「{subtopic}」的{genre}。\n"
    "要求：\n"
    "1. 长度约 {lo}–{hi} 字，分 3–6 个自然段，每段 2–5 句，段落之间空一行，结尾句子完整。\n"
    "2. 只写公认事实与通用知识；不确定的内容不要写；不要编造人名、数据、引语或文献。\n"
    "3. 不涉及政治、宗教、暴力、色情、违法活动、医疗诊断、投资建议、争议话题和个人隐私。\n"
    "4. 不使用 Markdown 标题、列表符号、表格或加粗；不要开场白和结束语，不要出现\u201c作为AI\u201d之类的表述，直接输出正文。"
)

_ZH_GROUNDING = (
    "5. 今天是 {today}。仅依据检索到的最近几天的真实报道撰写综述，不推测、不评论；"
    "排除任何政治相关内容；正文中注明每个事件的日期（格式为 YYYY-MM-DD 或 X月X日）；"
    "涉及市场与经济时只陈述事实，不作任何投资建议。"
)

_EN_TEMPLATE = (
    "You are an experienced writer on {category}. Write a {genre} in English about \"{subtopic}\".\n"
    "Requirements:\n"
    "1. About {words} words, 3–6 paragraphs of 2–5 sentences each, separated by a blank line, "
    "ending with a complete sentence.\n"
    "2. State only well-established facts and general knowledge; omit anything uncertain; "
    "never invent names, figures, quotes or references.\n"
    "3. Avoid politics, religion, violence, sexual content, illegal activity, medical diagnosis, "
    "investment advice, controversial topics and personal data.\n"
    "4. No Markdown headings, bullets, tables or bold text; no preamble or closing remarks; "
    "do not refer to yourself as an AI. Output the article text only."
)

_EN_GROUNDING = (
    "5. Today is {today}. Base the roundup only on real news reports from the past few days "
    "found by search; do not speculate or editorialize; exclude anything political; "
    "state the date of each event in the text (YYYY-MM-DD or Month D); "
    "for markets and the economy report facts only and give no investment advice."
)


def build_prompt(spec: TopicSpec, taxonomy: TopicTaxonomy, today: date | None = None) -> str:
    """按 spec.lang 生成 Gemini 提示词；grounding 规格附加检索与日期要求。"""
    if today is None:
        today = datetime.now(timezone.utc).date()
    category = taxonomy.category_label(spec.category, spec.lang)
    genre = taxonomy.genre_label(spec.genre, spec.lang)

    if spec.lang == "zh":
        lo, hi = target_chars(spec)
        prompt = _ZH_TEMPLATE.format(category=category, subtopic=spec.subtopic, genre=genre, lo=lo, hi=hi)
        if spec.grounding:
            prompt += "\n" + _ZH_GROUNDING.format(today=f"{today.isoformat()}（{today.month}月{today.day}日）")
    elif spec.lang == "en":
        prompt = _EN_TEMPLATE.format(category=category, subtopic=spec.subtopic, genre=genre,
                                     words=target_words(spec))
        if spec.grounding:
            prompt += "\n" + _EN_GROUNDING.format(today=f"{today.isoformat()} ({_MONTHS_EN[today.month - 1]} {today.day})")
    else:
        raise ValueError(f"unsupported lang: {spec.lang!r}")
    return prompt


__all__ = ["build_prompt", "load_blocklist", "target_chars", "target_words"]
