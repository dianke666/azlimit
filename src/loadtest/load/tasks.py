"""任务指令库 TaskCatalog（Requirement 14.4 / 14.6 / 14.7）。

`tasks.yaml` 中每种语言定义 system_prompt、统一后缀 suffix 与指令列表；
加载时把后缀附加到每条指令，得到最终 `instruction`。
组装器（assembler）负责把 `instruction` 放在正文之前或之后。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Mapping

import yaml

DEFAULT_TASKS_PATH = Path(__file__).with_name("tasks.yaml")

LANGS: tuple[str, ...] = ("zh", "en")
MIN_TASKS_PER_LANG = 30

# Requirement 14.6：固定后缀，加载时校验 YAML 中的 suffix 与之一致。
REQUIRED_SUFFIX: dict[str, str] = {
    "zh": "只输出答案，不要解释，不超过 30 字。",
    "en": "Answer only, no explanation, under 20 words.",
}


class TaskCatalogError(ValueError):
    """tasks.yaml 结构或内容不合规。"""


@dataclass(frozen=True)
class TaskSpec:
    id: str
    lang: str
    system_prompt: str
    base_text: str       # 不含后缀的原始指令
    instruction: str     # base_text + 后缀，直接放入 user 消息


def _join(base: str, suffix: str, lang: str) -> str:
    base = base.strip()
    # 英文指令与后缀之间留一个空格；中文直接拼接。
    return f"{base}{suffix}" if lang == "zh" else f"{base} {suffix}"


class TaskCatalog:
    def __init__(self, tasks: list[TaskSpec]):
        by_lang: dict[str, list[TaskSpec]] = {lang: [] for lang in LANGS}
        for t in tasks:
            by_lang[t.lang].append(t)
        self._by_lang = {k: tuple(v) for k, v in by_lang.items()}

    # ---- 构造 ----
    @classmethod
    def load(cls, path: str | Path = DEFAULT_TASKS_PATH) -> "TaskCatalog":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Any) -> "TaskCatalog":
        if not isinstance(data, Mapping):
            raise TaskCatalogError("tasks.yaml 顶层必须是映射")
        errors: list[str] = []
        tasks: list[TaskSpec] = []
        seen_ids: set[str] = set()
        for lang in LANGS:
            section = data.get(lang)
            if not isinstance(section, Mapping):
                errors.append(f"{lang}: 缺少语言段")
                continue
            system_prompt = str(section.get("system_prompt") or "").strip()
            suffix = str(section.get("suffix") or "").strip()
            if not system_prompt:
                errors.append(f"{lang}: system_prompt 为空")
            if suffix != REQUIRED_SUFFIX[lang]:
                errors.append(f"{lang}: suffix 必须为 {REQUIRED_SUFFIX[lang]!r}")
            items = section.get("tasks") or []
            count = 0
            for item in items:
                tid = str((item or {}).get("id") or "").strip()
                text = str((item or {}).get("text") or "").strip()
                if not tid or not text:
                    errors.append(f"{lang}: 存在缺少 id 或 text 的指令")
                    continue
                if tid in seen_ids:
                    errors.append(f"{lang}: 重复的指令 id {tid}")
                    continue
                seen_ids.add(tid)
                tasks.append(TaskSpec(tid, lang, system_prompt, text,
                                      _join(text, REQUIRED_SUFFIX[lang], lang)))
                count += 1
            if count < MIN_TASKS_PER_LANG:
                errors.append(f"{lang}: 指令数 {count} < {MIN_TASKS_PER_LANG}")
        if errors:
            raise TaskCatalogError("; ".join(errors))
        return cls(tasks)

    # ---- 查询 ----
    def for_lang(self, lang: str) -> tuple[TaskSpec, ...]:
        try:
            return self._by_lang[lang]
        except KeyError:
            raise ValueError(f"unsupported lang: {lang!r}") from None

    def count(self, lang: str) -> int:
        return len(self.for_lang(lang))

    def sample(self, rng: Random, lang: str) -> TaskSpec:
        """在同语言子集内均匀抽取一条指令（Requirement 14.4）。"""
        return rng.choice(self.for_lang(lang))
