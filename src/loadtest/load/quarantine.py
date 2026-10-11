"""内容过滤隔离（Requirements 24.1, 24.2）。

``quarantine_passages`` 先从内存索引移除段落（同一运行立即不可再选，即使随后的数据库写入
失败），再在 Corpus_Store 中置为 quarantined 并以原因 ``azure_content_filter`` 写入 quarantine 表。
``extract_filter_categories`` 从 Azure 错误体 / 响应体中提取命中的过滤类别。
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

QUARANTINE_REASON = "azure_content_filter"
FILTER_CATEGORIES: tuple[str, ...] = ("hate", "sexual", "violence", "self_harm", "jailbreak")


async def quarantine_passages(
    store: Any,
    index: Any,
    passage_ids: Iterable[int],
    detail: str | None = None,
    run_id: str | None = None,
) -> list[int]:
    """隔离请求涉及的全部段落；返回去重后的段落 id 列表。"""
    ids = list(dict.fromkeys(int(i) for i in passage_ids))
    if not ids:
        return ids
    for pid in ids:
        index.remove(pid)  # 先移除：DB 写入失败也不会在本次运行中复用
    await store.quarantine(ids, QUARANTINE_REASON, detail, run_id=run_id)
    return ids


def _walk(obj: Any, hits: set[str], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            if isinstance(k, str) and k in FILTER_CATEGORIES and isinstance(v, Mapping):
                if v.get("filtered") is True:
                    hits.add(k)
            _walk(v, hits, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, hits, depth + 1)


def extract_filter_categories(source: Any) -> str | None:
    """返回命中（``filtered: true``）的过滤类别，按固定顺序以逗号连接；无法获得时 None。

    ``source`` 可以是错误体 / 响应体（dict、JSON 字符串或 bytes），或带 ``body`` /
    ``error_body`` / ``raw`` 属性的结果对象。兼容
    ``error.innererror.content_filter_result``、``prompt_filter_results[*].content_filter_results``
    与 ``choices[*].content_filter_results``。
    """
    if source is None:
        return None
    if not isinstance(source, (Mapping, list, tuple, str, bytes, bytearray)):
        for attr in ("body", "error_body", "raw", "response_body"):
            val = getattr(source, attr, None)
            if val is not None:
                source = val
                break
        else:
            return None
    if isinstance(source, (bytes, bytearray)):
        source = source.decode("utf-8", "replace")
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except ValueError:
            return None
    hits: set[str] = set()
    _walk(source, hits)
    if not hits:
        return None
    return ",".join(c for c in FILTER_CATEGORIES if c in hits)


__all__ = ["quarantine_passages", "extract_filter_categories", "QUARANTINE_REASON", "FILTER_CATEGORIES"]
