"""负载阶段共享的数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Lang = Literal["zh", "en"]
InstructionPosition = Literal["before", "after"]


@dataclass(frozen=True)
class PlannedRequest:
    """一个已组装、待派发的请求（design.md 数据模型）。

    ``input_tokens`` 为对完整 ``messages`` 以 o200k_base 复算的结果（含消息开销）。
    ``initial_passage_count`` 为按预算分档选取的初始段落数（不含补足片段，Req 14.9），
    ``filler_count`` 为补足片段数，二者之和 == ``len(passage_ids)``。
    """

    request_id: str
    lang: Lang
    messages: list[dict]
    input_tokens: int
    max_tokens: int  # 40
    passage_ids: tuple[int, ...]
    task_id: str
    target_tokens: int
    instruction_position: InstructionPosition = "after"
    initial_passage_count: int = 0
    filler_count: int = 0
    budget_tokens: int = field(default=0, compare=False)  # 段落预算（target − 指令/消息开销）


__all__ = ["PlannedRequest", "Lang", "InstructionPosition"]
