"""共享的结果分类类型：``Outcome`` 与 ``RateLimitHeaders``。

由 ThrottleController、AzureClient（``classify``）、Metrics 等模块共同引用，
放在独立模块中以避免循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Outcome(Enum):
    OK = "ok"
    THROTTLED = "throttled"  # 429, 403 配额, 503 —— 不重试
    CONTENT_FILTER = "content_filter"  # 400 + code=content_filter，或 finish_reason=content_filter
    CLIENT_ERROR = "client_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class RateLimitHeaders:
    """``x-ratelimit-remaining/limit-tokens/requests`` 响应头；缺失为 None。"""

    remaining_tokens: int | None = None
    limit_tokens: int | None = None
    remaining_requests: int | None = None
    limit_requests: int | None = None


__all__ = ["Outcome", "RateLimitHeaders"]
