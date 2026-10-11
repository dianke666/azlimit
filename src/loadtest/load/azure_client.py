"""Azure OpenAI 客户端与错误分类（设计第 7 节）。

纯函数：
- ``classify``：(status, body, exc, finish_reason) → ``Outcome``
- ``is_quota_error``：403 响应体是否表明 quota / rate limit exceeded
- ``parse_retry_after``：``retry-after-ms`` → ``retry-after``（秒或 HTTP 日期）
- ``parse_rate_limit_headers``：``x-ratelimit-*`` → ``RateLimitHeaders``

``AzureClient``：封装 ``openai.AsyncAzureOpenAI``（max_retries=0），``send`` → ``AzureResult``。
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
import openai

from loadtest.outcome import Outcome, RateLimitHeaders

log = logging.getLogger(__name__)

MAX_RETRY_AFTER_S = 120.0

_TIMEOUT_EXC: tuple[type[BaseException], ...] = (
    TimeoutError,
    asyncio.TimeoutError,
    httpx.TimeoutException,
    openai.APITimeoutError,
)

_QUOTA_MARKERS = ("quota", "rate limit", "ratelimit", "rate_limit")


# ---------------------------------------------------------------------------
# 响应体辅助
# ---------------------------------------------------------------------------


def _error_obj(body: Any) -> Mapping[str, Any] | None:
    """返回 ``body["error"]``（若为映射），否则当 body 本身是映射时返回 body。"""
    if not isinstance(body, Mapping):
        return None
    err = body.get("error")
    if isinstance(err, Mapping):
        return err
    return body


def error_code(body: Any) -> str | None:
    """提取 Azure 错误码：``{"error": {"code": ...}}``；兼容顶层 ``code``。"""
    err = _error_obj(body)
    if err is None:
        return None
    code = err.get("code")
    return str(code) if code is not None else None


def is_quota_error(body: Any) -> bool:
    """403 响应体是否表明 quota / rate limit exceeded（不区分大小写，检查 code 与 message）。"""
    if body is None:
        return False
    if isinstance(body, (str, bytes)):
        text = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
        parts = [text]
    else:
        err = _error_obj(body)
        if err is None:
            return False
        parts = [str(err.get(k, "")) for k in ("code", "message", "type")]
    text = " ".join(parts).lower()
    return any(m in text for m in _QUOTA_MARKERS)


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------


def classify(
    status: int | None,
    body: Any,
    exc: BaseException | None,
    finish_reason: str | None = None,
) -> Outcome:
    """每个 (status, body, exc, finish_reason) 组合恰好映射到一个 Outcome；错误路径永不为 OK。"""
    if exc is not None and isinstance(exc, _TIMEOUT_EXC):
        return Outcome.TIMEOUT
    if status is None:
        return Outcome.SERVER_ERROR  # 连接失败等无状态码的网络异常
    if status in (429, 503):
        return Outcome.THROTTLED
    if status == 403:
        return Outcome.THROTTLED if is_quota_error(body) else Outcome.CLIENT_ERROR
    if status == 400 and error_code(body) == "content_filter":
        return Outcome.CONTENT_FILTER
    if status in (500, 502, 504):
        return Outcome.SERVER_ERROR
    if 400 <= status < 500:
        return Outcome.CLIENT_ERROR
    if 200 <= status < 300:
        if exc is not None:
            return Outcome.SERVER_ERROR  # 2xx 但解析/读取失败：不能算 OK
        return Outcome.CONTENT_FILTER if finish_reason == "content_filter" else Outcome.OK
    return Outcome.SERVER_ERROR  # 1xx/3xx/501/505 等非预期状态码


# ---------------------------------------------------------------------------
# 响应头
# ---------------------------------------------------------------------------


def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
    """大小写不敏感读取头（httpx.Headers 本身不敏感；普通 dict 需回退扫描）。"""
    if not headers:
        return None
    v = headers.get(name)
    if v is None:
        lname = name.lower()
        for k, val in headers.items():
            if isinstance(k, str) and k.lower() == lname:
                v = val
                break
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _to_float(raw: str) -> float | None:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _to_epoch(now: float | datetime) -> float:
    if isinstance(now, datetime):
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now.timestamp()
    return float(now)


def _parse_http_date(raw: str) -> float | None:
    try:
        d = parsedate_to_datetime(raw)
    except (TypeError, ValueError, IndexError):
        return None
    if d is None:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def parse_retry_after(
    headers: Mapping[str, str] | None, now: float | datetime
) -> float | None:
    """解析 Retry-After（秒）。

    顺序：``retry-after-ms``（毫秒 → 秒）→ ``retry-after``（秒数或 HTTP 日期）。
    无法解析或负值 → 视为缺失，继续下一个头；HTTP 日期早于 ``now`` → 0；
    > 120s 截断为 120 并告警。两者都缺失返回 None（调用方使用 min_cooldown_s）。
    ``now`` 为 epoch 秒或 datetime（naive 视为 UTC）。结果 ∈ [0, 120] ∪ {None}。
    """
    for name, scale in (("retry-after-ms", 1 / 1000), ("retry-after", 1.0)):
        raw = _header(headers, name)
        if raw is None:
            continue
        v = _to_float(raw)
        if v is None and name == "retry-after":
            ts = _parse_http_date(raw)
            if ts is not None:
                v = max(0.0, ts - _to_epoch(now))
        if v is None or v < 0:
            continue
        v *= scale
        if v > MAX_RETRY_AFTER_S:
            log.warning("retry-after %.0fs capped to %.0fs", v, MAX_RETRY_AFTER_S)
            v = MAX_RETRY_AFTER_S
        return v
    return None


def _to_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    v = _to_float(raw)
    if v is None or v < 0:
        return None
    return int(v)


def parse_rate_limit_headers(headers: Mapping[str, str] | None) -> RateLimitHeaders:
    """读取 ``x-ratelimit-remaining/limit-tokens/requests``；缺失或无法解析为 None。"""
    return RateLimitHeaders(
        remaining_tokens=_to_int(_header(headers, "x-ratelimit-remaining-tokens")),
        limit_tokens=_to_int(_header(headers, "x-ratelimit-limit-tokens")),
        remaining_requests=_to_int(_header(headers, "x-ratelimit-remaining-requests")),
        limit_requests=_to_int(_header(headers, "x-ratelimit-limit-requests")),
    )


# ---------------------------------------------------------------------------
# AzureClient（任务 12.1）
# ---------------------------------------------------------------------------


@dataclass
class AzureResult:
    """单次 Azure 调用的结果。不包含 API Key；``text`` 仅为模型输出（不含请求消息）。"""

    request_id: str
    outcome: Outcome
    status: int | None
    latency_ms: float
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    completion_tokens: int | None = None
    retry_after_s: float | None = None
    remaining_tokens: int | None = None
    remaining_requests: int | None = None
    headers: RateLimitHeaders = field(default_factory=RateLimitHeaders)
    error_code: str | None = None
    filter_detail: str | None = None  # 被过滤的类别，逗号分隔（如 "hate,violence"）
    finish_reason: str | None = None
    text: str | None = None


def _filtered_categories(result: Any) -> list[str]:
    """``{"hate": {"filtered": true, ...}, ...}`` → 被过滤的类别名列表。"""
    if not isinstance(result, Mapping):
        return []
    out: list[str] = []
    for k, v in result.items():
        if isinstance(v, Mapping) and v.get("filtered") is True:
            out.append(str(k))
    return out


def _error_filter_detail(body: Any) -> str | None:
    """从 400 错误体 ``error.innererror.content_filter_result`` 提取被过滤类别。"""
    err = _error_obj(body)
    if err is None:
        return None
    inner = err.get("innererror")
    cats: list[str] = []
    if isinstance(inner, Mapping):
        cats = _filtered_categories(inner.get("content_filter_result"))
        if not cats and inner.get("code"):
            return str(inner.get("code"))
    return ",".join(cats) or None


def _choice_filter_detail(raw_json: Any) -> str | None:
    """从 200 响应 ``choices[0].content_filter_results`` 提取被过滤类别。"""
    if not isinstance(raw_json, Mapping):
        return None
    choices = raw_json.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    return ",".join(_filtered_categories(choices[0].get("content_filter_results"))) or None


def _response_body(exc: openai.APIStatusError) -> Any:
    """优先读取原始 JSON 响应体；失败时回退到 ``exc.body``（SDK 可能已剥离外层 ``error``）。"""
    try:
        return exc.response.json()
    except Exception:  # noqa: BLE001 —— 非 JSON 响应体
        pass
    body = exc.body
    if isinstance(body, Mapping) and "error" not in body and ("code" in body or "message" in body):
        return {"error": dict(body)}
    return body


class AzureClient:
    """封装 ``openai.AsyncAzureOpenAI``：关闭 SDK 重试，每个请求只发送一次，读取限流头并分类。

    安全：从不记录 API Key 与消息内容；日志只含 request_id / status / outcome。
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment: str,
        *,
        timeout_s: float = 30.0,
        max_connections: int = 64,
        clock: Callable[[], float] | Any | None = None,
        http_client: httpx.AsyncClient | None = None,
        wall: Callable[[], float] = time.time,
        temperature: float | None = None,
    ) -> None:
        self.deployment = deployment
        self.temperature = temperature
        if clock is None:
            self._now: Callable[[], float] = time.monotonic
        elif callable(clock):
            self._now = clock
        else:
            self._now = clock.now
        self._wall = wall
        self._owns_http = http_client is None
        if http_client is None:
            http_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_connections,
                ),
                timeout=timeout_s,
            )
        self._http = http_client
        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            max_retries=0,  # 永不重试（需求 22.2/22.3）
            timeout=timeout_s,
            http_client=http_client,
        )

    @classmethod
    def from_settings(
        cls,
        azure: Any,
        load: Any | None = None,
        *,
        timeout_s: float = 30.0,
        clock: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
        wall: Callable[[], float] = time.time,
    ) -> AzureClient:
        """由 ``settings.azure``（AzureSettings）与 ``settings.load``（LoadSettings）构造。"""
        key = azure.api_key
        key = key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
        max_conn = int(getattr(load, "max_concurrency", 64)) if load is not None else 64
        return cls(
            str(azure.endpoint),
            key,
            azure.api_version,
            azure.deployment,
            timeout_s=timeout_s,
            max_connections=max_conn,
            clock=clock,
            http_client=http_client,
            wall=wall,
        )

    async def send(self, req: Any) -> AzureResult:
        """发送单个请求（恰好一次，不重试）并返回分类结果。永不抛出（取消除外）。"""
        request_id = str(req.request_id)
        kwargs: dict[str, Any] = {
            "model": self.deployment,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        t0 = self._now()
        status: int | None = None
        body: Any = None
        exc: BaseException | None = None
        headers: Mapping[str, str] | None = None
        finish_reason: str | None = None
        text: str | None = None
        usage: Any = None
        filter_detail: str | None = None
        try:
            raw = await self._client.chat.completions.with_raw_response.create(**kwargs)
            status = raw.status_code
            headers = raw.headers
            try:
                completion = raw.parse()
                if completion.choices:
                    choice = completion.choices[0]
                    finish_reason = choice.finish_reason
                    text = choice.message.content if choice.message else None
                usage = completion.usage
                if finish_reason == "content_filter":
                    try:
                        filter_detail = _choice_filter_detail(raw.http_response.json())
                    except Exception:  # noqa: BLE001
                        filter_detail = None
            except Exception as e:  # noqa: BLE001 —— 2xx 但响应体无法解析
                exc = e
        except openai.APIStatusError as e:
            status = e.status_code
            headers = e.response.headers
            body = _response_body(e)
            exc = e
        except openai.APITimeoutError as e:
            exc = e
        except openai.APIConnectionError as e:
            exc = e.__cause__ if isinstance(e.__cause__, _TIMEOUT_EXC) else e
        except Exception as e:  # noqa: BLE001 —— 其他网络/未知异常
            exc = e
        latency_ms = max(0.0, (self._now() - t0) * 1000.0)

        outcome = classify(status, body, exc, finish_reason)
        rl = parse_rate_limit_headers(headers)
        retry_after = parse_retry_after(headers, self._wall()) if headers is not None else None
        code = error_code(body) if body is not None else None
        if outcome is Outcome.CONTENT_FILTER and filter_detail is None and body is not None:
            filter_detail = _error_filter_detail(body)
        if exc is not None and code is None and status is None:
            code = type(exc).__name__

        prompt_tokens = completion_tokens = cached_tokens = None
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                cached_tokens = getattr(details, "cached_tokens", None)

        if outcome is not Outcome.OK:
            log.debug(
                "azure request %s -> status=%s outcome=%s code=%s",
                request_id, status, outcome.value, code,
            )
        return AzureResult(
            request_id=request_id,
            outcome=outcome,
            status=status,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            retry_after_s=retry_after,
            remaining_tokens=rl.remaining_tokens,
            remaining_requests=rl.remaining_requests,
            headers=rl,
            error_code=code,
            filter_detail=filter_detail,
            finish_reason=finish_reason,
            text=text,
        )

    async def aclose(self) -> None:
        """关闭底层连接（仅关闭自行创建的 http_client）。"""
        if self._owns_http:
            await self._client.close()

    async def __aenter__(self) -> AzureClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


__all__ = [
    "MAX_RETRY_AFTER_S",
    "AzureClient",
    "AzureResult",
    "classify",
    "error_code",
    "is_quota_error",
    "parse_rate_limit_headers",
    "parse_retry_after",
]
