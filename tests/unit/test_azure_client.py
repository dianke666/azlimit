"""AzureClient 单元测试（任务 12.2）：respx 模拟 Azure 端点。

覆盖需求 19.3, 19.5, 22.2, 22.3, 22.5, 24.4：
每个请求恰好发送一次（不重试），并正确分类 / 读取 usage 与限流头。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import format_datetime
from datetime import datetime, timezone

import httpx
import pytest
import respx

from loadtest.clock import FakeClock
from loadtest.load.azure_client import AzureClient, AzureResult
from loadtest.outcome import Outcome

ENDPOINT = "https://unit-test.openai.azure.com"
DEPLOYMENT = "gpt-4.1-nano"
API_VERSION = "2024-10-21"
API_KEY = "sk-unit-test-secret-key"
URL = f"{ENDPOINT}/openai/deployments/{DEPLOYMENT}/chat/completions"
WALL_NOW = 1_700_000_000.0


@dataclass
class Req:
    request_id: str = "req-1"
    messages: list[dict] = field(
        default_factory=lambda: [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "SECRET-PASSAGE-CONTENT"},
        ]
    )
    max_tokens: int = 40
    lang: str = "zh"


def _completion(finish_reason: str = "stop", content: str | None = "ok", cached: int | None = 128,
                choice_extra: dict | None = None) -> dict:
    usage: dict = {"prompt_tokens": 2500, "completion_tokens": 37, "total_tokens": 2537}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    choice: dict = {
        "index": 0,
        "finish_reason": finish_reason,
        "message": {"role": "assistant", "content": content},
    }
    if choice_extra:
        choice.update(choice_extra)
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": DEPLOYMENT,
        "choices": [choice],
        "usage": usage,
    }


RL_HEADERS = {
    "x-ratelimit-remaining-tokens": "987654",
    "x-ratelimit-limit-tokens": "1000000",
    "x-ratelimit-remaining-requests": "990",
    "x-ratelimit-limit-requests": "1000",
}


@pytest.fixture
async def client():
    c = AzureClient(ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT, timeout_s=5,
                    clock=FakeClock(), wall=lambda: WALL_NOW)
    yield c
    await c.aclose()


@pytest.fixture
def mock():
    with respx.mock(assert_all_called=False) as m:
        yield m


def _route(mock, response=None, side_effect=None):
    route = mock.post(URL, params={"api-version": API_VERSION})
    if side_effect is not None:
        route.side_effect = side_effect
    else:
        route.return_value = response
    return route


# ---------------------------------------------------------------------------
# 200
# ---------------------------------------------------------------------------


async def test_ok_reads_usage_and_ratelimit_headers(client, mock):
    route = _route(mock, httpx.Response(200, json=_completion(), headers=RL_HEADERS))
    r = await client.send(Req())
    assert route.call_count == 1
    assert isinstance(r, AzureResult)
    assert r.request_id == "req-1"
    assert r.outcome is Outcome.OK
    assert r.status == 200
    assert (r.prompt_tokens, r.cached_tokens, r.completion_tokens) == (2500, 128, 37)
    assert r.remaining_tokens == 987654 and r.remaining_requests == 990
    assert r.headers.limit_tokens == 1_000_000 and r.headers.limit_requests == 1000
    assert r.retry_after_s is None
    assert r.finish_reason == "stop" and r.text == "ok"
    assert r.error_code is None and r.filter_detail is None
    assert r.latency_ms >= 0

    sent = route.calls.last.request
    assert sent.headers["api-key"] == API_KEY
    import json
    payload = json.loads(sent.content)
    assert payload["max_tokens"] == 40
    assert payload["messages"][1]["content"] == "SECRET-PASSAGE-CONTENT"


async def test_ok_without_cached_details(client, mock):
    route = _route(mock, httpx.Response(200, json=_completion(cached=None)))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.OK
    assert r.cached_tokens is None
    assert r.remaining_tokens is None and r.remaining_requests is None


async def test_ok_finish_reason_content_filter(client, mock):
    extra = {"content_filter_results": {
        "hate": {"filtered": False, "severity": "safe"},
        "violence": {"filtered": True, "severity": "medium"},
    }}
    route = _route(mock, httpx.Response(
        200, json=_completion(finish_reason="content_filter", content=None, choice_extra=extra)))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.CONTENT_FILTER
    assert r.status == 200
    assert r.finish_reason == "content_filter"
    assert r.filter_detail == "violence"
    assert r.prompt_tokens == 2500


async def test_ok_malformed_body_is_not_ok(client, mock):
    route = _route(mock, httpx.Response(200, content=b"not json",
                                        headers={"content-type": "application/json"}))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.SERVER_ERROR


# ---------------------------------------------------------------------------
# 429 / 503 / 403
# ---------------------------------------------------------------------------

_ERR_429 = {"error": {"code": "429", "message": "Requests exceeded rate limit"}}


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after": "7"}, 7.0),
        ({"retry-after": format_datetime(
            datetime.fromtimestamp(WALL_NOW + 12, tz=timezone.utc), usegmt=True)}, 12.0),
        ({"retry-after": "600"}, 120.0),
        ({}, None),
    ],
    ids=["ms", "seconds", "http-date", "capped", "none"],
)
async def test_429_retry_after_variants(client, mock, headers, expected):
    route = _route(mock, httpx.Response(429, json=_ERR_429, headers={**headers, **RL_HEADERS}))
    r = await client.send(Req())
    assert route.call_count == 1  # 不重试
    assert r.outcome is Outcome.THROTTLED
    assert r.status == 429
    if expected is None:
        assert r.retry_after_s is None
    else:
        assert r.retry_after_s == pytest.approx(expected, abs=1e-6)
    assert r.remaining_tokens == 987654
    assert r.error_code == "429"


async def test_503_throttled_not_retried(client, mock):
    route = _route(mock, httpx.Response(503, json={"error": {"code": "ServiceUnavailable",
                                                             "message": "busy"}},
                                        headers={"retry-after": "3"}))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.THROTTLED
    assert r.status == 503
    assert r.retry_after_s == pytest.approx(3.0)


async def test_403_quota_is_throttled(client, mock):
    body = {"error": {"code": "403", "message": "Quota exceeded for this deployment"}}
    route = _route(mock, httpx.Response(403, json=body))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.THROTTLED
    assert r.status == 403


async def test_403_non_quota_is_client_error(client, mock):
    body = {"error": {"code": "PermissionDenied", "message": "Access denied due to invalid key"}}
    route = _route(mock, httpx.Response(403, json=body))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.CLIENT_ERROR
    assert r.error_code == "PermissionDenied"


# ---------------------------------------------------------------------------
# 400
# ---------------------------------------------------------------------------


async def test_400_content_filter_with_categories(client, mock):
    body = {"error": {
        "code": "content_filter",
        "message": "The response was filtered",
        "param": "prompt",
        "status": 400,
        "innererror": {
            "code": "ResponsibleAIPolicyViolation",
            "content_filter_result": {
                "hate": {"filtered": True, "severity": "high"},
                "self_harm": {"filtered": False, "severity": "safe"},
                "sexual": {"filtered": False, "severity": "safe"},
                "violence": {"filtered": True, "severity": "medium"},
            },
        },
    }}
    route = _route(mock, httpx.Response(400, json=body))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.CONTENT_FILTER
    assert r.status == 400
    assert r.error_code == "content_filter"
    assert r.filter_detail == "hate,violence"


async def test_400_other_is_client_error(client, mock):
    body = {"error": {"code": "context_length_exceeded", "message": "too long"}}
    route = _route(mock, httpx.Response(400, json=body))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.CLIENT_ERROR
    assert r.error_code == "context_length_exceeded"
    assert r.filter_detail is None


# ---------------------------------------------------------------------------
# 5xx / 网络
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 502, 504])
async def test_5xx_server_error(client, mock, status):
    route = _route(mock, httpx.Response(status, json={"error": {"code": "InternalServerError",
                                                                "message": "boom"}}))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.SERVER_ERROR
    assert r.status == status


async def test_5xx_non_json_body(client, mock):
    route = _route(mock, httpx.Response(502, text="<html>Bad Gateway</html>"))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.SERVER_ERROR
    assert r.status == 502


async def test_timeout(client, mock):
    route = _route(mock, side_effect=httpx.ReadTimeout("read timed out"))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.TIMEOUT
    assert r.status is None
    assert r.prompt_tokens is None


async def test_connection_error(client, mock):
    route = _route(mock, side_effect=httpx.ConnectError("connection refused"))
    r = await client.send(Req())
    assert route.call_count == 1
    assert r.outcome is Outcome.SERVER_ERROR
    assert r.status is None


# ---------------------------------------------------------------------------
# 构造 / 日志安全
# ---------------------------------------------------------------------------


async def test_latency_uses_injected_clock(mock):
    clock = FakeClock()

    def respond(request):
        clock.advance(0.25)
        return httpx.Response(200, json=_completion())

    route = _route(mock, side_effect=respond)
    c = AzureClient(ENDPOINT, API_KEY, API_VERSION, DEPLOYMENT, clock=clock)
    try:
        r = await c.send(Req())
    finally:
        await c.aclose()
    assert route.call_count == 1
    assert r.latency_ms == pytest.approx(250.0)


async def test_sdk_retries_disabled(client):
    assert client._client.max_retries == 0


async def test_from_settings():
    from types import SimpleNamespace

    from pydantic import SecretStr

    azure = SimpleNamespace(endpoint=ENDPOINT + "/", api_key=SecretStr(API_KEY),
                            api_version=API_VERSION, deployment=DEPLOYMENT)
    load = SimpleNamespace(max_concurrency=8)
    c = AzureClient.from_settings(azure, load)
    try:
        assert c.deployment == DEPLOYMENT
        assert c._client.max_retries == 0
    finally:
        await c.aclose()


async def test_never_logs_key_or_content(client, mock, caplog):
    caplog.set_level("DEBUG")
    for status in (429, 400, 500):
        _route(mock, httpx.Response(status, json={"error": {"code": "x", "message": "m"}}))
        await client.send(Req())
    ours = [r for r in caplog.records if r.name.startswith("loadtest")]
    text = "\n".join(r.getMessage() for r in ours)
    assert API_KEY not in text
    assert "SECRET-PASSAGE-CONTENT" not in text
