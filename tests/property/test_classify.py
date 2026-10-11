"""Feature: azure-gpt-load-tester, Property 19: 错误分类完备
（Validates: Requirements 22.1, 22.2, 22.5, 19.5, 24.4）。"""

from __future__ import annotations

import asyncio

import httpx
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.load.azure_client import classify, is_quota_error
from loadtest.outcome import Outcome

statuses = st.one_of(st.none(), st.integers(min_value=100, max_value=599), st.integers(-5, 1000))
json_leaf = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=20))
bodies = st.one_of(
    st.none(),
    st.text(max_size=50),
    st.binary(max_size=50),
    st.integers(),
    st.lists(json_leaf, max_size=3),
    st.dictionaries(st.sampled_from(["code", "message", "type", "x"]), json_leaf, max_size=4),
    st.fixed_dictionaries(
        {"error": st.dictionaries(st.sampled_from(["code", "message", "type"]), json_leaf, max_size=3)}
    ),
    st.just({"error": {"code": "content_filter", "message": "filtered"}}),
    st.just({"error": {"code": "429", "message": "Rate limit is exceeded"}}),
)
_req = httpx.Request("POST", "https://example.invalid")
timeouts = st.sampled_from(
    [TimeoutError(), asyncio.TimeoutError(), httpx.ReadTimeout("t", request=_req), httpx.ConnectTimeout("t")]
)
other_exc = st.sampled_from(
    [ValueError("x"), httpx.ConnectError("c"), RuntimeError("r"), KeyError("k"), httpx.RemoteProtocolError("p")]
)
excs = st.one_of(st.none(), timeouts, other_exc)
finish = st.one_of(st.none(), st.sampled_from(["stop", "length", "content_filter", "tool_calls"]), st.text(max_size=10))


@settings(deadline=None, max_examples=1000)
@given(status=statuses, body=bodies, exc=excs, fr=finish)
def test_classify_total_and_rules(status, body, exc, fr) -> None:
    out = classify(status, body, exc, fr)
    assert isinstance(out, Outcome)
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        assert out is Outcome.TIMEOUT
        return
    if status is None:
        assert out is Outcome.SERVER_ERROR
        return
    if status in (429, 503):
        assert out is Outcome.THROTTLED
    elif status == 403:
        assert out is (Outcome.THROTTLED if is_quota_error(body) else Outcome.CLIENT_ERROR)
    elif status == 400 and isinstance(body, dict) and isinstance(body.get("error"), dict) and body["error"].get("code") == "content_filter":
        assert out is Outcome.CONTENT_FILTER
    elif 200 <= status < 300 and exc is None:
        assert out is (Outcome.CONTENT_FILTER if fr == "content_filter" else Outcome.OK)
    # 错误路径永不为 OK
    if exc is not None or not (200 <= status < 300):
        assert out is not Outcome.OK


@settings(deadline=None, max_examples=200)
@given(exc=timeouts, status=statuses, body=bodies, fr=finish)
def test_timeout_always_timeout(exc, status, body, fr) -> None:
    assert classify(status, body, exc, fr) is Outcome.TIMEOUT


@settings(deadline=None, max_examples=200)
@given(exc=st.one_of(st.none(), other_exc), body=bodies, fr=finish)
def test_no_status_server_error(exc, body, fr) -> None:
    assert classify(None, body, exc, fr) is Outcome.SERVER_ERROR


def test_examples() -> None:
    quota = {"error": {"code": "QuotaExceeded", "message": "Quota exceeded for this resource"}}
    assert classify(403, quota, None) is Outcome.THROTTLED
    assert classify(403, "Rate limit reached", None) is Outcome.THROTTLED
    assert classify(403, {"error": {"code": "PermissionDenied", "message": "no access"}}, None) is Outcome.CLIENT_ERROR
    assert classify(403, None, None) is Outcome.CLIENT_ERROR
    assert classify(400, {"error": {"code": "content_filter"}}, None) is Outcome.CONTENT_FILTER
    assert classify(400, {"error": {"code": "invalid_request"}}, None) is Outcome.CLIENT_ERROR
    assert classify(200, {}, None, "content_filter") is Outcome.CONTENT_FILTER
    assert classify(200, {}, None, "stop") is Outcome.OK
    assert classify(200, {}, ValueError("bad json")) is Outcome.SERVER_ERROR
    assert classify(429, None, None) is Outcome.THROTTLED
    assert classify(503, None, None) is Outcome.THROTTLED
    for s in (500, 502, 504, 501, 302):
        assert classify(s, None, None) is Outcome.SERVER_ERROR
    assert classify(None, None, httpx.ConnectError("x")) is Outcome.SERVER_ERROR
