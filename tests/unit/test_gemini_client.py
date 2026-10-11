"""GeminiClient 单元测试（Requirements 3.1, 3.4, 3.6, 5.1, 11.2–11.7, 12.6）。不发起任何真实网络调用。"""

from __future__ import annotations

import asyncio
from random import Random
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from loadtest.clock import FakeClock
from loadtest.corpus.gemini_client import (
    BackendResult,
    GeminiBudgetExhausted,
    GeminiCallFailed,
    GeminiClient,
    GeneratedText,
    VertexBackend,
    classify_gemini_error,
    normalize_response,
)
from loadtest.corpus.gemini_limiter import GeminiRateLimiter
from loadtest.corpus.model_chain import ModelChain
from loadtest.corpus.taxonomy import TopicSpec
from loadtest.cost import GEMINI_PROVIDERS, CostGuard
from loadtest.settings import CostSettings

PRICES = CostSettings().prices
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]


def count(text: str) -> int:
    return max(1, len(text) // 4)


def spec(grounding: bool = False, lang: str = "en") -> TopicSpec:
    return TopicSpec(
        category="daily_news" if grounding else "technology",
        subtopic="sub",
        genre="科普",
        lang=lang,  # type: ignore[arg-type]
        grounding=grounding,
        length_bucket="M",
        target_len=1000,
    )


class FakeBackend:
    def __init__(self, results: list) -> None:
        self.results = list(results)
        self.calls: list[dict] = []

    async def generate(self, model, prompt, grounding, timeout_s, **kw):
        self.calls.append({"model": model, "prompt": prompt, "grounding": grounding, "timeout_s": timeout_s, **kw})
        r = self.results.pop(0) if self.results else BackendResult("ok text", "STOP", 10, 20)
        if isinstance(r, BaseException):
            raise r
        return r


def make_client(results, *, cap: float = 1e6, max_retries: int = 3):
    clock = FakeClock()
    limiter = GeminiRateLimiter(clock=clock)
    guard = CostGuard(cap, 0.0, PRICES, GEMINI_PROVIDERS)
    chain = ModelChain(MODELS)
    backend = FakeBackend(results)
    client = GeminiClient(
        backend, limiter, chain, guard, clock=clock, rng=Random(1), max_retries=max_retries, token_count=count
    )
    return client, backend, guard, chain, limiter, clock


def google_response(
    text: str = "Para one.\n\nPara two.",
    finish=genai_types.FinishReason.STOP,
    uris=("https://a.example/1", "https://b.example/2"),
    prompt_tokens: int = 123,
    cand_tokens: int = 456,
):
    """形如 google-genai GenerateContentResponse 的对象（使用 SDK 的真实类型）。"""
    chunks = [genai_types.GroundingChunk(web=genai_types.GroundingChunkWeb(uri=u, title="t")) for u in uris]
    cand = genai_types.Candidate(
        content=genai_types.Content(role="model", parts=[genai_types.Part(text=text)]),
        finish_reason=finish,
        grounding_metadata=genai_types.GroundingMetadata(grounding_chunks=chunks) if uris else None,
    )
    return genai_types.GenerateContentResponse(
        candidates=[cand],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=prompt_tokens, candidates_token_count=cand_tokens
        ),
    )


# --------------------------------------------------------------------------- normalize_response


def test_normalize_response_real_sdk_types() -> None:
    text, finish, i, o, sources = normalize_response(google_response())
    assert text == "Para one.\n\nPara two."
    assert finish == "STOP"
    assert (i, o) == (123, 456)
    assert sources == ["https://a.example/1", "https://b.example/2"]


def test_normalize_response_duck_typed_and_dedup_sources() -> None:
    web = lambda u: SimpleNamespace(web=SimpleNamespace(uri=u))  # noqa: E731
    resp = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text="thinking", thought=True), SimpleNamespace(text="body")]),
                finish_reason="FinishReason.MAX_TOKENS",
                grounding_metadata=SimpleNamespace(grounding_chunks=[web("u1"), web("u1"), web(None), web("u2")]),
            )
        ],
        usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=7, thoughts_token_count=3),
    )
    text, finish, i, o, sources = normalize_response(resp)
    assert text == "body"
    assert finish == "MAX_TOKENS"
    assert (i, o) == (5, 10)
    assert sources == ["u1", "u2"]


def test_normalize_response_safety_and_prompt_block() -> None:
    resp = google_response(text="", finish=genai_types.FinishReason.SAFETY, uris=())
    text, finish, *_ = normalize_response(resp)
    assert finish == "SAFETY" and text == ""
    blocked = SimpleNamespace(candidates=None, prompt_feedback=SimpleNamespace(block_reason="SAFETY"), usage_metadata=None)
    text, finish, i, o, sources = normalize_response(blocked)
    assert (text, finish, i, o, sources) == ("", "SAFETY", 0, 0, [])


# --------------------------------------------------------------------------- classify_gemini_error


def test_classify_real_google_errors() -> None:
    e429 = genai_errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": "quota",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}],
            }
        },
    )
    assert isinstance(e429, genai_errors.APIError) and e429.code == 429
    assert classify_gemini_error(e429) == ("throttled", 7.0)
    re_only = genai_errors.APIError(0, {"error": {"code": 0, "status": "RESOURCE_EXHAUSTED", "message": "x"}})
    assert classify_gemini_error(re_only)[0] == "throttled"
    assert classify_gemini_error(genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}))[0] == "server"
    assert classify_gemini_error(genai_errors.ServerError(500, {"error": {"code": 500, "status": "INTERNAL"}}))[0] == "server"
    assert classify_gemini_error(genai_errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})) == ("client", None)
    assert classify_gemini_error(genai_errors.ClientError(403, {"error": {"code": 403, "status": "PERMISSION_DENIED"}}))[0] == "client"
    assert classify_gemini_error(asyncio.TimeoutError()) == ("timeout", None)
    assert classify_gemini_error(TimeoutError()) == ("timeout", None)
    assert classify_gemini_error(httpx.ReadTimeout("t")) == ("timeout", None)
    assert classify_gemini_error(httpx.ConnectError("c")) == ("server", None)
    assert classify_gemini_error(ValueError("bug")) == ("client", None)


def test_classify_retry_after_header() -> None:
    resp = httpx.Response(429, headers={"Retry-After": "12"}, request=httpx.Request("POST", "https://x.invalid"))
    e = genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}}, resp)
    assert classify_gemini_error(e) == ("throttled", 12.0)


# --------------------------------------------------------------------------- GeminiClient.generate


async def test_generated_text_fields_and_settled_per_model() -> None:
    client, backend, guard, chain, _, _ = make_client(
        [BackendResult("hello world", "STOP", 111, 222, ("https://s/1",))]
    )
    out = await client.generate(spec(grounding=True), "the prompt")
    assert isinstance(out, GeneratedText)
    assert out == GeneratedText(
        text="hello world", model=MODELS[0], input_tokens=111, output_tokens=222, finish_reason="STOP",
        grounded=True, sources=["https://s/1"], attempts=1, grounding_calls=1,
    )
    assert guard.outstanding == 0
    assert set(guard.spent_by_model) == {("gemini", MODELS[0]), ("gemini_grounding", "google-search-grounding")}
    entries = {(e.provider, e.model): e for e in guard.pending_entries("r")}
    assert entries[("gemini", MODELS[0])].input_tokens == 111
    assert entries[("gemini", MODELS[0])].output_tokens == 222
    assert entries[("gemini_grounding", "google-search-grounding")].units == 1


async def test_grounding_billed_per_unique_search_query() -> None:
    """Gemini 3：按响应中去重后的搜索查询数计费；预留 3 次，结算按实际。"""
    client, _, guard, *_ = make_client(
        [BackendResult("body", "STOP", 100, 200, ("https://s/1",), search_queries=("q1", "q2", "q1", " "))]
    )
    out = await client.generate(spec(grounding=True), "p")
    assert out.grounding_calls == 2
    gr = {(e.provider, e.model): e for e in guard.pending_entries("r")}[("gemini_grounding", "google-search-grounding")]
    assert gr.units == 2 and gr.cost_usd == pytest.approx(2 * 0.014)
    assert guard.reserved == 0.0


async def test_grounding_timeout_charges_reserved_queries() -> None:
    client, _, guard, *_ = make_client([asyncio.TimeoutError(), BackendResult("body", "STOP", 1, 1, ("u",))])
    out = await client.generate(spec(grounding=True), "p")
    # 超时一次按预留 3 次计 + 成功一次（无查询列表）按 1 次计
    assert out.grounding_calls == 3 + 1
    gr = {(e.provider, e.model): e for e in guard.pending_entries("r")}[("gemini_grounding", "google-search-grounding")]
    assert gr.units == 4


def test_grounding_query_count_from_sdk_response() -> None:
    from loadtest.corpus.gemini_client import grounding_query_count

    resp = google_response()
    assert grounding_query_count(resp) is None  # 无 web_search_queries
    resp.candidates[0].grounding_metadata.web_search_queries = ["a b", "c", "a b"]
    assert grounding_query_count(resp) == 2
    assert grounding_query_count(BackendResult("x")) is None


async def test_grounding_flag_only_for_grounding_specs() -> None:
    client, backend, *_ = make_client([])
    await client.generate(spec(grounding=False), "p1")
    await client.generate(spec(grounding=True), "p2")
    assert [c["grounding"] for c in backend.calls] == [False, True]
    assert all(c["max_output_tokens"] == 2048 and c["timeout_s"] == 60.0 for c in backend.calls)


def test_vertex_backend_config_tools_only_when_grounding() -> None:
    from google.genai import types

    plain = VertexBackend.build_config("gemini-3.5-flash-lite", False, 2048, 0.7, "minimal")
    assert not plain.tools
    assert plain.max_output_tokens == 2048 and plain.temperature == 0.7
    assert plain.thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    assert plain.thinking_config.thinking_budget is None
    g = VertexBackend.build_config("gemini-3.1-flash-lite", True, 1024, 0.5)
    assert len(g.tools) == 1 and g.tools[0].google_search is not None
    # gemini-3* 未指定时默认 low
    assert g.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_vertex_backend_disables_automatic_function_calling() -> None:
    for grounding in (False, True):
        cfg = VertexBackend.build_config("gemini-3.1-flash-lite", grounding, 2048, 1.0, "minimal")
        assert cfg.automatic_function_calling.disable is True


def test_vertex_backend_thinking_by_model_family() -> None:
    v25 = VertexBackend.build_config("gemini-2.5-flash-lite", False, 2048, 1.0)
    assert v25.thinking_config.thinking_budget == 0 and v25.thinking_config.thinking_level is None
    v20 = VertexBackend.build_config("gemini-2.0-flash-lite", False, 2048, 1.0)
    assert v20.thinking_config is None
    v38 = VertexBackend.build_config("gemini-3.8-flash", False, 2048, 1.0, "low")
    assert v38.thinking_config.thinking_level.value == "LOW"


async def test_vertex_backend_applies_per_model_thinking_level() -> None:
    seen = {}

    class Models:
        async def generate_content(self, *, model, contents, config):
            seen[model] = config
            return google_response()

    fake = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    be = VertexBackend(
        "proj", "global", "does-not-exist.json", client=fake,
        thinking_levels={"gemini-3.1-flash-lite": "minimal", "gemini-3.8-flash": "low"},
    )
    await be.generate("gemini-3.1-flash-lite", "hi", False, 60.0)
    await be.generate("gemini-3.8-flash", "hi", False, 60.0)
    assert seen["gemini-3.1-flash-lite"].thinking_config.thinking_level.value == "MINIMAL"
    assert seen["gemini-3.8-flash"].thinking_config.thinking_level.value == "LOW"
    assert seen["gemini-3.1-flash-lite"].temperature == 1.0


async def test_vertex_backend_uses_injected_client_without_network() -> None:
    seen = {}

    class Models:
        async def generate_content(self, *, model, contents, config):
            seen.update(model=model, contents=contents, config=config)
            return google_response()

    fake = SimpleNamespace(aio=SimpleNamespace(models=Models()))
    be = VertexBackend("proj", "global", "does-not-exist.json", client=fake)
    resp = await be.generate("gemini-3.1-flash-lite", "hi", True, 60.0, max_output_tokens=100, temperature=0.2)
    assert seen["model"] == "gemini-3.1-flash-lite" and seen["contents"] == "hi"
    assert seen["config"].tools and seen["config"].max_output_tokens == 100
    assert normalize_response(resp)[1] == "STOP"


async def test_safety_not_retried() -> None:
    client, backend, guard, chain, *_ = make_client([BackendResult("", "SAFETY", 50, 0)])
    out = await client.generate(spec(), "p")
    assert out.finish_reason == "SAFETY" and out.attempts == 1
    assert len(backend.calls) == 1
    assert chain.failures[MODELS[0]] == 0


async def test_client_4xx_not_retried_and_not_billed() -> None:
    err = genai_errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    client, backend, guard, chain, *_ = make_client([err, BackendResult("x", "STOP", 1, 1)])
    with pytest.raises(GeminiCallFailed) as ei:
        await client.generate(spec(grounding=True), "p")
    assert ei.value.kind == "client" and ei.value.attempts == 1
    assert len(backend.calls) == 1
    assert guard.spent == 0.0 and guard.outstanding == 0
    assert chain.failures[MODELS[0]] == 0  # 4xx 不计入模型链失败


async def test_retries_exhausted_reports_one_failure() -> None:
    errs = [genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}}) for _ in range(4)]
    client, backend, guard, chain, *_ = make_client(errs)
    with pytest.raises(GeminiCallFailed) as ei:
        await client.generate(spec(), "p")
    assert ei.value.attempts == 4 and ei.value.kind == "server"
    assert len(backend.calls) == 4
    assert chain.failures[MODELS[0]] == 1
    assert guard.spent > 0  # 5xx 按预留金额结算
    assert guard.outstanding == 0


async def test_429_then_success_throttles_and_holds() -> None:
    e429 = genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
    client, backend, guard, chain, limiter, clock = make_client([e429, BackendResult("ok", "STOP", 5, 5)])
    out = await client.generate(spec(), "p")
    assert out.attempts == 2
    assert limiter.throttle.level == pytest.approx(0.10)
    # 第二次尝试在 HOLD（≥ 10s）结束后才发起
    assert clock.now() >= 10.0
    assert client.last_attempts[0].outcome == "throttled"
    assert 0 <= client.last_attempts[0].backoff_s <= 30


async def test_timeout_via_wait_for_is_retried() -> None:
    class SlowBackend(FakeBackend):
        async def generate(self, model, prompt, grounding, timeout_s, **kw):
            self.calls.append({"model": model})
            if len(self.calls) == 1:
                await asyncio.sleep(10)
            return BackendResult("ok", "STOP", 1, 1)

    clock = FakeClock()
    limiter = GeminiRateLimiter(clock=clock)
    guard = CostGuard(1e6, 0.0, PRICES, GEMINI_PROVIDERS)
    backend = SlowBackend([])
    client = GeminiClient(backend, limiter, ModelChain(MODELS), guard, clock=clock, rng=Random(0),
                          timeout_s=0.01, token_count=count)
    out = await client.generate(spec(), "p")
    assert out.attempts == 2
    assert client.last_attempts[0].outcome == "timeout"


async def test_reservation_refused_no_call() -> None:
    client, backend, guard, *_ = make_client([], cap=1e-9)
    with pytest.raises(GeminiBudgetExhausted):
        await client.generate(spec(), "p")
    assert backend.calls == []
    assert client.calls == 0


def test_backoff_bounds() -> None:
    client, *_ = make_client([])
    for n in range(8):
        for ra in (None, 0.0, 5.0, 29.0, 31.0, 500.0):
            b = client.backoff(n, ra)
            assert 0.0 <= b <= 30.0
            if ra is not None:
                assert b >= min(ra, 30.0)
            else:
                assert b <= min(30.0, 2.0 * 2**n)
