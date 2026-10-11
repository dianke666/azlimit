"""Property 26: Gemini 重试与退避（Validates: Requirements 11.2, 11.3, 11.4, 11.6, 11.7）。

∀ 错误序列：单次 generate 的尝试次数 ≤ max_retries + 1；其他 4xx 与 SAFETY 不重试；
∀ 退避：0 ≤ backoff ≤ 30，有 Retry-After 时 backoff ≥ min(Retry-After, 30)；
任一 429 后 Gemini 节流档位立即 == 0.10；HOLD 期间发起的尝试数为 0；
预留被拒时不发起调用并抛出 GeminiBudgetExhausted；花费按实际使用的模型结算。
"""

from __future__ import annotations

import asyncio
from random import Random

import pytest
from google.genai import errors as genai_errors
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.corpus.gemini_client import (
    BackendResult,
    GeminiBudgetExhausted,
    GeminiCallFailed,
    GeminiClient,
)
from loadtest.corpus.gemini_limiter import GeminiRateLimiter
from loadtest.corpus.model_chain import ModelChain
from loadtest.corpus.taxonomy import TopicSpec
from loadtest.cost import GEMINI_PROVIDERS, CostGuard
from loadtest.cost.pricing import get_price, token_cost, worst_case_token_cost
from loadtest.settings import CostSettings

PRICES = CostSettings().prices
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
GROUNDING_KEY = "google-search-grounding"
MAX_OUT = 2048
CAP = 30.0
TOL = 1e-9


def count(text: str) -> int:
    return max(1, len(text) // 4)


def make_spec(grounding: bool) -> TopicSpec:
    return TopicSpec(
        category="daily_news" if grounding else "technology",
        subtopic="sub",
        genre="科普",
        lang="en",
        grounding=grounding,
        length_bucket="M",
        target_len=1000,
    )


def api_error(code: int, status: str, retry_after: float | None = None) -> Exception:
    err: dict = {"code": code, "status": status, "message": "injected"}
    if retry_after is not None:
        err["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{retry_after}s"}]
    cls = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
    return cls(code, {"error": err})


class RecordingLimiter(GeminiRateLimiter):
    """记录每次 on_throttle 后的档位。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.levels_after_throttle: list[float] = []

    def on_throttle(self, retry_after, now=None) -> None:
        super().on_throttle(retry_after, now)
        self.levels_after_throttle.append(self.throttle.level)


class ScriptedBackend:
    """按脚本逐次返回结果或抛出错误；记录每次调用时的模型与 HOLD 状态。"""

    def __init__(self, steps: list[tuple], limiter: GeminiRateLimiter) -> None:
        self.steps = list(steps)
        self.limiter = limiter
        self.calls: list[dict] = []

    async def generate(self, model, prompt, grounding, timeout_s, **kw):
        step = self.steps.pop(0) if self.steps else ("ok", 100, 900)
        self.calls.append(
            {"model": model, "step": step, "holding": self.limiter.is_holding(), "grounding": grounding}
        )
        kind = step[0]
        if kind == "ok":
            return BackendResult("text " * 50, "STOP", step[1], step[2], ("https://example.com/a",))
        if kind == "safety":
            return BackendResult("", "SAFETY", step[1], 0, ())
        if kind == "429":
            raise api_error(429, "RESOURCE_EXHAUSTED", step[1])
        if kind == "5xx":
            raise api_error(503, "UNAVAILABLE")
        if kind == "timeout":
            raise asyncio.TimeoutError()
        if kind == "4xx":
            raise api_error(400, "INVALID_ARGUMENT")
        raise AssertionError(step)


step_st = st.one_of(
    st.tuples(st.just("ok"), st.integers(1, 2000), st.integers(1, MAX_OUT)),
    st.tuples(st.just("safety"), st.integers(1, 2000)),
    st.tuples(st.just("429"), st.one_of(st.none(), st.floats(0, 200, allow_nan=False))),
    st.tuples(st.just("5xx")),
    st.tuples(st.just("timeout")),
    st.tuples(st.just("4xx")),
)


def expected_attempts(steps: list[tuple], max_retries: int) -> tuple[int, str]:
    """→ (尝试次数, 结局 ok / safety / client / exhausted)。"""
    for i, s in enumerate(steps[: max_retries + 1]):
        if s[0] in ("ok", "safety"):
            return i + 1, s[0]
        if s[0] == "4xx":
            return i + 1, "client"
    n = min(len(steps), max_retries + 1)
    if n < max_retries + 1:  # 脚本用尽后默认 ok
        return n + 1, "ok"
    return n, "exhausted"


@settings(max_examples=150, deadline=None)
@given(
    tasks=st.lists(
        st.tuples(st.lists(step_st, min_size=0, max_size=6), st.booleans()), min_size=1, max_size=5
    ),
    max_retries=st.integers(0, 3),
    seed=st.integers(0, 2**32 - 1),
)
def test_gemini_retry_backoff_throttle_and_cost(tasks, max_retries, seed) -> None:
    asyncio.run(_run_retry_case(tasks, max_retries, seed))


async def _run_retry_case(tasks, max_retries, seed) -> None:
    clock = FakeClock()
    limiter = RecordingLimiter(clock=clock)
    guard = CostGuard(1e6, 0.0, PRICES, GEMINI_PROVIDERS)
    chain = ModelChain(MODELS, failover_after=1, failback_after_s=1e9)
    backend = ScriptedBackend([], limiter)
    client = GeminiClient(
        backend, limiter, chain, guard, clock=clock, rng=Random(seed),
        max_retries=max_retries, backoff_base_s=2.0, backoff_cap_s=CAP, max_output_tokens=MAX_OUT,
        token_count=count,
    )
    expected_cost: dict[tuple[str, str], float] = {}

    def add(key, v):
        expected_cost[key] = expected_cost.get(key, 0.0) + v

    prompt = "write something " * 20
    in_est = count(prompt)
    for steps, grounding in tasks:
        spec = make_spec(grounding)
        backend.steps = list(steps)
        start = len(backend.calls)
        n_throttles_before = len(limiter.levels_after_throttle)
        exp_n, exp_outcome = expected_attempts(steps, max_retries)
        try:
            out = await client.generate(spec, prompt)
            outcome = "safety" if out.finish_reason == "SAFETY" else "ok"
            assert out.attempts == exp_n
            assert out.model == backend.calls[-1]["model"]
        except GeminiCallFailed as e:
            outcome = "client" if e.kind == "client" else "exhausted"
            assert e.attempts == exp_n
        calls = backend.calls[start:]
        # 尝试次数 ≤ max_retries + 1；4xx / SAFETY 不重试
        assert len(calls) == exp_n <= max_retries + 1
        assert outcome == exp_outcome
        # HOLD 期间不发起任何尝试；grounding 标志按 spec 传递
        assert all(not c["holding"] for c in calls)
        assert all(c["grounding"] == grounding for c in calls)
        # 每个 429 后档位立即 == 0.10
        n_429 = sum(1 for c in calls if c["step"][0] == "429")
        new_levels = limiter.levels_after_throttle[n_throttles_before:]
        assert len(new_levels) == n_429
        assert all(lv == pytest.approx(0.10) for lv in new_levels)
        # 退避范围
        records = client.last_attempts
        for i, rec in enumerate(records):
            if rec.outcome == "ok" or rec.outcome == "client":
                assert rec.backoff_s is None
                continue
            if i == len(records) - 1:
                assert rec.backoff_s is None  # 最后一次失败后不再等待
                continue
            assert rec.backoff_s is not None and 0.0 <= rec.backoff_s <= CAP
            assert rec.backoff_s <= min(CAP, 2.0 * 2**i) or (
                rec.retry_after is not None and rec.backoff_s <= min(CAP, rec.retry_after) + TOL
            )
            if rec.retry_after is not None:
                assert rec.backoff_s >= min(rec.retry_after, CAP) - TOL
        # 期望花费：按实际模型
        for c in calls:
            m, kind = c["model"], c["step"][0]
            price = get_price(PRICES, m)
            if kind == "ok":
                add(("gemini", m), token_cost(price, c["step"][1], c["step"][2]))
            elif kind == "safety":
                add(("gemini", m), token_cost(price, c["step"][1], 0))
            elif kind in ("5xx", "timeout"):
                add(("gemini", m), worst_case_token_cost(price, in_est, MAX_OUT))
            per_q = get_price(PRICES, GROUNDING_KEY).per_unit
            if grounding and kind in ("ok", "safety"):
                add(("gemini_grounding", GROUNDING_KEY), per_q)  # 假后端无查询列表 → 按 1 次
            elif grounding and kind in ("5xx", "timeout"):
                add(("gemini_grounding", GROUNDING_KEY), per_q * client.grounding_reserve_queries)  # 按预留

    assert guard.outstanding == 0
    assert guard.reserved == 0.0
    for key in set(expected_cost) | set(guard.spent_by_model):
        assert guard.spent_by_model.get(key, 0.0) == pytest.approx(expected_cost.get(key, 0.0), abs=1e-9)


@settings(max_examples=50, deadline=None)
@given(grounding=st.booleans(), steps=st.lists(step_st, max_size=3))
def test_reservation_refused_makes_no_call(grounding, steps) -> None:
    async def case() -> None:
        clock = FakeClock()
        limiter = GeminiRateLimiter(clock=clock)
        guard = CostGuard(1e-9, 0.0, PRICES, GEMINI_PROVIDERS)  # 任意预留都会被拒
        chain = ModelChain(MODELS)
        backend = ScriptedBackend(steps, limiter)
        client = GeminiClient(backend, limiter, chain, guard, clock=clock, rng=Random(0), token_count=count)
        with pytest.raises(GeminiBudgetExhausted):
            await client.generate(make_spec(grounding), "prompt")
        assert backend.calls == []
        assert guard.spent == 0.0 and guard.outstanding == 0
        assert chain.failures[MODELS[0]] == 0

    asyncio.run(case())
