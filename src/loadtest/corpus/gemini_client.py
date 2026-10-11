"""GeminiClient：Vertex AI 调用 + 限流 / 节流 / 预留结算 / 指数退避重试 / 模型链（组件 4）。

单次 ``generate(spec, prompt)`` 的每次尝试（Requirements 11.2–11.7, 12.6）::

    model = chain.current(now)
    await limiter.wait_if_holding()                 # HOLD 期间不发起任何尝试（11.7）
    await limiter.acquire(est_tokens)               # 双桶 × 节流档位
    r = await cost_guard.reserve(model, in, max_out, units=int(spec.grounding))
    r is None → GeminiBudgetExhausted（不发起调用，11.6）
    backend.generate(...)（asyncio.wait_for 超时）
      成功（含 SAFETY）   → settle(r, 实际 usage, model) ；chain.report_success；返回 GeneratedText
      429 / RESOURCE_EXHAUSTED → settle_zero；limiter.on_throttle(retry_after)（立即 10% + HOLD）；退避后重试
      5xx / 超时          → settle(r, None)（按预留金额结算，11.5）；退避后重试
      其他 4xx            → settle_zero；不重试，GeminiCallFailed
    max_retries + 1 次尝试均失败 → chain.report_failure(model)；GeminiCallFailed

退避（11.4）：第 n 次（从 0 起）= min(cap, max(Retry-After 或 0, Uniform(0, min(cap, base × 2^n))))。

``backend`` 是可替换的抽象（测试使用可编程的假实现）；生产实现为 :class:`VertexBackend`
（google-genai SDK，Vertex AI 模式，服务账号 JSON 凭据，Requirement 3.1）。
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Literal, Protocol

from loadtest.clock import Clock
from loadtest.corpus.taxonomy import TopicSpec
from loadtest.cost.guard import Usage

log = logging.getLogger(__name__)

ErrorKind = Literal["throttled", "server", "timeout", "client"]
VERTEX_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


# --------------------------------------------------------------------------- 结果与异常


@dataclass
class GeneratedText:
    text: str
    model: str  # 实际使用的模型（分模型计费 / gen_model）
    input_tokens: int
    output_tokens: int
    finish_reason: str  # 大写名称：STOP / SAFETY / MAX_TOKENS ...
    grounded: bool
    sources: list[str]  # grounding 元数据中的来源 URL
    attempts: int
    grounding_calls: int = 0  # 本任务中已计费的 grounding 尝试次数（含失败的重试）


class GeminiBudgetExhausted(RuntimeError):
    """Gemini CostGuard 拒绝预留：未发起调用。"""

    def __init__(self, msg: str = "gemini budget exhausted", *, attempts: int = 0, grounding_calls: int = 0):
        super().__init__(msg)
        self.attempts = attempts
        self.grounding_calls = grounding_calls


class GeminiCallFailed(RuntimeError):
    """重试耗尽或不可重试的错误（其他 4xx）。"""

    def __init__(
        self,
        msg: str,
        *,
        kind: ErrorKind,
        model: str,
        attempts: int,
        grounding_calls: int = 0,
        cause: BaseException | None = None,
    ):
        super().__init__(msg)
        self.kind = kind
        self.model = model
        self.attempts = attempts
        self.grounding_calls = grounding_calls
        self.cause = cause


# --------------------------------------------------------------------------- 后端抽象


@dataclass(frozen=True)
class BackendResult:
    """已归一化的后端结果（假后端可直接返回）。"""

    text: str
    finish_reason: str = "STOP"
    input_tokens: int = 0
    output_tokens: int = 0
    sources: tuple[str, ...] = ()
    search_queries: tuple[str, ...] | None = None  # None = 未知（grounding 时按 1 次计费）


def grounding_query_count(resp: Any) -> int | None:
    """响应中实际执行的去重搜索查询数（Gemini 3 grounding 按此计费）；无法得知时返回 None。"""
    if isinstance(resp, BackendResult):
        queries: Any = resp.search_queries
    else:
        candidates = list(getattr(resp, "candidates", None) or [])
        gm = getattr(candidates[0], "grounding_metadata", None) if candidates else None
        queries = getattr(gm, "web_search_queries", None) if gm is not None else None
    if not isinstance(queries, (list, tuple)):
        return None
    return len({q.strip() for q in queries if isinstance(q, str) and q.strip()})


class GeminiBackend(Protocol):
    async def generate(
        self,
        model: str,
        prompt: str,
        grounding: bool,
        timeout_s: float,
        *,
        max_output_tokens: int,
        temperature: float,
    ) -> Any:
        """返回原始响应（google-genai ``GenerateContentResponse`` 或 :class:`BackendResult`），失败时抛出异常。"""
        ...


def _finish_name(fr: Any) -> str:
    if fr is None:
        return ""
    name = getattr(fr, "name", None)
    s = name if isinstance(name, str) else str(fr)
    return s.strip().rsplit(".", 1)[-1].upper()


def _int(v: Any) -> int:
    try:
        return int(v or 0)
    except (TypeError, ValueError):
        return 0


def normalize_response(resp: Any) -> tuple[str, str, int, int, list[str]]:
    """google-genai 响应 → (text, finish_reason, input_tokens, output_tokens, sources)。

    - text：首个候选中非 thought 文本 part 拼接（回退到 ``resp.text``）。
    - finish_reason：大写枚举名；无候选且 ``prompt_feedback.block_reason`` 非空 → ``SAFETY``。
    - output_tokens：``candidates_token_count`` + ``thoughts_token_count``（思考 token 按输出计费）。
    - sources：``grounding_metadata.grounding_chunks[*].web.uri``（去重保序）。
    """
    if isinstance(resp, BackendResult):
        return resp.text, _finish_name(resp.finish_reason), resp.input_tokens, resp.output_tokens, list(resp.sources)

    candidates = list(getattr(resp, "candidates", None) or [])
    cand = candidates[0] if candidates else None

    text = ""
    if cand is not None:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = [
            p.text
            for p in parts
            if isinstance(getattr(p, "text", None), str) and not getattr(p, "thought", False)
        ]
        text = "".join(chunks)
    if not text:
        try:
            t = getattr(resp, "text", None)
        except Exception:  # SDK 的 .text 在某些形态下可能抛出
            t = None
        text = t if isinstance(t, str) else ""

    if cand is not None:
        finish = _finish_name(getattr(cand, "finish_reason", None))
    else:
        fb = getattr(resp, "prompt_feedback", None)
        finish = "SAFETY" if getattr(fb, "block_reason", None) else ""

    um = getattr(resp, "usage_metadata", None)
    in_tok = _int(getattr(um, "prompt_token_count", 0))
    out_tok = _int(getattr(um, "candidates_token_count", 0)) + _int(getattr(um, "thoughts_token_count", 0))

    sources: list[str] = []
    gm = getattr(cand, "grounding_metadata", None) if cand is not None else None
    for ch in list(getattr(gm, "grounding_chunks", None) or []):
        web = getattr(ch, "web", None)
        uri = getattr(web, "uri", None)
        if isinstance(uri, str) and uri and uri not in sources:
            sources.append(uri)
    return text, finish, in_tok, out_tok, sources


_GEMINI_VER_RE = re.compile(r"^gemini-(\d+)(?:\.\d+)?-")


def _is_gemini3_plus(model: str) -> bool:
    m = _GEMINI_VER_RE.match(model)
    return bool(m) and int(m.group(1)) >= 3


class VertexBackend:
    """google-genai（Vertex AI 模式）后端。凭据文件只交给 google-auth 读取，不记录、不输出内容。"""

    def __init__(
        self,
        project: str,
        location: str,
        credentials_file: str | Path,
        *,
        client: Any = None,
        thinking_levels: Mapping[str, str] | None = None,
    ) -> None:
        self.project = project
        self.location = location
        self.credentials_file = Path(credentials_file)
        self._client = client
        self.thinking_levels: dict[str, str] = dict(thinking_levels or {})

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai
            from google.oauth2 import service_account

            creds = service_account.Credentials.from_service_account_file(
                str(self.credentials_file), scopes=list(VERTEX_SCOPES)
            )
            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location, credentials=creds
            )
        return self._client

    @staticmethod
    def build_config(
        model: str,
        grounding: bool,
        max_output_tokens: int,
        temperature: float,
        thinking_level: str | None = None,
    ) -> Any:
        """构造 GenerateContentConfig。

        思考配置（思考 token 按输出计费，且挤占 max_output_tokens）：
        - gemini-3* 及以后：``thinking_level``（显式传入优先，否则 "low"；Flash-Lite 建议 "minimal"）。
        - gemini-2.5*：``thinking_budget=0`` 关闭思考。
        - 其他（如 2.0）：不设置。
        """
        from google.genai import types

        kw: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            # 不使用函数调用：关闭 SDK 的自动函数调用（AFC），避免隐式多轮调用与日志噪音
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        }
        if grounding:
            kw["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        if thinking_level is not None or _is_gemini3_plus(model):
            level = (thinking_level or "low").upper()
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=types.ThinkingLevel(level))
        elif model.startswith("gemini-2.5"):
            kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        return types.GenerateContentConfig(**kw)

    async def generate(
        self,
        model: str,
        prompt: str,
        grounding: bool,
        timeout_s: float,
        *,
        max_output_tokens: int = 2048,
        temperature: float = 1.0,
    ) -> Any:
        client = self._get_client()
        cfg = self.build_config(
            model, grounding, max_output_tokens, temperature, self.thinking_levels.get(model)
        )
        return await client.aio.models.generate_content(model=model, contents=prompt, config=cfg)


# --------------------------------------------------------------------------- 错误分类

_DELAY_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")


def _parse_seconds(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v >= 0 else None
    if isinstance(v, str):
        m = _DELAY_RE.match(v)
        if m:
            return float(m.group(1))
    return None


def _retry_after_of(exc: BaseException) -> float | None:
    """Retry-After：异常属性 ``retry_after`` → HTTP 头 → google.rpc.RetryInfo.retryDelay。"""
    ra = _parse_seconds(getattr(exc, "retry_after", None))
    if ra is not None:
        return ra
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is not None:
        try:
            ra = _parse_seconds(headers.get("retry-after"))
        except Exception:
            ra = None
        if ra is not None:
            return ra
    details = getattr(exc, "details", None)
    err = details.get("error", details) if isinstance(details, Mapping) else None
    items = err.get("details") if isinstance(err, Mapping) else None
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes)):
        for d in items:
            if isinstance(d, Mapping) and "RetryInfo" in str(d.get("@type", "")):
                ra = _parse_seconds(d.get("retryDelay"))
                if ra is not None:
                    return ra
    return None


def classify_gemini_error(exc: BaseException) -> tuple[ErrorKind, float | None]:
    """→ (kind, retry_after)。

    429 / RESOURCE_EXHAUSTED → throttled；5xx → server；超时（含 408）→ timeout；其他 4xx → client。
    网络传输错误按 server（可重试）；无法识别的异常按 client（不重试）。
    """
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout", None
    try:
        import httpx

        if isinstance(exc, httpx.TimeoutException):
            return "timeout", None
        if isinstance(exc, httpx.TransportError):
            return "server", None
    except ImportError:  # pragma: no cover
        pass

    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return "throttled", _retry_after_of(exc)
    if isinstance(code, int):
        if code == 408 or status == "DEADLINE_EXCEEDED":
            return "timeout", None
        if 500 <= code <= 599:
            return "server", _retry_after_of(exc)
        if 400 <= code <= 499:
            return "client", None
    if isinstance(exc, (ConnectionError, OSError)):
        return "server", None
    return "client", None


# --------------------------------------------------------------------------- GeminiClient


def _default_count(text: str) -> int:
    from loadtest.tokens import count

    return count(text)


@dataclass
class AttemptRecord:
    """单次尝试记录（测试与诊断用）。"""

    model: str
    outcome: str  # ok / throttled / server / timeout / client
    backoff_s: float | None = None
    retry_after: float | None = None


class GeminiClient:
    def __init__(
        self,
        backend: GeminiBackend,
        limiter: Any,
        chain: Any,
        cost_guard: Any,
        *,
        clock: Clock,
        rng: Random,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_base_s: float = 2.0,
        backoff_cap_s: float = 30.0,
        max_output_tokens: int = 2048,
        temperature: float = 1.0,
        token_count: Callable[[str], int] | None = None,
        grounding_reserve_queries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if grounding_reserve_queries < 1:
            raise ValueError("grounding_reserve_queries must be >= 1")
        # grounding 每次尝试预留的搜索查询数；成功后按响应中实际去重查询数结算
        self.grounding_reserve_queries = int(grounding_reserve_queries)
        self.backend = backend
        self.limiter = limiter
        self.chain = chain
        self.cost_guard = cost_guard
        self.clock = clock
        self.rng = rng
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.backoff_base_s = float(backoff_base_s)
        self.backoff_cap_s = float(backoff_cap_s)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self._count = token_count or _default_count
        # 统计
        self.calls = 0  # 实际发起的后端调用次数
        self.grounding_calls = 0  # 已计费的 grounding 尝试
        self.last_attempts: list[AttemptRecord] = []

    @classmethod
    def from_settings(
        cls, gs: Any, backend: GeminiBackend, limiter: Any, chain: Any, cost_guard: Any, *, clock: Clock, rng: Random,
        **kw: Any,
    ) -> GeminiClient:
        return cls(
            backend, limiter, chain, cost_guard, clock=clock, rng=rng,
            timeout_s=gs.timeout_s, max_retries=gs.max_retries,
            backoff_base_s=gs.backoff_base_s, backoff_cap_s=gs.backoff_cap_s,
            **{
                "temperature": getattr(gs, "temperature", 1.0),
                "grounding_reserve_queries": getattr(gs, "grounding_reserve_queries", 3),
                **kw,
            },
        )

    # ---- 退避 ----
    def backoff(self, attempt: int, retry_after: float | None) -> float:
        """min(cap, max(retry_after 或 0, Uniform(0, min(cap, base × 2^attempt))))。"""
        cap = self.backoff_cap_s
        hi = min(cap, self.backoff_base_s * (2 ** min(attempt, 62)))
        jitter = self.rng.uniform(0.0, hi) if hi > 0 else 0.0
        return max(0.0, min(cap, max(retry_after or 0.0, jitter)))

    # ---- 主入口 ----
    async def generate(self, spec: TopicSpec, prompt: str) -> GeneratedText:
        grounding = bool(spec.grounding)
        units = self.grounding_reserve_queries if grounding else 0
        in_est = self._count(prompt)
        est_tokens = in_est + self.max_output_tokens
        records: list[AttemptRecord] = []
        self.last_attempts = records
        g_calls = 0
        model = self.chain.current(self.clock.now())
        last_kind: ErrorKind = "server"
        last_exc: BaseException | None = None

        for attempt in range(self.max_retries + 1):
            model = self.chain.current(self.clock.now())
            await self.limiter.wait_if_holding()
            await self.limiter.acquire(est_tokens)
            r = await self.cost_guard.reserve(model, in_est, self.max_output_tokens, units=units)
            if r is None:
                raise GeminiBudgetExhausted(attempts=attempt, grounding_calls=g_calls)
            self.calls += 1
            try:
                raw = await asyncio.wait_for(
                    self.backend.generate(
                        model,
                        prompt,
                        grounding,
                        self.timeout_s,
                        max_output_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                    ),
                    timeout=self.timeout_s,
                )
                text, finish, in_tok, out_tok, sources = normalize_response(raw)
                n_queries = grounding_query_count(raw) if grounding else 0
            except asyncio.CancelledError:
                await self.cost_guard.release(r)
                raise
            except Exception as exc:  # noqa: BLE001 — 按类别处理
                kind, retry_after = classify_gemini_error(exc)
                last_kind, last_exc = kind, exc
                rec = AttemptRecord(model, kind, retry_after=retry_after)
                records.append(rec)
                if kind == "throttled":
                    await self.cost_guard.settle_zero(r)  # 429 不计费
                    self.limiter.on_throttle(retry_after, self.clock.now())
                elif kind in ("server", "timeout"):
                    await self.cost_guard.settle(r, None, model=model)  # 按预留金额结算（11.5）
                    if grounding:  # 按预留的查询数保守计入
                        g_calls += units
                        self.grounding_calls += units
                else:  # 其他 4xx：不重试
                    await self.cost_guard.settle_zero(r)
                    raise GeminiCallFailed(
                        f"gemini client error on {model}: {type(exc).__name__}",
                        kind=kind, model=model, attempts=attempt + 1, grounding_calls=g_calls, cause=exc,
                    ) from exc
                log.info("gemini attempt %d on %s failed: %s", attempt + 1, model, kind)
                if attempt < self.max_retries:
                    delay = self.backoff(attempt, retry_after)
                    rec.backoff_s = delay
                    await self.clock.sleep(delay)
                continue

            # 成功（SAFETY 也算一次成功调用：由调用方以 safety_block 拒绝，不重试）
            if in_tok <= 0:
                in_tok = in_est
            if out_tok <= 0 and text:
                out_tok = self._count(text)
            # Gemini 3 grounding 按实际去重搜索查询数计费；响应未给出查询列表时按 1 次计
            billed = (1 if n_queries is None else n_queries) if grounding else 0
            await self.cost_guard.settle(r, Usage(in_tok, out_tok), model=model, units=billed if grounding else None)
            if grounding:
                g_calls += billed
                self.grounding_calls += billed
            records.append(AttemptRecord(model, "ok"))
            self.chain.report_success(model, self.clock.now())
            return GeneratedText(
                text=text,
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                finish_reason=finish,
                grounded=grounding,
                sources=list(sources),
                attempts=attempt + 1,
                grounding_calls=g_calls,
            )

        self.chain.report_failure(model, self.clock.now())
        raise GeminiCallFailed(
            f"gemini call failed after {self.max_retries + 1} attempts on {model}: {last_kind}",
            kind=last_kind, model=model, attempts=self.max_retries + 1, grounding_calls=g_calls, cause=last_exc,
        )


__all__ = [
    "AttemptRecord",
    "BackendResult",
    "GeminiBackend",
    "GeminiBudgetExhausted",
    "GeminiCallFailed",
    "GeminiClient",
    "GeneratedText",
    "VertexBackend",
    "classify_gemini_error",
    "normalize_response",
]
