"""上线前自检（``loadtest doctor``）：用极少量真实调用验证每个外部依赖。

检查项（按顺序，互不阻断，除非前置条件缺失）：

1. tokenizer —— tiktoken o200k_base 可加载（首次需联网下载或预置缓存）。
2. database  —— SQLite 可打开（含 ledger 迁移），报告语料库存是否够一次运行。
3. azure     —— 发送 1 个 "Reply with OK." 请求（max_tokens=5），检查鉴权 / deployment 名 /
                限流头中的部署 TPM、RPM 是否覆盖 target_tpm / rpm_max。
4. gemini:*  —— 对 model_chain 中每个模型发 1 个极短请求（不开 grounding），验证模型在该
                项目/区域可用、thinking_level 被接受、输出没有被思考 token 挤占。
5. grounding —— 用主模型发 1 个 Google Search grounding 请求，报告来源数与计费查询数。

所有花费都经 CostGuard 预留/结算并写入账本（run kind = generate，run_id = doctor-*），
总花费通常 < $0.01（grounding 最多预留 3 次查询 ≈ $0.042）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

log = logging.getLogger(__name__)

Status = Literal["ok", "warn", "fail", "skip"]

AZURE_PROBE_MESSAGES: list[dict] = [{"role": "user", "content": "Reply with OK."}]
AZURE_PROBE_INPUT_TOKENS = 16
AZURE_PROBE_MAX_TOKENS = 5
GEMINI_PING_PROMPT = "Reply with the single word OK."
GEMINI_PING_IN_TOKENS = 16
# 思考 token 也计入输出上限；3.8-flash 最低为 low，留足空间以便判断思考是否挤占输出
GEMINI_PING_MAX_OUT = 512


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def _grounding_prompt(today: datetime) -> str:
    return (
        f"Today is {today.date().isoformat()}. Using Google Search, name one notable sports or "
        "economy news event from the past few days in one sentence and state its date. "
        "Do not include anything political."
    )


class Doctor:
    def __init__(
        self,
        settings: Any,
        *,
        store: Any,
        azure_client: Any = None,
        gemini_backend: Any = None,
        now: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self.s = settings
        self.store = store
        self._azure_client = azure_client
        self._gemini_backend = gemini_backend
        self._now = now
        self._wall = wall
        self.run_id = f"doctor-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.checks: list[Check] = []
        self.azure_guard: Any = None
        self.gemini_guard: Any = None

    def add(self, name: str, status: Status, detail: str) -> None:
        self.checks.append(Check(name, status, detail))

    # ------------------------------------------------------------------ 本地

    def check_tokenizer(self) -> None:
        from loadtest import tokens

        try:
            n = tokens.count("hello world 你好")
        except Exception as e:  # noqa: BLE001
            self.add(
                "tokenizer", "fail",
                f"tiktoken o200k_base 加载失败（{type(e).__name__}）；需联网下载一次，或预置 TIKTOKEN_CACHE_DIR",
            )
            return
        self.add("tokenizer", "ok", f"o200k_base 可用（样例 {n} token）")

    async def check_database(self) -> None:
        stats = await self.store.stats()
        need = int(self.s.corpus.min_runnable)
        healthy = int(self.s.corpus.healthy_size)
        msg = f"active={stats.active}（最低 {need}，健康 {healthy}）"
        if stats.active < need:
            self.add("database", "warn", msg + "；正式运行前请先 `loadtest warmup`")
        else:
            self.add("database", "ok", msg)

    # ------------------------------------------------------------------ Azure

    async def check_azure(self) -> None:
        from loadtest.cost.guard import CostGuard
        from loadtest.load.azure_client import AzureClient
        from loadtest.load.models import PlannedRequest
        from loadtest.outcome import Outcome

        az, load = self.s.azure, self.s.load
        self.azure_guard = guard = await CostGuard.azure_from_settings(self.store, self.s.cost)
        planned = PlannedRequest(
            request_id=f"{self.run_id}-azure",
            lang="en",
            messages=[dict(m) for m in AZURE_PROBE_MESSAGES],
            input_tokens=AZURE_PROBE_INPUT_TOKENS,
            max_tokens=AZURE_PROBE_MAX_TOKENS,
            passage_ids=(),
            task_id="doctor",
            target_tokens=AZURE_PROBE_INPUT_TOKENS,
        )
        r = await guard.reserve(az.model_family, planned.input_tokens, planned.max_tokens)
        if r is None:
            self.add("azure", "fail", f"Azure 预算已到上限（已用 ${guard.spent:.2f} / ${guard.cap:.2f}）")
            return
        own = self._azure_client is None
        client = self._azure_client or AzureClient.from_settings(az, load, timeout_s=30.0)
        try:
            res = await client.send(planned)
        finally:
            if own:
                await client.aclose()
        from loadtest.cost.guard import Usage

        if res.outcome is Outcome.THROTTLED:
            await guard.settle_zero(r)
        elif res.prompt_tokens is not None:
            await guard.settle(r, Usage(int(res.prompt_tokens), int(res.completion_tokens or 0),
                                        int(res.cached_tokens or 0)))
        else:
            await guard.settle(r, None)

        base = f"deployment={az.deployment} status={res.status} {res.latency_ms:.0f}ms"
        if res.status == 401:
            self.add("azure", "fail", base + "：鉴权失败，检查 AZURE_OPENAI_API_KEY 与 endpoint 是否属于同一资源")
            return
        if res.status == 404:
            self.add("azure", "fail", base + "：deployment 不存在，检查 azure.deployment（部署名，不是模型名）与 api_version")
            return
        if res.outcome is Outcome.THROTTLED:
            self.add("azure", "warn", base + f"：被限流（retry-after={res.retry_after_s}），稍后重试 doctor")
            return
        if res.outcome is not Outcome.OK:
            self.add("azure", "fail", base + f"：{res.outcome.value} code={res.error_code}")
            return
        self.add(
            "azure", "ok",
            base + f" usage={res.prompt_tokens}/{res.completion_tokens} finish={res.finish_reason}",
        )

        # 限流头：部署实际 TPM / RPM 是否覆盖目标
        h = res.headers
        target_tpm, rpm_max = int(load.target_tpm), int(load.rpm_max)
        if h.limit_tokens is None and h.limit_requests is None:
            self.add("azure:quota", "warn", "响应未带 x-ratelimit-limit-* 头，无法核对部署配额；请在 Azure 门户确认")
            return
        problems: list[str] = []
        if h.limit_tokens is not None and h.limit_tokens < target_tpm:
            problems.append(f"部署 TPM {h.limit_tokens:,} < target_tpm {target_tpm:,}")
        if h.limit_requests is not None and h.limit_requests < rpm_max:
            problems.append(f"部署 RPM {h.limit_requests:,} < rpm_max {rpm_max:,}")
        detail = (
            f"部署限额 TPM={h.limit_tokens if h.limit_tokens is not None else '?'} "
            f"RPM={h.limit_requests if h.limit_requests is not None else '?'}；"
            f"配置 quota_tpm={az.quota_tpm:,} quota_rpm={az.quota_rpm:,}"
        )
        if problems:
            self.add("azure:quota", "warn", detail + "；" + "；".join(problems) + "（调高部署配额，或运行时用 --tpm 降低目标）")
        else:
            self.add("azure:quota", "ok", detail)

    # ------------------------------------------------------------------ Gemini

    def _backend(self) -> Any:
        if self._gemini_backend is None:
            from loadtest.corpus.gemini_client import VertexBackend

            gs = self.s.gemini
            self._gemini_backend = VertexBackend(
                gs.project, gs.location, gs.credentials_file, thinking_levels=gs.thinking_levels
            )
        return self._gemini_backend

    async def _gemini_call(
        self, model: str, prompt: str, grounding: bool, max_out: int
    ) -> tuple[Any, float, str | None, int]:
        """返回 (raw|None, 耗时秒, 错误描述|None, 计费查询数)。所有花费经 gemini_guard 结算。"""
        from loadtest.corpus.gemini_client import classify_gemini_error, grounding_query_count, normalize_response
        from loadtest.cost.guard import Usage

        gs = self.s.gemini
        guard = self.gemini_guard
        reserve_q = int(getattr(gs, "grounding_reserve_queries", 3)) if grounding else 0
        r = await guard.reserve(model, GEMINI_PING_IN_TOKENS + len(prompt) // 3, max_out, units=reserve_q)
        if r is None:
            return None, 0.0, f"Gemini 预算已到上限（已用 ${guard.spent:.2f} / ${guard.cap:.2f}）", 0
        t0 = self._now()
        try:
            raw = await asyncio.wait_for(
                self._backend().generate(
                    model, prompt, grounding, float(gs.timeout_s),
                    max_output_tokens=max_out, temperature=float(gs.temperature),
                ),
                timeout=float(gs.timeout_s),
            )
        except asyncio.CancelledError:
            await guard.release(r)
            raise
        except Exception as e:  # noqa: BLE001
            kind, _ = classify_gemini_error(e)
            if kind in ("server", "timeout"):
                await guard.settle(r, None)
            else:
                await guard.settle_zero(r)
            code = getattr(e, "code", None) or getattr(e, "status_code", None) or type(e).__name__
            msg = str(getattr(e, "message", None) or e).strip()
            first = msg.splitlines()[0][:160] if msg else ""
            return None, self._now() - t0, f"{kind} {code} {first}".strip(), 0
        elapsed = self._now() - t0
        _, _, in_tok, out_tok, _ = normalize_response(raw)
        billed = 0
        if grounding:
            n = grounding_query_count(raw)
            billed = 1 if n is None else n
        await guard.settle(
            r, Usage(in_tok or GEMINI_PING_IN_TOKENS, out_tok), model=model, units=billed if grounding else None
        )
        return raw, elapsed, None, billed

    async def check_gemini_models(self) -> None:
        from loadtest.corpus.gemini_client import normalize_response
        from loadtest.cost.guard import CostGuard

        gs = self.s.gemini
        self.gemini_guard = await CostGuard.gemini_from_settings(self.store, self.s.cost, gs.project)
        for model in gs.model_chain:
            name = f"gemini:{model}"
            level = gs.thinking_levels.get(model, "-")
            raw, dt, err, _ = await self._gemini_call(model, GEMINI_PING_PROMPT, False, GEMINI_PING_MAX_OUT)
            if err is not None:
                hint = ""
                if err.startswith("client"):
                    hint = "；模型名/区域不可用或 thinking_level 不被支持（404/400），检查 model_chain、location、thinking_levels"
                status: Status = "fail" if err.startswith("client") or "预算" in err else "warn"
                self.add(name, status, f"thinking={level}：{err}{hint}")
                continue
            text, finish, in_tok, out_tok, _ = normalize_response(raw)
            detail = f"thinking={level} {dt * 1000:.0f}ms in={in_tok} out(含思考)={out_tok} finish={finish}"
            if not text.strip():
                self.add(name, "warn", detail + "：没有正文输出（思考 token 可能挤占了输出上限）")
            elif finish not in ("STOP", ""):
                self.add(name, "warn", detail + f"：finish_reason={finish}")
            else:
                self.add(name, "ok", detail + f" text={text.strip()[:20]!r}")

    async def check_grounding(self) -> None:
        from loadtest.corpus.gemini_client import normalize_response

        gs = self.s.gemini
        model = gs.model_chain[0]
        today = datetime.fromtimestamp(self._wall(), tz=timezone.utc)
        raw, dt, err, billed = await self._gemini_call(model, _grounding_prompt(today), True, 1024)
        if err is not None:
            self.add("grounding", "fail" if err.startswith("client") else "warn", f"{model}：{err}")
            return
        text, finish, _, _, sources = normalize_response(raw)
        detail = f"{model} {dt * 1000:.0f}ms 来源 {len(sources)} 个，计费查询 {billed} 次"
        if not sources:
            self.add("grounding", "warn", detail + "：未返回来源，grounding 可能未生效（新闻类语料会被过滤为 grounding 拒绝）")
        else:
            self.add("grounding", "ok", detail + f"：{text.strip()[:60]!r}")

    # ------------------------------------------------------------------ 汇总

    async def run(self, *, azure: bool = True, gemini: bool = True, grounding: bool = True) -> list[Check]:
        self.check_tokenizer()
        await self.check_database()
        await self.store.create_run(self.run_id, "generate", json.dumps({"doctor": True}))
        try:
            if azure:
                try:
                    await self.check_azure()
                except Exception as e:  # noqa: BLE001
                    log.exception("azure check crashed")
                    self.add("azure", "fail", f"检查异常：{type(e).__name__}")
            else:
                self.add("azure", "skip", "--skip-azure")
            if gemini:
                try:
                    await self.check_gemini_models()
                    if grounding:
                        await self.check_grounding()
                    else:
                        self.add("grounding", "skip", "--no-grounding")
                except Exception as e:  # noqa: BLE001
                    log.exception("gemini check crashed")
                    self.add("gemini", "fail", f"检查异常：{type(e).__name__}（凭据文件或网络）")
            else:
                self.add("gemini", "skip", "--skip-gemini")
        finally:
            spent = 0.0
            for g in (self.azure_guard, self.gemini_guard):
                if g is not None:
                    spent += g.run_spent
                    await g.flush(self.store, self.run_id)
            await self.store.finish_run(self.run_id, "doctor")
        self.add("cost", "ok", self._budget_line(spent))
        return self.checks

    def _budget_line(self, spent: float) -> str:
        parts = [f"本次自检花费 ${spent:.4f}"]
        if self.azure_guard is not None:
            g = self.azure_guard
            parts.append(f"Azure 已用 ${g.spent:.2f}/${g.cap:.2f}")
        if self.gemini_guard is not None:
            g = self.gemini_guard
            parts.append(f"Gemini[{g.account or '-'}] 已用 ${g.spent:.2f}/${g.cap:.2f}")
        return "；".join(parts)


async def run_doctor(
    settings: Any,
    *,
    azure: bool = True,
    gemini: bool = True,
    grounding: bool = True,
    azure_client: Any = None,
    gemini_backend: Any = None,
) -> list[Check]:
    from loadtest.clock import RealClock
    from loadtest.corpus.store import CorpusStore

    store = await CorpusStore(settings.db_path, RealClock()).open()
    try:
        doc = Doctor(settings, store=store, azure_client=azure_client, gemini_backend=gemini_backend)
        return await doc.run(azure=azure, gemini=gemini, grounding=grounding)
    finally:
        await store.close()


__all__ = ["Check", "Doctor", "run_doctor"]
