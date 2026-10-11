"""语料生成与入库：CorpusGenerator（单条流水线）+ WarmupRunner（warmup / generate 并发驱动）。

单条流水线 ``generate_and_store(spec)``（设计"语料生成与入库"）::

    build_prompt → client.generate → SafetyFilter.check → store.insert_text（去重）→ planner 回报
      SAFETY / 空正文 → taxonomy.penalize(subtopic, 0.5)，以 safety_block 拒绝（不重试，5.1）
      过滤失败       → store.record_rejection(原因) + planner.note_rejected(spec, 原因)
      重复           → store 已写 duplicate 拒绝记录；planner.note_rejected(spec, 'duplicate')
      GeminiCallFailed → planner.note_rejected(spec, 'error')
      GeminiBudgetExhausted → planner.note_rejected(spec, 'budget') 并向上抛出
      每次已计费的 grounding 尝试 → planner.note_grounding_call()

WarmupRunner.run()（Requirements 3.5, 3.7, 4.5, 4.6, 7.6, 26.1）::

    create_run(kind) → store.expire_due() → stats → planner.set_grounding_used(当日已用) → planner.reset
    缺口为 0（且不是带 --count 的 generate）→ 不发起任何 Gemini 调用，直接报告库存（7.6）
    否则 concurrency 个 worker 循环 next_spec，直到：
      缺口填满（filled）/ 达到 --count（limit）/ Gemini 预算拒绝预留（budget）/
      本次花费 ≥ --max-cost（max_cost，在发出每条新任务前检查）/ 停止事件（stopped）/
      仅剩当日 grounding 已达上限的类别（grounding_limit）
    每 flush_interval_s（按完成时检查）与结束时刷写 Gemini 账本；finish_run(passages_added=…)
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from random import Random
from typing import Any, Literal

from loadtest.clock import Clock
from loadtest.corpus.gemini_client import (
    GeminiBudgetExhausted,
    GeminiCallFailed,
    GeneratedText,
)
from loadtest.corpus.prompts import build_prompt
from loadtest.corpus.store import PoolStats
from loadtest.corpus.taxonomy import TopicSpec

log = logging.getLogger(__name__)

RunKind = Literal["warmup", "generate"]
StopReason = Literal["filled", "budget", "max_cost", "limit", "stopped", "grounding_limit", "error"]


# --------------------------------------------------------------------------- 单条流水线


@dataclass(frozen=True)
class GenOutcome:
    status: Literal["inserted", "rejected", "error"]
    reason: str | None = None  # 拒绝 / 失败原因
    passage_id: int | None = None
    model: str | None = None
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class CorpusGenerator:
    """生成 → 过滤 → 去重入库；负责 planner / taxonomy / rejections 的回报。"""

    def __init__(
        self,
        client: Any,
        store: Any,
        planner: Any,
        taxonomy: Any,
        safety: Any,
        *,
        run_id: str | None = None,
        clock: Clock | None = None,
        today: date | None = None,
        store_lock: asyncio.Lock | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.planner = planner
        self.taxonomy = taxonomy
        self.safety = safety
        self.run_id = run_id
        self.clock = clock
        self.today = today
        # CorpusStore 的单个连接不支持并发事务：同一 store 上的写操作串行化
        self.store_lock = store_lock or asyncio.Lock()

    def _today(self) -> date:
        if self.today is not None:
            return self.today
        if self.clock is not None:
            return datetime.fromtimestamp(self.clock.wall(), tz=timezone.utc).date()
        return datetime.now(timezone.utc).date()

    def _note_grounding(self, n: int) -> None:
        for _ in range(max(0, int(n))):
            self.planner.note_grounding_call()

    async def generate_and_store(self, spec: TopicSpec) -> GenOutcome:
        noted = False
        try:
            prompt = build_prompt(spec, self.taxonomy, self._today())
            try:
                out: GeneratedText = await self.client.generate(spec, prompt)
            except GeminiBudgetExhausted as e:
                self._note_grounding(e.grounding_calls)
                noted = True
                self.planner.note_rejected(spec, "budget")
                raise
            except GeminiCallFailed as e:
                self._note_grounding(e.grounding_calls)
                noted = True
                self.planner.note_rejected(spec, "error")
                return GenOutcome("error", "error", model=e.model, attempts=e.attempts)

            self._note_grounding(out.grounding_calls)
            common = dict(model=out.model, attempts=out.attempts,
                          input_tokens=out.input_tokens, output_tokens=out.output_tokens)
            fr = self.safety.check(out.text, spec, out.finish_reason, out.sources)
            if not fr.ok:
                reason = fr.reason or "structure"
                if reason == "safety_block":
                    try:
                        self.taxonomy.penalize(spec.subtopic, 0.5)
                    except KeyError:
                        pass
                async with self.store_lock:
                    await self.store.record_rejection(
                        spec.category, spec.lang, out.model, reason, run_id=self.run_id
                    )
                noted = True
                self.planner.note_rejected(spec, reason)
                return GenOutcome("rejected", reason, **common)

            async with self.store_lock:
                res = await self.store.insert_text(
                    category=spec.category,
                    subtopic=spec.subtopic,
                    genre=spec.genre,
                    lang=spec.lang,
                    token_len=fr.token_len,
                    length_bucket=fr.length_bucket or spec.length_bucket,
                    text=out.text,
                    gen_model=out.model,
                    grounded=out.grounded,
                    sources=tuple(out.sources),
                    run_id=self.run_id,
                )
            noted = True
            if not res.inserted:  # store 已以 duplicate 写入 rejections
                self.planner.note_rejected(spec, "duplicate")
                return GenOutcome("rejected", "duplicate", **common)
            self.planner.note_inserted(spec)
            return GenOutcome("inserted", None, passage_id=res.id, **common)
        except BaseException:
            if not noted:  # 异常路径也要归还 planner 占用的缺口
                self.planner.note_rejected(spec, "error")
            raise


# --------------------------------------------------------------------------- WarmupRunner


@dataclass
class WarmupReport:
    run_id: str
    kind: str
    stop_reason: str
    inserted: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    tasks: int = 0  # 发出的生成任务数
    calls: int = 0  # Gemini 尝试次数（含重试）
    cost_usd: float = 0.0  # 本次运行 Gemini 花费
    input_tokens: int = 0
    output_tokens: int = 0
    deficit_at_start: int = 0
    inventory: PoolStats = field(default_factory=PoolStats)

    @property
    def rejected(self) -> int:
        return sum(self.rejected_by_reason.values())

    @property
    def rejection_rate(self) -> float:
        done = self.inserted + self.rejected
        return self.rejected / done if done else 0.0

    @property
    def by_lang(self) -> dict[str, int]:
        return dict(self.inventory.by_lang)

    @property
    def by_category(self) -> dict[tuple[str, str], int]:
        return dict(self.inventory.by_category)

    @property
    def by_bucket(self) -> dict[tuple[str, str], int]:
        return dict(self.inventory.by_bucket)


class WarmupRunner:
    def __init__(
        self,
        settings: Any,
        store: Any,
        client: Any,
        planner: Any,
        taxonomy: Any,
        safety: Any,
        cost_guard: Any,
        *,
        clock: Clock,
        rng: Random,
        run_id: str,
        concurrency: int = 32,
        max_cost: float = 10.0,
        kind: RunKind = "warmup",
        stop_event: asyncio.Event | None = None,
        flush_interval_s: float = 10.0,
        config_json: str | None = None,
        today: date | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if kind not in ("warmup", "generate"):
            raise ValueError("kind must be 'warmup' or 'generate'")
        self.settings = settings
        self.store = store
        self.client = client
        self.planner = planner
        self.taxonomy = taxonomy
        self.safety = safety
        self.cost_guard = cost_guard
        self.clock = clock
        self.rng = rng
        self.run_id = run_id
        self.concurrency = int(concurrency)
        self.max_cost = float(max_cost)
        self.kind = kind
        self.stop_event = stop_event or asyncio.Event()
        self.flush_interval_s = float(flush_interval_s)
        self.config_json = config_json
        self._store_lock = asyncio.Lock()
        self.generator = CorpusGenerator(
            client, store, planner, taxonomy, safety, run_id=run_id, clock=clock, today=today,
            store_lock=self._store_lock,
        )
        self._stop_reason: str | None = None
        self._last_flush = 0.0

    # ---- 辅助 ----
    def _config_json(self) -> str:
        if self.config_json is not None:
            return self.config_json
        try:
            from loadtest.settings import Settings, config_snapshot_json

            if isinstance(self.settings, Settings):
                return config_snapshot_json(self.settings)
        except Exception:  # pragma: no cover - 快照失败不影响生成
            log.exception("config snapshot failed")
        return "{}"

    def _spent(self) -> float:
        return float(getattr(self.cost_guard, "run_spent", 0.0)) + float(getattr(self.cost_guard, "reserved", 0.0))

    def _stop(self, reason: str) -> None:
        if self._stop_reason is None:
            self._stop_reason = reason
            log.info("%s %s stopping: %s", self.kind, self.run_id, reason)

    def _should_stop(self) -> bool:
        if self._stop_reason is not None:
            return True
        if self.stop_event.is_set():
            self._stop("stopped")
            return True
        if self._spent() >= self.max_cost:
            self._stop("max_cost")
            return True
        return False

    async def _flush(self, force: bool = False) -> None:
        now = self.clock.now()
        if not force and now - self._last_flush < self.flush_interval_s:
            return
        async with self._store_lock:
            self._last_flush = now
            await self.cost_guard.flush(self.store, self.run_id)

    # ---- 主流程 ----
    async def run(self) -> WarmupReport:
        report = WarmupReport(self.run_id, self.kind, "filled")
        await self.store.create_run(self.run_id, self.kind, self._config_json())
        self._last_flush = self.clock.now()
        error: BaseException | None = None
        try:
            await self.store.expire_due()
            stats = await self.store.stats()
            self.planner.set_grounding_used(await self.store.grounding_calls_today())
            self.planner.reset(stats)
            deficit = sum(d.n for d in self.planner.plan(stats, self.planner.target_size))
            report.deficit_at_start = deficit
            limit = getattr(self.planner, "limit", None)
            if self.kind == "generate" and limit is not None:
                proceed = limit > 0
            else:
                proceed = deficit > 0
            if not proceed:
                self._stop("limit" if self.kind == "generate" and limit is not None else "filled")
            else:
                await self._run_workers(report)
        except BaseException as e:  # noqa: BLE001 - 记录后在 finally 之后重新抛出
            error = e
            self._stop("error")
        finally:
            try:
                await self._flush(force=True)
            except BaseException as e:  # noqa: BLE001
                log.exception("final ledger flush failed")
                if error is None:
                    error = e
                self._stop_reason = "error"
            report.stop_reason = self._final_reason()
            report.cost_usd = float(getattr(self.cost_guard, "run_spent", 0.0))
            try:
                report.inventory = await self.store.stats()
                await self.store.finish_run(
                    self.run_id,
                    report.stop_reason,
                    passages_added=report.inserted,
                    requests_ok=report.inserted + report.rejected,
                    requests_err=report.errors,
                    input_tokens=report.input_tokens,
                    output_tokens=report.output_tokens,
                )
            except BaseException:  # noqa: BLE001
                log.exception("finish_run failed")
                if error is None:
                    raise
        if error is not None:
            raise error
        return report

    def _final_reason(self) -> str:
        if self._stop_reason is not None:
            return self._stop_reason
        limit = getattr(self.planner, "limit", None)
        if limit is not None and self.planner.inserted >= limit:
            return "limit"
        if getattr(self.planner, "remaining_total", 0) > 0 and not self.planner.grounding_available():
            return "grounding_limit"
        return "filled"

    async def _run_workers(self, report: WarmupReport) -> None:
        cond = asyncio.Condition()
        rejected: Counter[str] = Counter()

        async def worker() -> None:
            while True:
                async with cond:
                    while True:
                        if self._should_stop():
                            cond.notify_all()
                            return
                        spec = self.planner.next_spec(self.rng)
                        if spec is not None:
                            break
                        if self.planner.inflight == 0:
                            cond.notify_all()
                            return
                        await cond.wait()  # 在途任务完成后可能归还缺口
                report.tasks += 1
                try:
                    out = await self.generator.generate_and_store(spec)
                except GeminiBudgetExhausted as e:
                    report.calls += e.attempts
                    self._stop("budget")
                    out = None
                finally:
                    async with cond:
                        cond.notify_all()
                if out is not None:
                    report.calls += out.attempts
                    report.input_tokens += out.input_tokens
                    report.output_tokens += out.output_tokens
                    if out.status == "inserted":
                        report.inserted += 1
                    elif out.status == "rejected":
                        rejected[out.reason or "unknown"] += 1
                    else:
                        report.errors += 1
                report.rejected_by_reason = dict(rejected)
                await self._flush()

        tasks = [asyncio.create_task(worker(), name=f"{self.kind}-worker-{i}") for i in range(self.concurrency)]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        report.rejected_by_reason = dict(rejected)
        for r in results:
            if isinstance(r, BaseException):
                raise r


# --------------------------------------------------------------------------- 组装（CLI 使用）


async def build_corpus_pipeline(
    settings: Any,
    store: Any,
    clock: Clock,
    *,
    run_id: str,
    kind: RunKind = "warmup",
    rng: Random | None = None,
    target: int | None = None,
    concurrency: int | None = None,
    max_cost: float = 10.0,
    categories: list[str] | None = None,
    langs: list[str] | None = None,
    count: int | None = None,
    stop_event: asyncio.Event | None = None,
    backend: Any = None,
) -> WarmupRunner:
    """用真实组件（VertexBackend、GeminiRateLimiter、ModelChain、Gemini CostGuard）组装 WarmupRunner。

    warmup：planner 目标为 ``target``（默认 settings.corpus.target_size）。
    generate：planner 只生成 ``categories`` / ``langs``，本次入库 ≤ ``count``（4.6）。
    """
    from loadtest.corpus.gemini_client import GeminiClient, VertexBackend
    from loadtest.corpus.gemini_limiter import GeminiRateLimiter
    from loadtest.corpus.model_chain import ModelChain
    from loadtest.corpus.planner import SupplyPlanner
    from loadtest.corpus.safety import SafetyFilter
    from loadtest.corpus.taxonomy import TopicTaxonomy
    from loadtest.cost.guard import CostGuard

    gs = settings.gemini
    rng = rng or Random()
    taxonomy = TopicTaxonomy.load(grounding_categories=gs.grounding_categories)
    cost_guard = await CostGuard.gemini_from_settings(store, settings.cost, gs.project)
    chain = ModelChain(
        gs.model_chain,
        gs.failover_after_failures,
        gs.failback_after_s,
        budget_ratio_fn=cost_guard.remaining_ratio,
        expensive=gs.expensive_models,
    )
    limiter = GeminiRateLimiter.from_settings(gs, clock, replenisher=False)
    backend = backend or VertexBackend(
        gs.project, gs.location, gs.credentials_file, thinking_levels=gs.thinking_levels
    )
    client = GeminiClient.from_settings(gs, backend, limiter, chain, cost_guard, clock=clock, rng=rng)

    def utcnow() -> datetime:
        return datetime.fromtimestamp(clock.wall(), tz=timezone.utc)

    if kind == "generate":
        planner = SupplyPlanner(
            taxonomy, settings, target_size=target, categories=categories, langs=langs, limit=count, utcnow=utcnow
        )
    else:
        planner = SupplyPlanner(taxonomy, settings, target_size=target, utcnow=utcnow)
    safety = SafetyFilter.default(settings=settings.corpus)
    return WarmupRunner(
        settings, store, client, planner, taxonomy, safety, cost_guard,
        clock=clock, rng=rng, run_id=run_id,
        concurrency=concurrency or gs.warmup_concurrency,
        max_cost=max_cost, kind=kind, stop_event=stop_event,
    )


__all__ = [
    "CorpusGenerator",
    "GenOutcome",
    "WarmupReport",
    "WarmupRunner",
    "build_corpus_pipeline",
]
