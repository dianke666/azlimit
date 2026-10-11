"""请求 worker：发送单个 PlannedRequest 并处理结果（design.md 第 4 节 worker、任务 15.1）。

处理顺序（Requirements 19.1, 19.5, 19.6, 19.7, 20.1, 20.3, 21.1, 23.1, 24.1, 24.4, 25.3）：

1. ``res = await client.send(planned)``；任何异常（取消除外）视为 SERVER_ERROR 结果。
2. 收到响应后、在任何 ``await`` 之前同步调用 ``throttle.on_result(ticket, res, now)``
   （探测 → on_probe_result；普通 THROTTLED → on_throttle(retry_after, now, ticket.dispatched_at)
   以识别在途旧请求；OK → on_success(headers) 主动减速），使 Throttle_Event 立即生效（19.1）。
   同步记录熔断样本（OK / SERVER_ERROR / TIMEOUT / CLIENT_ERROR，含探测，23.1）、
   内容过滤 60s 窗口命中（24.3）与指标 / JSONL（含 is_probe）。
3. 结算：THROTTLED → settle_zero（19.5）；有 usage → 按 usage（25.3）；无 usage → 按预留。
4. CONTENT_FILTER → 隔离全部段落（探测同样隔离，20.3 / 24.1）。
5. ``finally``：``throttle.note_done(ticket)``、释放并发信号量。
   被取消（drain 超时 / 二次 SIGINT）且尚未结算时按预留金额结算（27.4 / 27.6）。
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loadtest.cost.guard import Usage
from loadtest.load.azure_client import AzureResult
from loadtest.load.quarantine import quarantine_passages
from loadtest.outcome import Outcome

log = logging.getLogger(__name__)

#: 计入熔断样本的结果（23.1）；THROTTLED / CONTENT_FILTER 不计入
BREAKER_OUTCOMES = frozenset({Outcome.OK, Outcome.SERVER_ERROR, Outcome.TIMEOUT, Outcome.CLIENT_ERROR})


class ContentFilterWindow:
    """内容过滤 60 秒滑动窗口（Requirement 24.3）。

    任一 ``window_s`` 窗口内命中次数 > ``threshold`` → ``pause_until = now + pause_s``，
    期间调度器不派发新请求，并输出一条包含窗口内命中次数的告警。
    """

    def __init__(self, window_s: float = 60.0, threshold: int = 5, pause_s: float = 30.0) -> None:
        self.window_s = float(window_s)
        self.threshold = int(threshold)
        self.pause_s = float(pause_s)
        self.hits: deque[float] = deque()
        self.pause_until = float("-inf")
        self.total_hits = 0
        self.pauses = 0  # 触发暂停的次数
        self.warnings: list[tuple[float, int]] = []  # (时刻, 窗口内命中数)

    def _prune(self, now: float) -> None:
        lo = now - self.window_s
        while self.hits and self.hits[0] <= lo:
            self.hits.popleft()

    def record(self, now: float) -> bool:
        """记录一次命中；触发（或延长）暂停时返回 True。"""
        self.total_hits += 1
        self.hits.append(now)
        self._prune(now)
        n = len(self.hits)
        if n > self.threshold:
            if now >= self.pause_until:
                self.pauses += 1
                self.warnings.append((now, n))
                log.warning(
                    "content filter hit %d times in the last %.0fs; pausing dispatch for %.0fs",
                    n, self.window_s, self.pause_s,
                )
            self.pause_until = max(self.pause_until, now + self.pause_s)
            return True
        return False

    def count(self, now: float) -> int:
        self._prune(now)
        return len(self.hits)

    def paused(self, now: float) -> bool:
        return now < self.pause_until


@dataclass
class WorkerContext:
    """worker 所需组件。``metrics`` / ``store`` / ``jsonl`` / ``semaphore`` 可为 None。"""

    client: Any  # AzureClient 或兼容的 ``async send(planned) -> AzureResult``
    throttle: Any  # ThrottleController
    breaker: Any  # CircuitBreaker
    cost_guard: Any  # CostGuard
    index: Any  # PassageIndex
    clock: Any  # Clock（now() / wall()）
    metrics: Any = None  # MetricsCollector
    store: Any = None  # CorpusStore（quarantine）
    run_id: str | None = None
    cf_window: ContentFilterWindow = field(default_factory=ContentFilterWindow)
    jsonl: Any = None  # JsonlWriter（write_result）
    semaphore: asyncio.Semaphore | None = None
    #: 可选回调：每个结果处理完毕（同步部分）后调用 (planned, result, ticket, now)
    on_result: Callable[[Any, Any, Any, float], None] | None = None

    def now(self) -> float:
        c = self.clock
        return float(c.now() if hasattr(c, "now") else c())

    def wall(self) -> float | None:
        w = getattr(self.clock, "wall", None)
        return float(w()) if callable(w) else None


def _error_result(planned: Any, exc: BaseException) -> AzureResult:
    return AzureResult(
        request_id=str(getattr(planned, "request_id", "")),
        outcome=Outcome.SERVER_ERROR,
        status=None,
        latency_ms=0.0,
        error_code=type(exc).__name__,
    )


def _as_outcome(o: Any) -> Outcome:
    return o if isinstance(o, Outcome) else Outcome(getattr(o, "value", o))


def _usage_of(res: Any) -> Usage | None:
    prompt = getattr(res, "prompt_tokens", None)
    completion = getattr(res, "completion_tokens", None)
    if prompt is None and completion is None:
        return None
    return Usage(int(prompt or 0), int(completion or 0), int(getattr(res, "cached_tokens", None) or 0))


def handle_result_sync(ctx: WorkerContext, planned: Any, res: Any, ticket: Any, now: float) -> Outcome:
    """处理响应中必须同步完成的部分（不含任何 await）：节流、熔断、内容过滤窗口、指标。"""
    outcome = _as_outcome(res.outcome)
    ctx.throttle.on_result(ticket, res, now)  # 19.1：处理响应时同步进入 HOLD
    if outcome in BREAKER_OUTCOMES:
        ctx.breaker.record(outcome, now)  # 23.1（含探测请求）
    if outcome is Outcome.CONTENT_FILTER:
        # 先从索引移除，保证同一运行立即不可再选（DB 写入在 await 之后）
        for pid in getattr(planned, "passage_ids", ()) or ():
            ctx.index.remove(int(pid))
        ctx.cf_window.record(now)  # 24.3
    if ctx.metrics is not None:
        ctx.metrics.record_result(planned, res, now, is_probe=ticket.is_probe)
    if ctx.jsonl is not None:
        try:
            ctx.jsonl.write_result(planned, res, ctx.wall(), is_probe=ticket.is_probe)
        except Exception:  # noqa: BLE001 —— 指标写入失败不影响压测
            log.exception("jsonl write failed")
    if ctx.on_result is not None:
        ctx.on_result(planned, res, ticket, now)
    return outcome


async def run_request(ctx: WorkerContext, planned: Any, reservation: Any, ticket: Any) -> Any:
    """发送一个已派发的请求并完成结算。返回 AzureResult（被取消时抛出 CancelledError）。"""
    settled = False
    try:
        try:
            res = await ctx.client.send(planned)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 —— 客户端异常按 SERVER_ERROR 处理
            log.warning("azure send raised %s; treating as SERVER_ERROR", type(e).__name__)
            res = _error_result(planned, e)

        now = ctx.now()
        outcome = handle_result_sync(ctx, planned, res, ticket, now)

        # ---- 结算（CostGuard 方法内部无真正挂起点）----
        settled = True
        if outcome is Outcome.THROTTLED:
            await ctx.cost_guard.settle_zero(reservation)  # 19.5
        else:
            await ctx.cost_guard.settle(reservation, _usage_of(res))  # 25.3：无 usage → 按预留

        # ---- 内容过滤隔离（24.1；探测命中同样隔离）----
        if outcome is Outcome.CONTENT_FILTER:
            try:
                if ctx.store is not None:
                    await quarantine_passages(
                        ctx.store, ctx.index, planned.passage_ids, getattr(res, "filter_detail", None), ctx.run_id
                    )
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 —— 索引已移除；DB 写入失败只记录
                log.exception("quarantine write failed for request %s", getattr(planned, "request_id", "?"))
        return res
    except asyncio.CancelledError:
        if not settled:
            # drain 超时 / 立即退出：按预留金额结算（27.4 / 27.6）。settle 内部不挂起，取消期间可安全完成。
            settled = True
            try:
                await ctx.cost_guard.settle(reservation, None)
            except Exception:  # noqa: BLE001
                log.exception("settle on cancel failed")
        raise
    finally:
        ctx.throttle.note_done(ticket)
        if ctx.semaphore is not None:
            ctx.semaphore.release()


__all__ = ["BREAKER_OUTCOMES", "ContentFilterWindow", "WorkerContext", "handle_result_sync", "run_request"]
