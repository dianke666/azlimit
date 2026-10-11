"""RunController 单元测试（Requirements 16.8, 22.4, 22.8, 22.9, 25.7, 26.5, 27.2, 27.6, 27.9 等）。

全部在 SimClock 离散事件模拟下运行（不真正等待），使用内存 FakeStore 与 FakeClient，无网络调用。
本模块同时提供 runner 测试共用的辅助（``make_settings`` / ``FakeStore`` / ``FakeClient`` /
``IndexAssembler`` / ``run_ctrl``），供 ``test_runner_supply`` 与 ``test_graceful_stop`` 复用。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable, Coroutine
from pathlib import Path
from random import Random
from typing import Any

import pytest

from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import LedgerEntry, LedgerWriteError, Passage, PoolStats, throttle_to_json
from loadtest.cost.guard import CostGuard
from loadtest.load.assembler import CorpusExhausted
from loadtest.load.azure_client import AzureResult
from loadtest.load.models import PlannedRequest
from loadtest.load.runner import (
    EXIT_CIRCUIT,
    EXIT_CONFIG,
    EXIT_COST,
    EXIT_ERROR,
    EXIT_LIMITS,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_PROBE,
    RunController,
    RunOptions,
    RunResult,
    check_limits,
    pid_alive,
    run_from_config,
)
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import CostSettings, Settings
from loadtest.throttle import ThrottleState
from tests.property.test_global_window import SimClock, run_sim

PRICES = CostSettings().prices
BUCKET_LEN = {"S": 650, "M": 1000, "L": 1350}

# ====================================================================== 共用辅助


def make_settings(root: Path, *, load: dict | None = None, corpus: dict | None = None,
                  cost: dict | None = None) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    cred = root / "cred.json"
    cred.write_text("{}", encoding="utf-8")
    return Settings(
        azure={"endpoint": "https://example.openai.azure.com/", "api_key": "sk-test-secret-key"},
        gemini={"credentials_file": str(cred), "project": "p"},
        load={"pid_file": str(root / "runs" / "loadtest.pid"), "duration_s": 60, **(load or {})},
        corpus={"min_runnable": 10, "healthy_size": 20, **(corpus or {})},
        cost=cost or {},
        log_dir=str(root / "runs"),
        db_path=str(root / "corpus.db"),
    )


def make_passages(zh: int = 1800, en: int = 1200) -> list[Passage]:
    """默认规模足够 600s 默认 TPM 运行（zh 需 1185、en 需 790）。"""
    out: list[Passage] = []
    pid = 1
    for lang, n in (("zh", zh), ("en", en)):
        for i in range(n):
            b = "SML"[i % 3]
            out.append(Passage(id=pid, category="c", lang=lang, token_len=BUCKET_LEN[b], use_count=0,
                               expires_at=None, text="x", length_bucket=b))
            pid += 1
    return out


class FakeStore:
    """CorpusStore 的内存替身（runner 用到的全部方法）。"""

    def __init__(self, passages: list[Passage] | None = None, *, fail_ledger: bool = False) -> None:
        self.passages = list(make_passages() if passages is None else passages)
        self.fail_ledger = fail_ledger
        self.runs: dict[str, dict[str, Any]] = {}
        self.ledger: list[LedgerEntry] = []
        self.ledger_attempts = 0
        self.use_counts: dict[int, int] = {}
        self.use_flushes: list[dict[int, int]] = []
        self.quarantined: list[int] = []
        self.calls: list[str] = []

    async def stats(self, now: Any = None) -> PoolStats:
        self.calls.append("stats")
        st = PoolStats()
        for p in self.passages:
            st.active += 1
            st.by_lang[p.lang] = st.by_lang.get(p.lang, 0) + 1
            k = (p.lang, p.length_bucket)
            st.by_bucket[k] = st.by_bucket.get(k, 0) + 1
        return st

    async def load_active_passages(self, now: Any = None) -> list[Passage]:
        self.calls.append("load_active_passages")
        return list(self.passages)

    async def create_run(self, run_id: str, kind: str, config_json: str, *, started_at: Any = None) -> None:
        self.calls.append("create_run")
        self.runs[run_id] = {"kind": kind, "config_json": config_json, "cost_usd": 0.0}

    async def update_run(self, run_id: str, *, throttle_stats: Any = None, **cols: Any) -> None:
        if throttle_stats is not None:
            cols["throttle_json"] = throttle_to_json(throttle_stats)
        self.runs[run_id].update(cols)

    async def finish_run(self, run_id: str, stop_reason: str, *, ended_at: Any = None,
                         throttle_stats: Any = None, **counters: Any) -> None:
        self.calls.append("finish_run")
        await self.update_run(run_id, throttle_stats=throttle_stats, stop_reason=stop_reason, ended=True,
                              **counters)

    async def flush_use_counts(self, counts: dict[int, int]) -> None:
        self.use_flushes.append(dict(counts))
        for pid, n in counts.items():
            self.use_counts[pid] = self.use_counts.get(pid, 0) + n

    async def write_ledger(self, entries: list[LedgerEntry]) -> None:
        self.ledger_attempts += 1
        if self.fail_ledger:
            raise LedgerWriteError("simulated ledger failure")
        self.ledger.extend(entries)
        for e in entries:
            if e.run_id in self.runs:
                self.runs[e.run_id]["cost_usd"] += e.cost_usd

    async def ledger_totals(self, providers: Any = None) -> list:
        return []

    async def quarantine(self, ids: Any, reason: str, detail: Any = None, run_id: Any = None) -> int:
        self.quarantined.extend(int(i) for i in ids)
        return len(list(ids))

    async def close(self) -> None:
        self.calls.append("close")

    @property
    def ledger_cost(self) -> float:
        return sum(e.cost_usd for e in self.ledger)


ProbeResult = tuple[Outcome, int | None, float | None]  # (outcome, status, retry_after)
Behavior = Callable[[Any, float, int], tuple[float, Outcome, int | None, float | None]]


def ok_behavior(latency: float = 1.0) -> Behavior:
    return lambda planned, now, n: (latency, Outcome.OK, 200, None)


class FakeClient:
    """脚本化 Azure 客户端。启动探测（task_id == startup_probe）按 probe_results 依次返回（立即返回）；
    其余请求按 behavior(planned, now, n) → (latency, outcome, status, retry_after)。"""

    def __init__(self, clock: SimClock, probe_results: list[ProbeResult] | None = None,
                 behavior: Behavior | None = None) -> None:
        self.clock = clock
        self.probe_results = list(probe_results or [])
        self.behavior = behavior or ok_behavior()
        self.sent: list[tuple[float, str, str]] = []  # (t, request_id, task_id)
        self.done: dict[str, float] = {}  # request_id -> 完成时刻
        self.planned: dict[str, Any] = {}
        self.n = 0
        self.closed = False

    async def send(self, planned: Any) -> AzureResult:
        now = self.clock.now()
        self.sent.append((now, planned.request_id, planned.task_id))
        self.planned[planned.request_id] = planned
        if planned.task_id == "startup_probe":
            outcome, status, ra = self.probe_results.pop(0) if self.probe_results else (Outcome.OK, 200, None)
            latency = 0.0
        else:
            self.n += 1
            latency, outcome, status, ra = self.behavior(planned, now, self.n)
            await self.clock.sleep(latency)
        self.done[planned.request_id] = self.clock.now()
        usage = outcome is Outcome.OK
        return AzureResult(
            request_id=planned.request_id,
            outcome=outcome,
            status=status,
            latency_ms=latency * 1000,
            prompt_tokens=planned.input_tokens if usage else None,
            completion_tokens=(1 if planned.task_id == "startup_probe" else 30) if usage else None,
            cached_tokens=0 if usage else None,
            retry_after_s=ra,
            headers=RateLimitHeaders(),
        )

    async def aclose(self) -> None:
        self.closed = True

    @property
    def probe_sends(self) -> list[float]:
        return [t for t, _, task in self.sent if task == "startup_probe"]

    @property
    def load_sends(self) -> list[float]:
        return [t for t, _, task in self.sent if task != "startup_probe"]


class IndexAssembler:
    """不依赖 tiktoken 的组装器：从 PassageIndex 选 1 个段落并 mark_used；input_tokens = target。"""

    def __init__(self, index: PassageIndex, lang_ratio: dict[str, float] | None = None) -> None:
        self.index = index
        self.ratio = lang_ratio or {"zh": 0.6, "en": 0.4}
        self.n = 0

    def assemble(self, target: int, rng: Random, now: float | None = None) -> PlannedRequest:
        lang = "zh" if rng.random() < self.ratio["zh"] else "en"
        p = self.index.pick(lang, 1000, rng=rng, now=now)
        if p is None:
            raise CorpusExhausted(lang)
        self.index.mark_used(p.id)
        self.n += 1
        return PlannedRequest(
            request_id=f"r{self.n}", lang=lang, messages=[], input_tokens=int(target), max_tokens=40,  # type: ignore[arg-type]
            passage_ids=(p.id,), task_id="t", target_tokens=int(target),
        )


class Ctx:
    def __init__(self, clock: SimClock, store: FakeStore, client: FakeClient, ctrl: RunController,
                 factory_calls: list[int]) -> None:
        self.clock, self.store, self.client, self.ctrl = clock, store, client, ctrl
        self.factory_calls = factory_calls
        self.out: list[str] = []


def make_ctrl(settings: Settings, opts: RunOptions | None = None, *, store: FakeStore | None = None,
              client: FakeClient | None = None, clock: SimClock | None = None, **kw: Any) -> Ctx:
    clock = clock or SimClock()
    store = store if store is not None else FakeStore()
    client = client or FakeClient(clock)
    calls: list[int] = []

    def client_factory(s: Settings) -> FakeClient:
        calls.append(1)
        return client

    out: list[str] = []
    kw.setdefault("assembler_factory", lambda idx, s: IndexAssembler(idx))
    kw.setdefault("run_id", "run-test")
    ctrl = RunController(
        settings, opts or RunOptions(no_replenish=True, seed=1), clock=clock, store=store,
        client_factory=client_factory, install_signals=False, show_console=False, out=out.append, **kw,
    )
    ctx = Ctx(clock, store, client, ctrl, calls)
    ctx.out = out
    return ctx


def run_ctx(ctx: Ctx, drivers: list[Callable[[Ctx], Coroutine[Any, Any, Any]]] | None = None) -> RunResult:
    async def main() -> RunResult:
        bg = [asyncio.ensure_future(d(ctx)) for d in (drivers or [])]
        try:
            return await ctx.ctrl.run()
        finally:
            for b in bg:
                b.cancel()
            await asyncio.gather(*bg, return_exceptions=True)

    async def go() -> RunResult:
        return await run_sim(ctx.clock, main())

    return asyncio.run(go())


def at(t: float, fn: Callable[[Ctx], Any]) -> Callable[[Ctx], Coroutine[Any, Any, Any]]:
    async def driver(ctx: Ctx) -> None:
        await ctx.clock.sleep(t - ctx.clock.now())
        fn(ctx)

    return driver


def opts(**kw: Any) -> RunOptions:
    kw.setdefault("no_replenish", True)
    kw.setdefault("seed", 1)
    return RunOptions(**kw)


# ====================================================================== pid 锁（27.9）


def test_pid_lock_alive_refuses(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    pf = Path(s.load.pid_file)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(str(os.getpid()), encoding="utf-8")  # 当前进程必然存活
    ctx = make_ctrl(s)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_LOCKED
    assert ctx.store.calls == [] and ctx.factory_calls == [] and res.azure_requests == 0
    assert pf.read_text(encoding="utf-8") == str(os.getpid())  # 不删除他人的 pid 文件
    assert any("pid" in m for m in res.messages)


def test_pid_lock_stale_overwritten_and_removed(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 20})
    pf = Path(s.load.pid_file)
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text("424242", encoding="utf-8")
    seen: list[str] = []
    ctx = make_ctrl(s, pid_alive_fn=lambda pid: False)
    res = run_ctx(ctx, [at(5.0, lambda c: seen.append(pf.read_text(encoding="utf-8")))])
    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    assert seen == [str(os.getpid())]  # 运行期间写入当前 pid
    assert not pf.exists()  # 结束删除


def test_pid_removed_after_prestart_failure(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    ctx = make_ctrl(s, opts(tpm=900_000))
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_LIMITS
    assert not Path(s.load.pid_file).exists()


def test_stale_stop_file_removed_at_start(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 15})
    stop = Path(s.load.pid_file).parent / "STOP"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("", encoding="utf-8")
    res = run_ctx(make_ctrl(s))
    assert res.stop_reason == "duration"


def test_pid_alive_function() -> None:
    assert pid_alive(os.getpid())
    assert not pid_alive(0)
    assert not pid_alive(2**31 - 7)


# ====================================================================== 上界校验（16.8）


@pytest.mark.parametrize(
    "tpm,rpm,ok",
    [(None, None, True), (825_000, None, True), (826_000, None, False), (900_000, None, False),
     (None, 876, True), (None, 877, False), (100_000, 500, True)],
)
def test_check_limits(tmp_path: Path, tpm: int | None, rpm: int | None, ok: bool) -> None:
    s = make_settings(tmp_path, load={"rpm_max": rpm} if rpm else None)
    assert (check_limits(s, tpm) == []) is ok


def test_tpm_over_limit_refuses_without_any_call(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    ctx = make_ctrl(s, opts(tpm=830_000))
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_LIMITS and res.azure_requests == 0
    assert ctx.factory_calls == [] and "create_run" not in ctx.store.calls
    assert any("--tpm" in m for m in res.messages)


def test_rpm_max_over_limit_refuses(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    res = run_ctx(make_ctrl(s, opts(rpm_max=900)))
    assert res.exit_code == EXIT_LIMITS and res.azure_requests == 0
    assert any("--rpm-max" in m for m in res.messages)


# ====================================================================== 成本预估（25.7）


def test_cost_estimate_over_cap_refuses(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    guard = CostGuard(1.0, 0.0, PRICES, spent=0.99)
    ctx = make_ctrl(s, cost_guard=guard)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_COST and res.stop_reason == "budget"
    assert res.azure_requests == 0 and ctx.factory_calls == []
    assert res.cost_estimate_usd is not None and res.cost_estimate_usd > 0.01
    assert ctx.store.runs["run-test"]["stop_reason"] == "budget"
    assert any("预估" in m for m in res.messages)


def test_cost_estimate_allow_partial_runs_until_budget(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    guard = CostGuard(1.0, 0.0, PRICES, spent=0.99)
    ctx = make_ctrl(s, opts(allow_partial=True), cost_guard=guard)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_OK and res.stop_reason == "budget"
    assert res.azure_requests >= 1
    assert guard.spent <= guard.cap + 1e-12 and guard.outstanding == 0


def test_cost_estimate_printed(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 10})
    res = run_ctx(make_ctrl(s))
    assert res.cost_estimate_usd is not None
    assert any("预估花费" in m for m in res.messages)


# ====================================================================== 启动探测（22.4, 22.6–22.9）


@pytest.mark.parametrize("status", [401, 404])
def test_startup_probe_401_404_exits(tmp_path: Path, status: int) -> None:
    s = make_settings(tmp_path)
    clock = SimClock()
    client = FakeClient(clock, probe_results=[(Outcome.CLIENT_ERROR, status, None)])
    guard = CostGuard(1e9, 0.0, PRICES)
    ctx = make_ctrl(s, client=client, clock=clock, cost_guard=guard)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_PROBE and res.stop_reason == "error"
    assert res.azure_requests == 1 and len(client.sent) == 1
    assert guard.outstanding == 0  # 经 CostGuard 预留并结算（22.6）
    assert ctx.store.runs["run-test"]["stop_reason"] == "error"
    assert any(str(status) in m for m in res.messages)
    assert not Path(s.load.pid_file).exists()


def test_startup_probe_throttled_starts_in_hold(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 120})
    clock = SimClock()
    client = FakeClient(clock, probe_results=[(Outcome.THROTTLED, 429, 20.0)])
    guard = CostGuard(1e9, 0.0, PRICES)
    ctx = make_ctrl(s, client=client, clock=clock, cost_guard=guard)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    assert res.state_at_main_loop_start is ThrottleState.HOLD  # 22.8
    assert res.startup_probe_outcomes == ["throttled"]
    th = ctx.ctrl.throttle
    assert th is not None
    assert th.stats.timeline[0][1:] == (ThrottleState.NORMAL, ThrottleState.HOLD)
    assert th.stats.timeline[1][1:] == (ThrottleState.HOLD, ThrottleState.PROBE)
    # HOLD 期间零派发；首个调度派发是 Throttle_Probe，且不早于 hold_until（20s）
    loads = client.load_sends
    assert loads and loads[0] >= 20.0 - 1e-9
    assert th.stats.probes_sent >= 1
    assert guard.outstanding == 0


def test_startup_probe_5xx_retries_then_fails(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    clock = SimClock()
    fails = [(Outcome.SERVER_ERROR, 500, None), (Outcome.TIMEOUT, None, None)] * 2
    client = FakeClient(clock, probe_results=list(fails))
    ctx = make_ctrl(s, client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_PROBE and res.stop_reason == "error"
    assert client.probe_sends == pytest.approx([0.0, 5.0, 10.0, 15.0])  # 5s 间隔重试 3 次（22.9）
    assert client.load_sends == []


def test_startup_probe_5xx_then_ok_starts(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 30})
    clock = SimClock()
    client = FakeClient(clock, probe_results=[(Outcome.SERVER_ERROR, 502, None), (Outcome.OK, 200, None)])
    ctx = make_ctrl(s, client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    assert client.probe_sends == pytest.approx([0.0, 5.0])
    assert res.state_at_main_loop_start is ThrottleState.NORMAL
    assert client.load_sends and min(client.load_sends) >= 5.0
    assert ctx.ctrl.throttle is not None and ctx.ctrl.throttle.stats.probes_sent == 0  # 与 Throttle_Probe 区分


def test_dry_run_no_azure_no_ledger(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 30})
    ctx = make_ctrl(s, opts(dry_run=True))
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    assert ctx.factory_calls == [] and res.azure_requests == 0 and ctx.client.sent == []
    assert ctx.store.ledger == [] and ctx.store.use_counts == {}
    assert res.summary is not None and res.summary.dispatched > 0
    assert any(m.startswith("[dry-run] 爬坡曲线") for m in res.messages)
    assert any(m.startswith("[dry-run] 语言配比") for m in res.messages)
    assert (res.run_dir / "report.json").exists()


# ====================================================================== stop_reason 判定（27.2, 27.8, 23.5, 25.5）


def test_stop_reason_duration(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 40})
    ctx = make_ctrl(s)
    res = run_ctx(ctx)
    assert res.stop_reason == "duration" and res.exit_code == EXIT_OK
    assert ctx.ctrl.stop_requested_at == pytest.approx(40.0)
    assert max(ctx.client.load_sends) <= 40.0
    run = ctx.store.runs["run-test"]
    assert run["stop_reason"] == "duration" and run["ended"]
    assert (res.run_dir / "report.json").exists() and (res.run_dir / "report.md").exists()
    assert (res.run_dir / "requests.jsonl").exists() and (res.run_dir / "metrics.csv").exists()
    rj = json.loads((res.run_dir / "report.json").read_text(encoding="utf-8"))
    assert rj["stop_reason"] == "duration"
    assert "sk-test-secret-key" not in (res.run_dir / "report.md").read_text(encoding="utf-8")


def test_stop_reason_signal(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    ctx = make_ctrl(s)
    res = run_ctx(ctx, [at(12.0, lambda c: c.ctrl.on_signal(signal.SIGTERM))])
    assert res.stop_reason == "signal" and res.exit_code == EXIT_OK
    assert max(ctx.client.load_sends) <= 12.0


def test_stop_reason_signal_with_stop_file_is_stop_cmd(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    ctx = make_ctrl(s, stop_poll_s=1000.0)  # 只靠信号判定
    stop_file = Path(s.load.pid_file).parent / "STOP"

    def fire(c: Ctx) -> None:
        stop_file.write_text("", encoding="utf-8")
        c.ctrl.on_signal(signal.SIGTERM)

    res = run_ctx(ctx, [at(8.0, fire)])
    assert res.stop_reason == "stop_cmd"
    assert not stop_file.exists()


def test_stop_reason_stop_file_only(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    ctx = make_ctrl(s)
    stop_file = Path(s.load.pid_file).parent / "STOP"
    res = run_ctx(ctx, [at(9.5, lambda c: stop_file.write_text("", encoding="utf-8"))])
    assert res.stop_reason == "stop_cmd"
    assert ctx.ctrl.stop_requested_at is not None and 9.5 <= ctx.ctrl.stop_requested_at <= 10.5 + 1e-9


def test_stop_reason_corpus_exhausted(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600}, corpus={"per_run_max_uses": 1})
    store = FakeStore(make_passages(zh=30, en=20))
    ctx = make_ctrl(s, opts(allow_low_supply=True), store=store)
    res = run_ctx(ctx)
    assert res.supply is not None and res.supply.verdict == "refuse"
    assert res.stop_reason == "corpus_exhausted" and res.exit_code == EXIT_OK
    assert any("corpus_exhausted" in m for m in res.messages)
    assert all(v == 1 for v in store.use_counts.values())


def test_stop_reason_budget(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    guard = CostGuard(0.01, 0.0, PRICES)
    ctx = make_ctrl(s, opts(allow_partial=True), cost_guard=guard)
    res = run_ctx(ctx)
    assert res.stop_reason == "budget"
    assert guard.outstanding == 0 and guard.spent <= guard.cap + 1e-12


def test_stop_reason_circuit_open(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    clock = SimClock()
    client = FakeClient(clock, behavior=lambda p, now, n: (0.5, Outcome.SERVER_ERROR, 500, None))
    ctx = make_ctrl(s, client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.stop_reason == "circuit_open" and res.exit_code == EXIT_CIRCUIT
    assert ctx.ctrl.stop_requested_at is None
    assert res.summary is not None and res.summary.t_stop < 600


# ====================================================================== 二次 SIGINT（27.6）


def test_second_sigint_exits_immediately_and_settles(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600, "drain_timeout_s": 30})
    clock = SimClock()
    client = FakeClient(clock, behavior=lambda p, now, n: (100.0, Outcome.OK, 200, None))
    guard = CostGuard(1e9, 0.0, PRICES)
    ctx = make_ctrl(s, client=client, clock=clock, cost_guard=guard)
    res = run_ctx(ctx, [at(10.0, lambda c: c.ctrl.on_signal(signal.SIGINT)),
                        at(12.0, lambda c: c.ctrl.on_signal(signal.SIGINT))])
    assert res.stop_reason == "signal"
    assert ctx.ctrl.forced
    assert clock.now() < 13.0  # 未等满 30s drain
    assert res.summary is not None and not res.summary.drained and res.summary.cancelled > 0
    assert guard.outstanding == 0 and guard.reserved == 0
    assert ctx.store.ledger_cost == pytest.approx(guard.run_spent)  # 按预留结算后已刷盘
    assert ctx.store.use_counts == ctx.ctrl.index.run_uses
    assert not Path(s.load.pid_file).exists()


def test_first_sigint_drains_up_to_timeout(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600, "drain_timeout_s": 30})
    clock = SimClock()
    client = FakeClient(clock, behavior=lambda p, now, n: (100.0, Outcome.OK, 200, None))
    ctx = make_ctrl(s, client=client, clock=clock)
    res = run_ctx(ctx, [at(10.0, lambda c: c.ctrl.on_signal(signal.SIGINT))])
    assert res.stop_reason == "signal"
    assert clock.now() == pytest.approx(40.0)
    assert res.summary is not None and res.summary.cancelled > 0


# ====================================================================== 账本写失败（26.5）


def test_ledger_write_failure_stops_with_error(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 600})
    store = FakeStore(fail_ledger=True)
    ctx = make_ctrl(s, store=store)
    res = run_ctx(ctx)
    assert res.stop_reason == "error" and res.exit_code == EXIT_ERROR
    assert ctx.ctrl.ledger_failed
    assert ctx.ctrl.stop_requested_at == pytest.approx(10.0)  # 首次 10s 刷写失败即停止
    assert store.runs["run-test"]["stop_reason"] == "error"
    assert not Path(s.load.pid_file).exists()


def test_periodic_flush_every_10s(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 45})
    ctx = make_ctrl(s)
    flush_times: list[float] = []
    orig = ctx.store.write_ledger

    async def spy(entries: Any) -> None:
        flush_times.append(ctx.clock.now())
        await orig(entries)

    ctx.store.write_ledger = spy  # type: ignore[method-assign]
    res = run_ctx(ctx)
    assert res.stop_reason == "duration"
    assert flush_times[:4] == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert len(ctx.store.use_flushes) >= 4
    assert ctx.store.use_counts == ctx.ctrl.index.run_uses  # 增量合计恰好等于 run_uses


# ====================================================================== runs.probes / throttle_json


def test_runs_probes_and_throttle_json_written(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 200})
    clock = SimClock()

    def beh(p: Any, now: float, n: int) -> tuple[float, Outcome, int, float | None]:
        if n == 20:
            return 1.0, Outcome.THROTTLED, 429, 15.0
        return 1.0, Outcome.OK, 200, None

    client = FakeClient(clock, behavior=beh)
    ctx = make_ctrl(s, client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.stop_reason == "duration"
    run = ctx.store.runs["run-test"]
    th = ctx.ctrl.throttle
    assert th is not None and th.stats.throttle_events >= 1
    assert run["probes"] == th.stats.probes_sent >= 1
    data = json.loads(run["throttle_json"])
    assert data["probes_sent"] == run["probes"] and data["throttle_events"] >= 1
    assert run["throttled"] >= 1 and run["requests_ok"] > 0
    rj = res.report
    assert rj is not None and rj["probes"] == run["probes"] and rj["throttle_timeline"]


# ====================================================================== Replenisher 接线


class FakeReplenisher:
    def __init__(self, clock: SimClock) -> None:
        self.clock = clock
        self.added = 0
        self.stopped_at: float | None = None

    async def run(self, stop: asyncio.Event, out: asyncio.Queue) -> None:
        while not stop.is_set():
            await self.clock.sleep(3.0)
            self.added += 1
        self.stopped_at = self.clock.now()


def test_replenisher_started_and_closed(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 30})
    clock = SimClock()
    rep = FakeReplenisher(clock)
    closed: list[float] = []

    async def factory(settings: Any, clk: Any, **kw: Any) -> tuple[Any, Any]:
        assert kw["run_id"] == "run-test" and isinstance(kw["queue"], asyncio.Queue)

        async def closer() -> None:
            closed.append(clk.now())

        return rep, closer

    ctx = make_ctrl(s, RunOptions(seed=1), clock=clock, replenisher_factory=factory)
    res = run_ctx(ctx)
    assert res.stop_reason == "duration"
    assert rep.stopped_at is not None and closed
    assert ctx.store.runs["run-test"]["passages_added"] == rep.added > 0


def test_replenisher_setup_failure_does_not_stop_run(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 20})

    async def factory(settings: Any, clk: Any, **kw: Any) -> Any:
        raise RuntimeError("no gemini")

    ctx = make_ctrl(s, RunOptions(seed=1), replenisher_factory=factory)
    res = run_ctx(ctx)
    assert res.stop_reason == "duration" and res.exit_code == EXIT_OK
    assert any("补充" in m for m in res.messages)


# ====================================================================== 配置加载


def test_run_from_config_invalid_exits_without_calls(tmp_path: Path) -> None:
    out: list[str] = []
    res = asyncio.run(run_from_config(tmp_path / "missing.yaml", None, RunOptions(), out=out.append))
    assert res.exit_code == EXIT_CONFIG and res.azure_requests == 0
    assert out and "config" in out[0]
