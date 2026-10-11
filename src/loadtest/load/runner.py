"""RunController：``loadtest run`` 的生命周期、信号、pid / STOP 文件（design.md 组件 13、任务 16.1）。

启动顺序（任一步失败都不会发出后续步骤的外部调用）::

    1. 加载并校验配置（``run_from_config``；失败 → EXIT_CONFIG，无任何外部调用）
    2. pid 单实例锁：pid 文件存在且进程存活 → EXIT_LOCKED；失效 pid 覆盖；退出时删除（27.9）
    3. ``--tpm`` / ``--rpm-max`` 上界校验（16.8）→ EXIT_LIMITS
    4. 供应检查（8.3 / 8.6 / 8.7）：refuse 且无 --allow-low-supply → stop_reason=supply_refused、
       0 个 Azure 请求（含探测）、输出缺口与 warmup 命令 → EXIT_SUPPLY
    5. 成本预估（25.7）：已花费 + 预估 > 上限且无 --allow-partial → EXIT_COST（0 个 Azure 请求）
    6. 启动 Probe_Request（22.4 / 22.6–22.9，--dry-run 跳过）：经 Azure CostGuard 预留 / 结算；
       401/404 → EXIT_PROBE；THROTTLED → throttle.on_throttle(retry_after) 后启动（主循环从
       HOLD → PROBE 开始）；SERVER_ERROR / TIMEOUT → 间隔 probe_retry_interval_s 重试
       probe_max_retries 次，仍失败 → EXIT_PROBE。启动探测与运行中的 Throttle_Probe 区分：
       不计入 ThrottleStats 探测计数，也不写入指标。
    7. 加载 PassageIndex → 启动 Scheduler（主循环开始 = 爬坡零点）与 Replenisher

停止（27.2–27.6, 27.8, 23.5, 26.4, 26.5）：
    SIGINT / SIGTERM（存在 runs/STOP 时为 stop_cmd，否则 signal）、runs/STOP 文件（stop_cmd）、
    duration 到期（duration）、调度器自身（budget / corpus_exhausted / circuit_open）、
    账本写入失败（error）。停止后 Scheduler 不再派发，drain ≤ drain_timeout_s，未完成请求按预留结算；
    优雅停止期间再次 SIGINT → 立即取消在途请求（按预留结算）并刷盘退出。
    运行中每 10 秒刷写账本与 use_count 增量（use_count 总增量恰好等于 run_uses，9.4）。

收尾：run_uses 累加回 use_count、刷写账本、写 runs 记录（计数、probes、throttle_json）、
生成 report.json / report.md、删除 pid 文件。

``--dry-run``（27.7）：只运行调度与组装，Azure 调用替换为模拟结果；不发送 Probe_Request、
不写 Azure 账本、不累加 use_count、不启动 Replenisher；结束时输出预期 TPM/RPM、爬坡曲线、
语言配比、复用分布与长度分布。

所有外部依赖（store、CostGuard、Azure 客户端、组装器、Replenisher、时钟、pid 存活判断、
信号安装）均可注入，便于在 FakeClock 下测试。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

from loadtest.clock import RealClock
from loadtest.corpus.index import PassageIndex
from loadtest.corpus.planner import AVG_INPUT_TOKENS, SupplyCheck, check_for_run
from loadtest.corpus.store import LedgerWriteError
from loadtest.cost.guard import CostGuard, Usage
from loadtest.load.azure_client import AzureResult
from loadtest.load.breaker import CircuitBreaker
from loadtest.load.models import PlannedRequest
from loadtest.load.scheduler import RunSummary, Scheduler
from loadtest.metrics.collector import MetricsCollector, max_run_uses
from loadtest.metrics.report import probes_count, write_report
from loadtest.metrics.writers import CsvWriter, JsonlWriter
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import (
    EXIT_CONFIG_ERROR,
    Settings,
    SettingsError,
    config_snapshot_json,
    load_settings,
    redact,
)
from loadtest.throttle import ThrottleController, ThrottleState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 退出码
# ---------------------------------------------------------------------------

EXIT_OK = 0
EXIT_ERROR = 1  # stop_reason=error（账本写入失败 / 未预期异常）
EXIT_CONFIG = EXIT_CONFIG_ERROR  # 2：配置校验失败
EXIT_LOCKED = 3  # 已有存活的压测进程
EXIT_LIMITS = 4  # --tpm / --rpm-max 超出配额上界
EXIT_SUPPLY = 5  # 供应检查 refuse
EXIT_COST = 6  # 成本预估超限
EXIT_PROBE = 7  # 启动探测 401/404 或重试耗尽
EXIT_CIRCUIT = 8  # 熔断第 3 次 open

#: 运行结束时 stop_reason → 退出码（未列出的为 EXIT_OK）
_EXIT_BY_REASON = {"error": EXIT_ERROR, "circuit_open": EXIT_CIRCUIT}

PROBE_INPUT_TOKENS = 16  # 启动探测请求的输入 token 预估（预留按此值 × 1.02）
PROBE_MAX_TOKENS = 5
PROBE_MESSAGES: list[dict] = [{"role": "user", "content": "Reply with OK."}]
QUOTA_SAFETY = 0.95  # 60s 窗口上界不得超过配额的 95%


# ---------------------------------------------------------------------------
# 选项与结果
# ---------------------------------------------------------------------------


@dataclass
class RunOptions:
    """``loadtest run`` 的命令行选项（None 表示使用配置值）。"""

    duration_s: float | None = None
    tpm: int | None = None
    rpm_max: int | None = None
    deployment: str | None = None
    no_replenish: bool = False
    no_ramp: bool = False
    allow_low_supply: bool = False
    allow_partial: bool = False
    dry_run: bool = False
    seed: int | None = None


@dataclass
class RunResult:
    exit_code: int
    stop_reason: str | None
    run_id: str | None
    messages: list[str] = field(default_factory=list)
    azure_requests: int = 0  # 真实 Azure 请求数（含启动探测）
    startup_probe_outcomes: list[str] = field(default_factory=list)
    supply: SupplyCheck | None = None
    cost_estimate_usd: float | None = None
    state_at_main_loop_start: ThrottleState | None = None
    summary: RunSummary | None = None
    report: dict[str, Any] | None = None
    run_dir: Path | None = None


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def pid_alive(pid: int) -> bool:
    """进程是否存活。Windows 上不能用 ``os.kill(pid, 0)``（会终止目标进程），改用 OpenProcess。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        error_access_denied = 5
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(process_query_limited_information, False, int(pid))
        if not h:
            return ctypes.get_last_error() == error_access_denied  # 存在但无权限
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == still_active
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _usage_of(res: Any) -> Usage | None:
    prompt = getattr(res, "prompt_tokens", None)
    completion = getattr(res, "completion_tokens", None)
    if prompt is None and completion is None:
        return None
    return Usage(int(prompt or 0), int(completion or 0), int(getattr(res, "cached_tokens", None) or 0))


def _outcome(o: Any) -> Outcome:
    return o if isinstance(o, Outcome) else Outcome(getattr(o, "value", o))


class CountingClient:
    """包装真实 Azure 客户端，统计实际发送的请求数（含启动探测）。"""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.sent = 0

    async def send(self, planned: Any) -> Any:
        self.sent += 1
        return await self.inner.send(planned)

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            await close()


class DryRunClient:
    """--dry-run 的模拟 Azure 客户端：不发网络请求，返回 OK 与模拟 usage。"""

    def __init__(self, clock: Any, rng: Random, latency_s: tuple[float, float] = (0.3, 1.5)) -> None:
        self.clock = clock
        self.rng = rng
        self.latency_s = latency_s
        self.sent = 0

    async def send(self, planned: Any) -> AzureResult:
        self.sent += 1
        lat = self.rng.uniform(*self.latency_s)
        await self.clock.sleep(lat)
        return AzureResult(
            request_id=str(planned.request_id),
            outcome=Outcome.OK,
            status=200,
            latency_ms=lat * 1000.0,
            prompt_tokens=int(planned.input_tokens),
            cached_tokens=0,
            completion_tokens=self.rng.randint(20, 35),
            headers=RateLimitHeaders(),
            finish_reason="stop",
        )


def _default_client_factory(settings: Settings, clock: Any) -> Any:
    from loadtest.load.azure_client import AzureClient

    return AzureClient.from_settings(settings.azure, settings.load, clock=clock)


def _default_assembler_factory(index: PassageIndex, settings: Settings) -> Any:
    from loadtest.load.assembler import RequestAssembler
    from loadtest.load.tasks import TaskCatalog

    return RequestAssembler(index, TaskCatalog.load(), cfg=settings.load)


async def _default_replenisher_factory(settings: Settings, clock: Any, *, run_id: str, rng: Random,
                                       queue: asyncio.Queue) -> tuple[Any, Callable[[], Awaitable[None]]]:
    from loadtest.corpus.replenisher import build_replenisher

    return await build_replenisher(settings, settings.db_path, clock, run_id=run_id, rng=rng, queue=queue)


def apply_overrides(settings: Settings, opts: RunOptions) -> Settings:
    """把 --duration / --rpm-max / --deployment 覆盖到配置副本上（--tpm 由调度器 tpm_override 处理）。"""
    load_upd: dict[str, Any] = {}
    if opts.duration_s is not None:
        load_upd["duration_s"] = int(math.ceil(opts.duration_s))
    if opts.rpm_max is not None:
        load_upd["rpm_max"] = int(opts.rpm_max)
    upd: dict[str, Any] = {}
    if load_upd:
        upd["load"] = settings.load.model_copy(update=load_upd)
    if opts.deployment:
        upd["azure"] = settings.azure.model_copy(update={"deployment": opts.deployment})
    return settings.model_copy(update=upd) if upd else settings


def check_limits(settings: Settings, tpm: float | None) -> list[str]:
    """Requirement 16.8：T × 1.0625 × (1 + burst/60) 与 rpm_max × (1 + burst/60) 不得超过配额的 95%。"""
    load, az = settings.load, settings.azure
    burst = 1.0 + load.bucket_burst_s / 60.0
    errs: list[str] = []
    if tpm is not None:
        if tpm <= 0:
            errs.append("--tpm must be > 0")
        high = float(tpm) * load.walk_high_ratio
        name = f"--tpm {int(tpm)}"
    else:
        high = float(load.tpm_walk_high)
        name = f"load.tpm_walk_high {load.tpm_walk_high}"
    if high * burst > QUOTA_SAFETY * az.quota_tpm:
        errs.append(
            f"{name}: 60s 窗口上界 {high * burst:,.0f} TPM 超过配额 95%（{QUOTA_SAFETY * az.quota_tpm:,.0f}）"
        )
    if load.rpm_max * burst > QUOTA_SAFETY * az.quota_rpm:
        errs.append(
            f"--rpm-max {load.rpm_max}: 60s 窗口上界 {load.rpm_max * burst:,.1f} RPM 超过配额 95%"
            f"（{QUOTA_SAFETY * az.quota_rpm:,.1f}）"
        )
    return errs


# ---------------------------------------------------------------------------
# RunController
# ---------------------------------------------------------------------------


class RunController:
    def __init__(
        self,
        settings: Settings,
        options: RunOptions | None = None,
        *,
        clock: Any = None,
        store: Any = None,
        cost_guard: CostGuard | None = None,
        client_factory: Callable[[Settings], Any] | None = None,
        assembler_factory: Callable[[PassageIndex, Settings], Any] | None = None,
        replenisher_factory: Callable[..., Awaitable[tuple[Any, Callable[[], Awaitable[None]]]]] | None = None,
        pid_alive_fn: Callable[[int], bool] = pid_alive,
        install_signals: bool = True,
        show_console: bool = True,
        out: Callable[[str], None] | None = None,
        run_id: str | None = None,
        report_interval_s: float = 10.0,
        flush_interval_s: float = 10.0,
        stop_poll_s: float = 1.0,
        replenisher_stop_timeout_s: float = 5.0,
        max_breaker_opens: int = 3,
    ) -> None:
        self.opts = options or RunOptions()
        self.settings = apply_overrides(settings, self.opts)
        self.clock = clock or RealClock()
        self._store = store
        self._owns_store = store is None
        self._cost_guard = cost_guard
        self.client_factory = client_factory or (lambda s: _default_client_factory(s, self.clock))
        self.assembler_factory = assembler_factory or _default_assembler_factory
        self.replenisher_factory = replenisher_factory or (
            lambda s, clock, **kw: _default_replenisher_factory(s, clock, **kw)
        )
        self.pid_alive_fn = pid_alive_fn
        self.install_signals = install_signals
        self.show_console = show_console
        self._out = out or print
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.report_interval_s = float(report_interval_s)
        self.flush_interval_s = float(flush_interval_s)
        self.stop_poll_s = float(stop_poll_s)
        self.replenisher_stop_timeout_s = float(replenisher_stop_timeout_s)
        self.max_breaker_opens = int(max_breaker_opens)

        seed = self.opts.seed if self.opts.seed is not None else Random().getrandbits(32)
        self.seed = int(seed)
        self.rng = Random(self.seed)
        self.model = self.settings.azure.model_family  # 计价键
        self.tpm_override: float | None = float(self.opts.tpm) if self.opts.tpm is not None else None
        self.duration_s = float(self.opts.duration_s if self.opts.duration_s is not None
                                else self.settings.load.duration_s)
        self.pid_file = Path(self.settings.load.pid_file)
        self.stop_file = self.pid_file.parent / "STOP"
        self.run_dir = Path(self.settings.log_dir) / self.run_id

        self.messages: list[str] = []
        self.signals_received = 0
        self.forced = False
        self.ledger_failed = False
        self.fatal_error = False
        self.external_reason: str | None = None
        self.stop_requested_at: float | None = None  # 外部停止（信号 / STOP / duration / error）时刻
        self.state_at_main_loop_start: ThrottleState | None = None
        self.counting_client: CountingClient | None = None
        self.startup_probe_outcomes: list[str] = []
        self.supply: SupplyCheck | None = None
        self.cost_estimate: float | None = None

        self.store: Any = None
        self.guard: CostGuard | None = None
        self.throttle: ThrottleController | None = None
        self.scheduler: Scheduler | None = None
        self.index: PassageIndex | None = None
        self.metrics: MetricsCollector | None = None
        self.replenisher: Any = None
        self._stop: asyncio.Event | None = None
        self._force_task: asyncio.Future | None = None
        self._flushed_uses: dict[int, int] = {}
        self._flush_lock = asyncio.Lock()
        self._lock_held = False
        self._run_created = False
        self._prev_handlers: dict[int, Any] = {}
        self._loop_handlers: list[int] = []

    # ------------------------------------------------------------------ 输出
    def emit(self, msg: str) -> None:
        msg = redact(msg)
        self.messages.append(msg)
        try:
            self._out(msg)
        except Exception:  # noqa: BLE001
            log.exception("output failed")

    @property
    def azure_requests(self) -> int:
        return self.counting_client.sent if self.counting_client is not None else 0

    def _result(self, code: int, reason: str | None, **kw: Any) -> RunResult:
        return RunResult(
            exit_code=code,
            stop_reason=reason,
            run_id=self.run_id,
            messages=list(self.messages),
            azure_requests=self.azure_requests,
            startup_probe_outcomes=list(self.startup_probe_outcomes),
            supply=self.supply,
            cost_estimate_usd=self.cost_estimate,
            state_at_main_loop_start=self.state_at_main_loop_start,
            **kw,
        )

    # ------------------------------------------------------------------ pid 锁（27.9）
    def acquire_lock(self) -> bool:
        pf = self.pid_file
        if pf.exists():
            pid = _read_pid(pf)
            if pid is not None and self.pid_alive_fn(pid):
                self.emit(f"已有运行中的压测（pid {pid}，{pf}）；拒绝启动新的 run")
                return False
            log.info("stale pid file %s (pid=%s); overwriting", pf, pid)
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(str(os.getpid()), encoding="utf-8")
        self._lock_held = True
        try:  # 上次运行遗留的 STOP 文件不应立即停止本次运行
            self.stop_file.unlink(missing_ok=True)
        except OSError:
            log.warning("cannot remove stale %s", self.stop_file)
        return True

    def release_lock(self) -> None:
        if not self._lock_held:
            return
        self._lock_held = False
        try:
            if _read_pid(self.pid_file) == os.getpid():
                self.pid_file.unlink(missing_ok=True)
        except OSError:
            log.exception("cannot remove pid file %s", self.pid_file)
        try:
            self.stop_file.unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------ 停止控制
    def request_stop(self, reason: str) -> None:
        """开始优雅停止（只记录第一个原因）。"""
        stop = self._stop
        if stop is None or stop.is_set():
            return
        self.external_reason = reason
        self.stop_requested_at = self.clock.now()
        stop.set()

    def on_signal(self, signum: int) -> None:
        """SIGINT / SIGTERM 处理（在事件循环线程中调用）。"""
        self.signals_received += 1
        stop = self._stop
        if stop is None:
            return
        if stop.is_set():
            # 27.6：优雅停止期间再次 SIGINT → 立即结算退出
            if signum == signal.SIGINT and not self.forced:
                self.force_exit()
            return
        self.request_stop("stop_cmd" if self.stop_file.exists() else "signal")

    def force_exit(self) -> None:
        """不再等待在途请求：立即取消（worker 按预留结算），随后刷盘退出。"""
        if self.forced:
            return
        self.forced = True
        if self._stop is not None:
            self._stop.set()
        sched = self.scheduler
        if sched is not None:
            sched.drained = False
            self._force_task = asyncio.ensure_future(sched.cancel_inflight())

    def _install_signal_handlers(self) -> None:
        if not self.install_signals:
            return
        loop = asyncio.get_running_loop()
        sigs = [signal.SIGINT, signal.SIGTERM]
        for sig in sigs:
            try:
                loop.add_signal_handler(sig, self.on_signal, sig)
                self._loop_handlers.append(sig)
                continue
            except (NotImplementedError, RuntimeError, ValueError):
                pass
            try:  # Windows：signal.signal + call_soon_threadsafe 唤醒事件循环
                self._prev_handlers[sig] = signal.signal(
                    sig, lambda s, _f: loop.call_soon_threadsafe(self.on_signal, s)
                )
            except (ValueError, OSError):  # 非主线程
                log.debug("cannot install handler for %s", sig)

    def _remove_signal_handlers(self) -> None:
        if self._loop_handlers:
            loop = asyncio.get_running_loop()
            for sig in self._loop_handlers:
                try:
                    loop.remove_signal_handler(sig)
                except Exception:  # noqa: BLE001
                    pass
            self._loop_handlers.clear()
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError, TypeError):
                pass
        self._prev_handlers.clear()

    # ------------------------------------------------------------------ 入口
    async def run(self) -> RunResult:
        if not self.acquire_lock():
            return self._result(EXIT_LOCKED, None)
        try:
            return await self._run_locked()
        finally:
            self.release_lock()

    async def _run_locked(self) -> RunResult:
        errs = check_limits(self.settings, self.tpm_override)
        if errs:
            for e in errs:
                self.emit(e)
            return self._result(EXIT_LIMITS, None)

        if self._store is None:
            from loadtest.corpus.store import CorpusStore

            self._store = await CorpusStore(self.settings.db_path, self.clock).open()
        self.store = self._store
        client: Any = None
        try:
            await self.store.create_run(self.run_id, "load", config_snapshot_json(self.settings))
            self._run_created = True

            # ---- 4. 供应检查 ----
            stats = await self.store.stats()
            sc = check_for_run(stats, self.duration_s, self.settings, tpm=self.tpm_override)
            self.supply = sc
            if sc.verdict == "refuse":
                if not self.opts.allow_low_supply:
                    self.emit("供应检查未通过（refuse）：")
                    for m in sc.messages:
                        self.emit(f"  - {m}")
                    if sc.suggested_command:
                        self.emit(f"建议命令：{sc.suggested_command}")
                    return await self._finish_prestart(EXIT_SUPPLY, "supply_refused")
                self.emit("警告：供应检查 refuse，但指定了 --allow-low-supply；运行可能以 corpus_exhausted 提前结束")
                for m in sc.messages:
                    self.emit(f"  - {m}")
            elif sc.verdict == "warn":
                self.emit("供应检查告警：")
                for m in sc.messages:
                    self.emit(f"  - {m}")
                if sc.suggested_command:
                    self.emit(f"建议命令：{sc.suggested_command}")

            # ---- 5. 成本预估 ----
            guard = self._cost_guard or await CostGuard.azure_from_settings(self.store, self.settings.cost)
            self.guard = guard
            code = self._cost_estimate(guard)
            if code:
                return await self._finish_prestart(code, "budget")

            self.throttle = ThrottleController(self.settings.throttle, self.clock)

            # ---- 6. 启动探测 ----
            if self.opts.dry_run:
                client = DryRunClient(self.clock, Random(self.seed ^ 0xD27))
                sched_guard = CostGuard(math.inf, 0.0, self.settings.cost.prices)  # 不写 Azure 账本
                self.emit("--dry-run：不发送 Probe_Request，不调用 Azure，不写 Azure 账本")
            else:
                client = self.counting_client = CountingClient(self.client_factory(self.settings))
                code, reason = await self._startup_probe(client, guard, self.throttle)
                if code:
                    return await self._finish_prestart(code, reason)
                sched_guard = guard

            # ---- 7. 主循环 ----
            return await self._main(client, sched_guard)
        except Exception:
            log.exception("run failed")
            self.emit("运行异常终止（stop_reason=error）")
            return await self._finish_prestart(EXIT_ERROR, "error")
        finally:
            if client is not None:
                close = getattr(client, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:  # noqa: BLE001
                        log.exception("client close failed")
            if self._owns_store and self.store is not None:
                try:
                    await self.store.close()
                except Exception:  # noqa: BLE001
                    log.exception("store close failed")

    async def _finish_prestart(self, code: int, reason: str) -> RunResult:
        """启动前失败：刷写已产生的账本（如启动探测花费）并写 runs 记录。"""
        if self.guard is not None and not self.opts.dry_run:
            try:
                await self.guard.flush(self.store, self.run_id)
            except Exception:  # noqa: BLE001
                log.exception("ledger flush failed")
        if self._run_created:
            try:
                await self.store.finish_run(
                    self.run_id,
                    reason,
                    throttle_stats=self.throttle.stats if self.throttle is not None else None,
                    probes=0,
                )
            except Exception:  # noqa: BLE001
                log.exception("finish_run failed")
        return self._result(code, reason)

    def _cost_estimate(self, guard: CostGuard) -> int:
        load, az = self.settings.load, self.settings.azure
        tpm = self.tpm_override if self.tpm_override is not None else float(load.target_tpm)
        rpm = min(float(load.rpm_max), az.quota_rpm * 0.5, tpm / AVG_INPUT_TOKENS)
        minutes = self.duration_s / 60.0
        est = guard.estimate_run_cost(tpm, rpm, minutes, self.model)
        self.cost_estimate = est
        self.emit(
            f"本次运行预估花费 ${est:.2f}（TPM {tpm:,.0f}、RPM {rpm:,.0f}、{minutes:.1f} 分钟）；"
            f"Azure 已花费 ${guard.spent:.2f} / 上限 ${guard.cap:.2f}"
        )
        over = guard.spent + est > guard.cap
        if guard.per_run_cap is not None and est > guard.per_run_cap:
            over = True
        if over and not self.opts.dry_run:
            if not self.opts.allow_partial:
                self.emit("已花费 + 预估花费超过上限；拒绝启动（可用 --allow-partial 在预算触顶时提前结束）")
                return EXIT_COST
            self.emit("警告：预估花费超过上限，--allow-partial：预算触顶时将以 stop_reason=budget 提前结束")
        return 0

    def _probe_request(self, attempt: int) -> PlannedRequest:
        return PlannedRequest(
            request_id=f"startup-probe-{self.run_id}-{attempt}",
            lang="en",
            messages=[dict(m) for m in PROBE_MESSAGES],
            input_tokens=PROBE_INPUT_TOKENS,
            max_tokens=PROBE_MAX_TOKENS,
            passage_ids=(),
            task_id="startup_probe",
            target_tokens=PROBE_INPUT_TOKENS,
        )

    async def _startup_probe(self, client: Any, guard: CostGuard, throttle: ThrottleController) -> tuple[int, str]:
        """22.4 / 22.6–22.9。返回 (退出码, stop_reason)；成功时退出码为 0。"""
        load = self.settings.load
        attempts = 1 + int(load.probe_max_retries)
        for attempt in range(attempts):
            if attempt:
                await self.clock.sleep(load.probe_retry_interval_s)
            planned = self._probe_request(attempt)
            r = await guard.reserve(self.model, planned.input_tokens, planned.max_tokens)
            if r is None:
                self.emit("Azure 预算不足以发送启动探测请求；拒绝启动")
                return EXIT_COST, "budget"
            try:
                res = await client.send(planned)
            except asyncio.CancelledError:
                await guard.settle(r, None)
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("startup probe raised %s", type(e).__name__)
                res = AzureResult(planned.request_id, Outcome.SERVER_ERROR, None, 0.0, error_code=type(e).__name__)
            outcome = _outcome(res.outcome)
            self.startup_probe_outcomes.append(outcome.value)
            if outcome is Outcome.THROTTLED:
                await guard.settle_zero(r)
            else:
                await guard.settle(r, _usage_of(res))
            status = getattr(res, "status", None)
            if status in (401, 404):
                self.emit(
                    f"启动探测返回 {status}：请检查 AZURE_OPENAI_ENDPOINT / API Key / deployment 配置"
                )
                return EXIT_PROBE, "error"
            if outcome is Outcome.THROTTLED:
                ra = getattr(res, "retry_after_s", None)
                throttle.on_throttle(ra, self.clock.now())  # 22.8：主循环从 HOLD → PROBE 开始
                self.emit(f"启动探测被节流（retry-after={ra}）；进入 HOLD 后开始压测")
                return 0, ""
            if outcome in (Outcome.SERVER_ERROR, Outcome.TIMEOUT):
                self.emit(f"启动探测失败（{outcome.value}），第 {attempt + 1}/{attempts} 次")
                continue
            if outcome is not Outcome.OK:
                self.emit(f"启动探测结果 {outcome.value}（status={status}）；继续启动")
            return 0, ""
        self.emit(f"启动探测 {attempts} 次均失败（SERVER_ERROR / TIMEOUT）；终止")
        return EXIT_PROBE, "error"

    # ------------------------------------------------------------------ 主循环
    async def _main(self, client: Any, guard: CostGuard) -> RunResult:
        s = self.settings
        stop = asyncio.Event()
        self._stop = stop
        throttle = self.throttle
        assert throttle is not None

        passages = await self.store.load_active_passages()
        index = PassageIndex(
            passages,
            per_run_max_uses=s.corpus.per_run_max_uses,
            alpha=s.corpus.use_weight_alpha,
            buckets=s.corpus.length_buckets,
            clock=self.clock,
        )
        self.index = index
        assembler = self.assembler_factory(index, s)
        breaker = CircuitBreaker()
        metrics = MetricsCollector(self.clock, throttle=throttle)
        self.metrics = metrics
        self.run_dir.mkdir(parents=True, exist_ok=True)
        jsonl = JsonlWriter(self.run_dir / "requests.jsonl")
        csvw = CsvWriter(self.run_dir / "metrics.csv")
        console = None
        if self.show_console:
            from loadtest.metrics.console import ConsoleReporter

            console = ConsoleReporter()
        queue: asyncio.Queue = asyncio.Queue(maxsize=s.gemini.replenish_queue_size)

        closer: Callable[[], Awaitable[None]] | None = None
        if not self.opts.no_replenish and not self.opts.dry_run:
            try:
                self.replenisher, closer = await self.replenisher_factory(
                    s, self.clock, run_id=self.run_id, rng=Random(self.seed ^ 0x5A5A), queue=queue
                )
            except Exception:  # noqa: BLE001 —— 补充任务不可用不影响压测（10.4）
                log.exception("replenisher setup failed")
                self.emit("警告：后台补充任务初始化失败，本次运行不补充语料")
                self.replenisher, closer = None, None

        sched = Scheduler(
            s.load,
            assembler=assembler,
            cost_guard=guard,
            throttle=throttle,
            breaker=breaker,
            client=client,
            index=index,
            clock=self.clock,
            rng=self.rng,
            model=self.model,
            quota_rpm=s.azure.quota_rpm,
            metrics=metrics,
            store=self.store,
            run_id=self.run_id,
            jsonl=jsonl,
            replenish_queue=queue,
            stop_event=stop,
            tpm_override=self.tpm_override,
            ramp_enabled=not self.opts.no_ramp,
            max_breaker_opens=self.max_breaker_opens,
        )
        self.scheduler = sched

        self._install_signal_handlers()
        rep_task = asyncio.ensure_future(self.replenisher.run(stop, queue)) if self.replenisher else None
        flusher = asyncio.ensure_future(self._flush_loop(stop, guard, index))
        side = [
            asyncio.ensure_future(self._report_loop(stop, metrics, sched, index, guard, console, csvw, jsonl)),
            asyncio.ensure_future(self._watch_loop(stop, sched, breaker)),
            asyncio.ensure_future(self._duration_timer(stop)),
        ]
        self.state_at_main_loop_start = throttle.state
        summary: RunSummary | None = None
        try:
            summary = await sched.run()
        except Exception:  # noqa: BLE001
            log.exception("scheduler failed")
            self.fatal_error = True
            stop.set()
            try:
                await sched.cancel_inflight()
            except Exception:  # noqa: BLE001
                log.exception("cancel_inflight failed")
            summary = sched.summary()
        finally:
            stop.set()
            self._remove_signal_handlers()
            for t in side:
                t.cancel()
            await asyncio.gather(*side, return_exceptions=True)
            if self._force_task is not None:
                await asyncio.gather(self._force_task, return_exceptions=True)
            await asyncio.gather(flusher, return_exceptions=True)  # 不打断进行中的刷写
            await self._stop_replenisher(rep_task, closer)

        # ---- 收尾：use_count（9.4）、账本（26.4 / 26.5）----
        if not self.opts.dry_run:
            try:
                await self._flush_uses(index)
            except Exception:  # noqa: BLE001
                log.exception("use_count flush failed")
                self.emit("警告：use_count 刷写失败")
            try:
                async with self._flush_lock:
                    await guard.flush(self.store, self.run_id)
            except LedgerWriteError:
                log.exception("final ledger flush failed")
                self.ledger_failed = True
        jsonl.close()

        stop_reason = self._final_reason(sched)
        if self.ledger_failed:
            self.emit("账本写入失败（已重试 3 次）；stop_reason=error")

        report = await self._finish(stop_reason, metrics, index, guard, throttle, summary)
        code = _EXIT_BY_REASON.get(stop_reason, EXIT_OK)
        return self._result(code, stop_reason, summary=summary, report=report, run_dir=self.run_dir)

    def _final_reason(self, sched: Scheduler) -> str:
        if self.ledger_failed or self.fatal_error:
            return "error"
        if self.external_reason is not None:
            return self.external_reason
        if sched.stop_reason is not None:
            return sched.stop_reason
        return "error"

    async def _stop_replenisher(self, rep_task: asyncio.Future | None, closer: Any) -> None:
        if rep_task is not None:
            if not rep_task.done() and not self.forced and self.replenisher_stop_timeout_s > 0:
                sleeper = asyncio.ensure_future(self.clock.sleep(self.replenisher_stop_timeout_s))
                try:
                    await asyncio.wait({rep_task, sleeper}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    if not sleeper.done():
                        sleeper.cancel()
            if not rep_task.done():
                rep_task.cancel()
            await asyncio.gather(rep_task, return_exceptions=True)
        if closer is not None:
            try:
                await closer()  # 刷写 Gemini 账本并关闭补充任务连接
            except Exception:  # noqa: BLE001
                log.exception("replenisher closer failed")

    async def _finish(self, stop_reason: str, metrics: MetricsCollector, index: PassageIndex,
                      guard: CostGuard, throttle: ThrottleController, summary: RunSummary | None) -> dict | None:
        ok = metrics.outcomes.get(Outcome.OK.value, 0) + metrics.probe_outcomes.get(Outcome.OK.value, 0)
        throttled = metrics.throttled_total
        err = max(0, metrics.completed - ok - throttled)
        added = int(getattr(self.replenisher, "added", 0) or 0)
        try:
            await self.store.finish_run(
                self.run_id,
                stop_reason,
                throttle_stats=throttle.stats,
                requests_ok=ok,
                requests_err=err,
                throttled=throttled,
                probes=probes_count(throttle.stats),
                input_tokens=metrics.input_tokens_total,
                output_tokens=metrics.completion_tokens_total,
                passages_added=added,
            )
        except Exception:  # noqa: BLE001
            log.exception("finish_run failed")
        cost = {
            "azure_run_usd": guard.run_spent,
            "azure_total_usd": guard.spent,
            "azure_cap_usd": guard.cap,
            "dry_run": self.opts.dry_run,
        }
        report: dict | None = None
        try:
            report, _ = write_report(
                self.run_dir,
                metrics.summary(
                    stop_reason=stop_reason,
                    run_id=self.run_id,
                    run_uses=index.run_uses,
                    passages_added=added,
                    cost=cost,
                    startup_probe=list(self.startup_probe_outcomes),
                    drained=summary.drained if summary is not None else None,
                    cancelled=summary.cancelled if summary is not None else None,
                ),
            )
        except Exception:  # noqa: BLE001
            log.exception("report generation failed")
        self.emit(f"运行结束：stop_reason={stop_reason}，报告目录 {self.run_dir}")
        if report is not None:
            self._emit_report_summary(report)
        return report

    def _emit_report_summary(self, r: dict) -> None:
        tpm, rpm = r.get("avg_tpm"), r.get("avg_rpm")
        self.emit(
            f"派发 {r['requests']['dispatched']}，平均 TPM {tpm:,.0f}，平均 RPM {rpm:,.1f}"
            if tpm is not None and rpm is not None
            else f"派发 {r['requests']['dispatched']}"
        )
        if self.opts.dry_run:  # 27.7
            steps = ", ".join(f"{t:g}s→{lv:.0%}" for t, lv in self.settings.load.startup_ramp)
            self.emit(f"[dry-run] 爬坡曲线：{'关闭（100%）' if self.opts.no_ramp else steps}")
            self.emit(f"[dry-run] 语言配比：{r.get('lang_share_pct')}")
            self.emit(f"[dry-run] 复用分布：{r.get('reuse_distribution')}")
            tok = r.get("tokens", {})
            self.emit(f"[dry-run] 输入长度 P50 {tok.get('input_p50')} / P95 {tok.get('input_p95')}")
        for w in r.get("warnings", []) or []:
            self.emit(f"提示：{w}")

    # ------------------------------------------------------------------ 后台任务
    async def _wait(self, event: asyncio.Event, timeout: float) -> None:
        if event.is_set():
            return
        sleeper = asyncio.ensure_future(self.clock.sleep(max(0.0, timeout)))
        waiter = asyncio.ensure_future(event.wait())
        try:
            await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for f in (sleeper, waiter):
                if not f.done():
                    f.cancel()

    async def _duration_timer(self, stop: asyncio.Event) -> None:
        await self._wait(stop, self.duration_s)
        if not stop.is_set():
            self.request_stop("duration")

    async def _watch_loop(self, stop: asyncio.Event, sched: Scheduler, breaker: CircuitBreaker) -> None:
        """runs/STOP 文件（stop_cmd）与熔断第 3 次 open（circuit_open，23.5）。"""
        while not stop.is_set():
            try:
                if self.stop_file.exists():
                    self.request_stop("stop_cmd")
                    return
            except OSError:
                pass
            if breaker.open_count >= self.max_breaker_opens:
                sched.stop_reason = sched.stop_reason or "circuit_open"
                stop.set()
                return
            await self._wait(stop, self.stop_poll_s)

    async def _flush_uses(self, index: PassageIndex) -> None:
        """把 run_uses 相对上次刷写的增量累加到 use_count（总增量恰好 = run_uses，9.4）。"""
        current = index.run_uses
        delta = {pid: n - self._flushed_uses.get(pid, 0) for pid, n in current.items()}
        delta = {pid: d for pid, d in delta.items() if d > 0}
        if not delta:
            return
        await self.store.flush_use_counts(delta)
        for pid, d in delta.items():
            self._flushed_uses[pid] = self._flushed_uses.get(pid, 0) + d

    async def _flush_loop(self, stop: asyncio.Event, guard: CostGuard, index: PassageIndex) -> None:
        """每 flush_interval_s 刷写账本与 use_count（26.4）；账本写失败 → stop_reason=error（26.5）。"""
        if self.opts.dry_run:
            return
        while not stop.is_set():
            await self._wait(stop, self.flush_interval_s)
            if stop.is_set():
                return
            try:
                async with self._flush_lock:
                    await guard.flush(self.store, self.run_id)
            except LedgerWriteError:
                log.exception("ledger flush failed")
                self.ledger_failed = True
                self.request_stop("error")
                return
            try:
                await self._flush_uses(index)
            except Exception:  # noqa: BLE001
                log.exception("use_count flush failed")

    def _snapshot_extra(self, sched: Scheduler, index: PassageIndex, guard: CostGuard) -> dict[str, Any]:
        extra = sched.status_extra()
        extra.update(
            {
                "pool_active": len(index),
                "pool_added": int(getattr(self.replenisher, "added", 0) or 0),
                "max_run_uses": max_run_uses(index.run_uses),
                "cost_usd": guard.run_spent,
            }
        )
        return extra

    async def _report_loop(self, stop: asyncio.Event, metrics: MetricsCollector, sched: Scheduler,
                           index: PassageIndex, guard: CostGuard, console: Any, csvw: CsvWriter,
                           jsonl: JsonlWriter) -> None:
        """每 report_interval_s 输出控制台、写 CSV 行并刷写 JSONL（28.1–28.3）。"""
        while not stop.is_set():
            await self._wait(stop, self.report_interval_s)
            if stop.is_set():
                return
            try:
                snap = metrics.snapshot(extra=self._snapshot_extra(sched, index, guard))
                if console is not None:
                    console.print(snap)
                csvw.write_row(snap)
                jsonl.flush()
            except Exception:  # noqa: BLE001
                log.exception("metrics output failed")


# ---------------------------------------------------------------------------
# 便捷入口
# ---------------------------------------------------------------------------


async def run_load(settings: Settings, options: RunOptions | None = None, **kw: Any) -> RunResult:
    return await RunController(settings, options, **kw).run()


async def run_from_config(
    config_path: str | Path = "config.yaml",
    env_file: str | Path | None = ".env",
    options: RunOptions | None = None,
    **kw: Any,
) -> RunResult:
    """启动顺序第 1 步：加载并校验配置；失败 → EXIT_CONFIG，不发起任何外部调用。"""
    try:
        settings = load_settings(config_path, env_file)
    except SettingsError as e:
        out = kw.get("out") or print
        msg = e.format()
        out(msg)
        return RunResult(exit_code=EXIT_CONFIG, stop_reason=None, run_id=None, messages=[msg])
    return await run_load(settings, options, **kw)


__all__ = [
    "EXIT_CIRCUIT",
    "EXIT_CONFIG",
    "EXIT_COST",
    "EXIT_ERROR",
    "EXIT_LIMITS",
    "EXIT_LOCKED",
    "EXIT_OK",
    "EXIT_PROBE",
    "EXIT_SUPPLY",
    "CountingClient",
    "DryRunClient",
    "RunController",
    "RunOptions",
    "RunResult",
    "apply_overrides",
    "check_limits",
    "pid_alive",
    "run_from_config",
    "run_load",
]
