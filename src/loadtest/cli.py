"""loadtest 命令行（typer，Requirement 29）。

命令：warmup / generate / run / stop / status / budget / report / pool-stats。
不监听任何网络端口（29.6）。非法选项值 → 输出选项名并以非零退出码退出（29.7）。
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import signal
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import typer

from loadtest.settings import EXIT_CONFIG_ERROR, RedactingFilter, Settings, SettingsError, load_settings

app = typer.Typer(help="Azure OpenAI GPT load tester", no_args_is_help=True, add_completion=False)

EXIT_BAD_OPTION = 2
_STATE: dict[str, Any] = {"config": "config.yaml", "env_file": ".env"}

# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------


def _echo(msg: str = "") -> None:
    typer.echo(msg)


def _fail(msg: str, code: int = 1) -> None:
    typer.echo(msg, err=True)
    raise typer.Exit(code)


def _bad_option(name: str, reason: str) -> None:
    _fail(f"Invalid value for option {name}: {reason}", EXIT_BAD_OPTION)


def _settings() -> Settings:
    try:
        return load_settings(_STATE["config"], _STATE["env_file"])
    except SettingsError as e:
        _fail(e.format(), EXIT_CONFIG_ERROR)
        raise  # pragma: no cover


def _setup_logging(level: str) -> None:
    lvl = getattr(logging, level.upper(), None)
    if not isinstance(lvl, int):
        _bad_option("--log-level", f"unknown level {level!r}")
    root = logging.getLogger()
    if not any(isinstance(f, RedactingFilter) for h in root.handlers for f in h.filters):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.addFilter(RedactingFilter())
        root.addHandler(handler)
    root.setLevel(lvl)
    # SDK 的 DEBUG 日志可能包含请求头 / 消息内容：始终 ≥ INFO
    for name in ("openai", "httpx", "httpcore", "google", "urllib3"):
        logging.getLogger(name).setLevel(max(lvl, logging.INFO))


_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*$", re.IGNORECASE)


def parse_duration(text: str) -> float:
    """'20m' / '90s' / '1h' / '300' → 秒；非法或 ≤ 0 抛 ValueError。"""
    m = _DUR_RE.match(text or "")
    if not m:
        raise ValueError(f"cannot parse duration {text!r}")
    v = float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600}[m.group(2).lower()]
    if v <= 0:
        raise ValueError("duration must be > 0")
    return v


def _positive(name: str, v: Optional[int]) -> None:
    if v is not None and v <= 0:
        _bad_option(name, "must be a positive integer")


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _live_pid(settings: Settings) -> int | None:
    from loadtest.load.runner import pid_alive

    pid = _read_pid(Path(settings.load.pid_file))
    return pid if pid is not None and pid_alive(pid) else None


async def _open_store(settings: Settings):
    from loadtest.clock import RealClock
    from loadtest.corpus.store import CorpusStore

    return await CorpusStore(settings.db_path, RealClock()).open()


def _category_ids(settings: Settings) -> set[str]:
    from loadtest.corpus.taxonomy import TopicTaxonomy

    return {c.id for c in TopicTaxonomy.load(grounding_categories=settings.gemini.grounding_categories).categories()}


def _print_inventory(stats: Any) -> None:
    _echo(f"active={stats.active}  quarantined={stats.quarantined}  expired={stats.expired}")
    _echo("按语言: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.by_lang.items())))
    _echo("按长度桶: " + ", ".join(f"{l}/{b}={v}" for (l, b), v in sorted(stats.by_bucket.items())))


# ---------------------------------------------------------------------------
# 全局选项
# ---------------------------------------------------------------------------


@app.callback()
def main(
    config: Path = typer.Option(Path("config.yaml"), "--config", help="配置文件路径"),
    env_file: Path = typer.Option(Path(".env"), "--env-file", help=".env 路径"),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
) -> None:
    """loadtest 命令行。"""
    _STATE["config"] = config
    _STATE["env_file"] = env_file
    _setup_logging(log_level)


# ---------------------------------------------------------------------------
# warmup / generate
# ---------------------------------------------------------------------------


async def _corpus_job(settings: Settings, kind: str, **kw: Any) -> Any:
    from loadtest.clock import RealClock
    from loadtest.corpus.warmup import build_corpus_pipeline

    clock = RealClock()
    store = await _open_store(settings)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, stop.set)
    except (NotImplementedError, RuntimeError):  # Windows
        pass
    try:
        runner = await build_corpus_pipeline(
            settings, store, clock, run_id=f"{kind}-{uuid.uuid4().hex[:12]}", kind=kind, stop_event=stop, **kw
        )
        try:
            return await runner.run()
        except KeyboardInterrupt:  # pragma: no cover
            stop.set()
            raise
    finally:
        await store.close()


def _print_corpus_report(rep: Any) -> None:
    _echo(f"[{rep.kind}] run_id={rep.run_id} stop_reason={rep.stop_reason}")
    _echo(f"入库 {rep.inserted}，拒绝 {rep.rejected}（拒绝率 {rep.rejection_rate:.1%}），错误 {rep.errors}，"
          f"Gemini 调用 {rep.calls}，花费 ${rep.cost_usd:.4f}")
    if rep.rejected_by_reason:
        _echo("拒绝原因: " + ", ".join(f"{k}={v}" for k, v in sorted(rep.rejected_by_reason.items())))
    _print_inventory(rep.inventory)
    _echo("按类别: " + ", ".join(f"{l}/{c}={v}" for (l, c), v in sorted(rep.inventory.by_category.items())))


@app.command()
def warmup(
    target: int = typer.Option(8000, "--target", help="语料池目标规模"),
    concurrency: int = typer.Option(32, "--concurrency", help="并发数"),
    # 8,000 段 × gemini-3.1-flash-lite ≈ $14.3，留余量
    max_cost: float = typer.Option(20.0, "--max-cost", help="本次最大花费（USD）"),
) -> None:
    """预热语料池：按缺口生成，直到填满 / 预算 / --max-cost。"""
    _positive("--target", target)
    _positive("--concurrency", concurrency)
    if max_cost <= 0:
        _bad_option("--max-cost", "must be > 0")
    settings = _settings()
    rep = asyncio.run(_corpus_job(settings, "warmup", target=target, concurrency=concurrency, max_cost=max_cost))
    _print_corpus_report(rep)


@app.command()
def generate(
    count: int = typer.Option(..., "--count", help="本次最多入库段落数"),
    category: Optional[str] = typer.Option(None, "--category", help="类别 id，逗号分隔"),
    lang: Optional[str] = typer.Option(None, "--lang", help="zh 或 en"),
    max_cost: float = typer.Option(10.0, "--max-cost", help="本次最大花费（USD）"),
) -> None:
    """定向补充语料（例如每日 grounding 刷新）。"""
    _positive("--count", count)
    if max_cost <= 0:
        _bad_option("--max-cost", "must be > 0")
    if lang is not None and lang not in ("zh", "en"):
        _bad_option("--lang", "must be zh or en")
    settings = _settings()
    cats = [c.strip() for c in category.split(",") if c.strip()] if category else None
    if cats:
        unknown = sorted(set(cats) - _category_ids(settings))
        if unknown:
            _bad_option("--category", f"unknown categories: {', '.join(unknown)}")
    rep = asyncio.run(_corpus_job(
        settings, "generate", categories=cats, langs=[lang] if lang else None, count=count, max_cost=max_cost
    ))
    _print_corpus_report(rep)


# ---------------------------------------------------------------------------
# run / stop / status
# ---------------------------------------------------------------------------


@app.command()
def run(
    duration: str = typer.Option("20m", "--duration", help="运行时长，如 20m / 90s / 1h"),
    tpm: Optional[int] = typer.Option(None, "--tpm", help="目标 TPM"),
    rpm_max: Optional[int] = typer.Option(None, "--rpm-max", help="RPM 上限"),
    deployment: Optional[str] = typer.Option(None, "--deployment", help="Azure 部署名"),
    no_replenish: bool = typer.Option(False, "--no-replenish", help="不启动后台补充"),
    no_ramp: bool = typer.Option(False, "--no-ramp", help="关闭启动爬坡"),
    allow_low_supply: bool = typer.Option(False, "--allow-low-supply", help="供应不足仍启动"),
    allow_partial: bool = typer.Option(False, "--allow-partial", help="预估超预算仍启动"),
    dry_run: bool = typer.Option(False, "--dry-run", help="不调用 Azure，只模拟"),
    seed: Optional[int] = typer.Option(None, "--seed", help="随机种子"),
) -> None:
    """运行压测。"""
    try:
        duration_s = parse_duration(duration)
    except ValueError as e:
        _bad_option("--duration", str(e))
    _positive("--tpm", tpm)
    _positive("--rpm-max", rpm_max)
    from loadtest.load.runner import RunController, RunOptions

    settings = _settings()
    opts = RunOptions(
        duration_s=duration_s, tpm=tpm, rpm_max=rpm_max, deployment=deployment,
        no_replenish=no_replenish, no_ramp=no_ramp, allow_low_supply=allow_low_supply,
        allow_partial=allow_partial, dry_run=dry_run, seed=seed,
    )
    res = asyncio.run(RunController(settings, opts).run())
    _echo(f"run_id={res.run_id} stop_reason={res.stop_reason} exit={res.exit_code}")
    raise typer.Exit(res.exit_code)


@app.command()
def stop() -> None:
    """优雅停止运行中的压测（写 runs/STOP 并发送 SIGTERM）。"""
    settings = _settings()
    pid = _live_pid(settings)
    if pid is None:
        _fail("没有运行中的压测", 1)
    stop_file = Path(settings.load.pid_file).parent / "STOP"
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    if os.name == "nt":
        _echo(f"已写入 {stop_file}；压测进程（pid {pid}）将在 1 秒内开始优雅停止")
        return
    os.kill(pid, signal.SIGTERM)
    _echo(f"已写入 {stop_file} 并向 pid {pid} 发送 SIGTERM")


async def _latest_load_run(settings: Settings) -> Any:
    store = await _open_store(settings)
    try:
        return await store.latest_run("load")
    finally:
        await store.close()


@app.command()
def status() -> None:
    """显示是否有运行中的压测。"""
    settings = _settings()
    pid = _live_pid(settings)
    if pid is None:
        _echo("没有运行中的压测")
        return
    run_rec = asyncio.run(_latest_load_run(settings))
    if run_rec is not None and run_rec.ended_at is None:
        started = datetime.fromisoformat(run_rec.started_at)
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
        _echo(f"运行中：pid={pid} run_id={run_rec.run_id} 已运行 {elapsed // 60}m{elapsed % 60:02d}s")
    else:
        _echo(f"运行中：pid={pid}（尚未写入 run 记录）")


# ---------------------------------------------------------------------------
# budget / report / pool-stats
# ---------------------------------------------------------------------------


async def _budget(settings: Settings) -> list[str]:
    from loadtest.cost.guard import CostGuard

    store = await _open_store(settings)
    try:
        az = await CostGuard.azure_from_settings(store, settings.cost)
        gm = await CostGuard.gemini_from_settings(store, settings.cost, settings.gemini.project)
        gm_accounts = await store.ledger_accounts(["gemini", "gemini_grounding"])
    finally:
        await store.close()
    load = settings.load
    tpm = float(load.target_tpm)
    rpm = min(float(load.rpm_max), tpm / 3040)
    est = az.estimate_run_cost(tpm, rpm, 20.0, settings.azure.model_family)
    lines = [f"Azure: 已用 ${az.spent:.4f} / 上限 ${az.cap:.2f}（安全余量 ${az.margin:.2f}）"]
    for (prov, model), v in sorted(az.spent_by_model.items()):
        lines.append(f"  {prov}/{model}: ${v:.4f}")
    lines.append(f"  单次 20 分钟运行预估: ${est:.2f}（TPM {tpm:,.0f}、RPM {rpm:,.0f}，{settings.azure.model_family}）")
    lines.append(f"  剩余可运行次数: {az.remaining_runs(est)}")
    lines.append(f"Gemini（当前账号 {gm.account or '-'}）: 已用 ${gm.spent:.4f} / 上限 ${gm.cap:.2f}")
    for (prov, model), v in sorted(gm.spent_by_model.items()):
        lines.append(f"  {prov}/{model}: ${v:.4f}")
    others = [(a, v) for a, v in gm_accounts if a != gm.account]
    if others:
        lines.append(f"  其他账号累计: ${sum(v for _, v in others):.4f}（{len(others)} 个，不计入当前上限）")
        for a, v in others:
            lines.append(f"    {a or '(未标记)'}: ${v:.4f}")
    return lines


@app.command()
def budget() -> None:
    """显示 Azure / Gemini 分模型花费、上限、预估与剩余可运行次数。"""
    for line in asyncio.run(_budget(_settings())):
        _echo(line)


async def _get_run(settings: Settings, run_id: Optional[str]) -> Any:
    store = await _open_store(settings)
    try:
        return await (store.get_run(run_id) if run_id else store.latest_run("load"))
    finally:
        await store.close()


@app.command()
def report(run_id: Optional[str] = typer.Argument(None, help="运行 id，缺省为最近一次")) -> None:
    """输出指定运行（缺省最近一次）的报告。"""
    settings = _settings()
    rec = asyncio.run(_get_run(settings, run_id))
    if rec is None:
        _fail(f"找不到运行 {run_id}" if run_id else "没有任何压测运行记录", 1)
    md = Path(settings.log_dir) / rec.run_id / "report.md"
    if md.exists():
        _echo(md.read_text(encoding="utf-8"))
        return
    _echo(f"run_id={rec.run_id} kind={rec.kind} started={rec.started_at} ended={rec.ended_at}")
    _echo(f"stop_reason={rec.stop_reason} ok={rec.requests_ok} err={rec.requests_err} "
          f"throttled={rec.throttled} probes={rec.probes}")
    _echo(f"input_tokens={rec.input_tokens} output_tokens={rec.output_tokens} "
          f"passages_added={rec.passages_added} cost=${rec.cost_usd:.4f}")


async def _pool_stats(settings: Settings) -> list[str]:
    from loadtest.corpus.planner import check_for_run

    store = await _open_store(settings)
    try:
        stats = await store.stats()
        since = (datetime.now(timezone.utc) - timedelta(days=7)).timestamp()
        rej = await store.rejection_counts(since)
        gem = await store.ledger_totals(["gemini", "gemini_grounding"])
    finally:
        await store.close()
    lines = [
        f"active={stats.active}  quarantined={stats.quarantined}  expired={stats.expired}",
        "按语言: " + (", ".join(f"{k}={v}" for k, v in sorted(stats.by_lang.items())) or "-"),
        "按长度桶: " + (", ".join(f"{l}/{b}={v}" for (l, b), v in sorted(stats.by_bucket.items())) or "-"),
        "按 use_count: " + (", ".join(f"{k}={v}" for k, v in sorted(stats.by_use_count.items())) or "-"),
        "按类别:",
    ]
    for (l, c), v in sorted(stats.by_category.items()):
        lines.append(f"  {l}/{c}: {v}")
    total_rej = sum(rej.values())
    lines.append("近 7 天拒绝原因: " + (
        ", ".join(f"{k}={v}（{v / total_rej:.1%}）" for k, v in sorted(rej.items())) if total_rej else "-"
    ))
    lines.append("Gemini 花费: " + (", ".join(f"{t.provider}/{t.model}=${t.total_usd:.4f}" for t in gem) or "-"))
    sc = check_for_run(stats, 1200, settings)
    lines.append(f"下一次 20 分钟运行供应检查: {sc.verdict}")
    for lang in sorted(sc.required):
        lines.append(f"  {lang}: 需要 {sc.required[lang]}，可用 {sc.available.get(lang, 0)}")
    for m in sc.messages:
        lines.append(f"  - {m}")
    return lines


@app.command()
def doctor(
    skip_azure: bool = typer.Option(False, "--skip-azure", help="不检查 Azure"),
    skip_gemini: bool = typer.Option(False, "--skip-gemini", help="不检查 Gemini（含 grounding）"),
    no_grounding: bool = typer.Option(False, "--no-grounding", help="不检查 Google Search grounding"),
) -> None:
    """上线前自检：分词器、数据库、Azure 部署与配额、Gemini 各模型、grounding（真实调用，花费 < $0.05）。"""
    from loadtest.doctor import run_doctor

    settings = _settings()
    checks = asyncio.run(run_doctor(
        settings, azure=not skip_azure, gemini=not skip_gemini, grounding=not no_grounding
    ))
    for c in checks:
        _echo(f"[{c.status.upper():4}] {c.name}: {c.detail}")
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    _echo(f"结果：{fails} 项失败，{warns} 项警告")
    if fails:
        raise typer.Exit(1)


@app.command("pool-stats")
def pool_stats() -> None:
    """语料池库存、分布、拒绝原因、Gemini 花费与供应检查判定。"""
    for line in asyncio.run(_pool_stats(_settings())):
        _echo(line)


if __name__ == "__main__":  # pragma: no cover
    app()
