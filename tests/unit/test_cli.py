"""CLI 单元测试（Requirements 25.8, 27.10, 29.1–29.5, 29.7）。"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from loadtest import cli
from loadtest.cli import app, parse_duration
from loadtest.clock import RealClock
from loadtest.corpus.store import CorpusStore, LedgerEntry
from loadtest.load.runner import RunResult
from loadtest.settings import clear_registered_secrets

REPO = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    clear_registered_secrets()
    cfg = yaml.safe_load((REPO / "config.example.yaml").read_text(encoding="utf-8"))
    cfg["db_path"] = str(tmp_path / "corpus.db")
    cfg["log_dir"] = str(tmp_path / "runs")
    cfg["load"]["pid_file"] = str(tmp_path / "runs" / "loadtest.pid")
    cfg["gemini"]["project"] = "test-project"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    cred = tmp_path / "sa.json"
    cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-cli-test-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    yield tmp_path, ["--config", str(cfg_path), "--env-file", str(tmp_path / "none.env")]
    clear_registered_secrets()


def invoke(args, base):
    return runner.invoke(app, [*base, *args])


def test_help_lists_commands():
    out = runner.invoke(app, ["--help"]).output
    for c in ("warmup", "generate", "run", "stop", "status", "budget", "report", "pool-stats"):
        assert c in out


def test_run_options_present():
    out = runner.invoke(app, ["run", "--help"], terminal_width=200).output
    for o in ("--duration", "--tpm", "--rpm-max", "--deployment", "--no-replenish", "--no-ramp",
              "--allow-low-supply", "--allow-partial", "--dry-run"):
        assert o in out
    w = runner.invoke(app, ["warmup", "--help"], terminal_width=200).output
    for o in ("--target", "--concurrency", "--max-cost"):
        assert o in w
    g = runner.invoke(app, ["generate", "--help"], terminal_width=200).output
    for o in ("--count", "--category", "--lang"):
        assert o in g


@pytest.mark.parametrize("text,expected", [("20m", 1200), ("90s", 90), ("1h", 3600), ("300", 300), ("2.5m", 150)])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize(
    "args,name",
    [
        (["run", "--duration", "abc"], "--duration"),
        (["run", "--duration", "0"], "--duration"),
        (["run", "--tpm", "-5"], "--tpm"),
        (["run", "--rpm-max", "0"], "--rpm-max"),
        (["warmup", "--target", "0"], "--target"),
        (["warmup", "--concurrency", "-1"], "--concurrency"),
        (["warmup", "--max-cost", "0"], "--max-cost"),
        (["generate", "--count", "0"], "--count"),
        (["generate", "--count", "5", "--lang", "fr"], "--lang"),
        (["generate", "--count", "5", "--category", "no_such_cat"], "--category"),
    ],
)
def test_invalid_options(env, args, name):
    _, base = env
    r = invoke(args, base)
    assert r.exit_code != 0
    assert name in r.output


def test_stop_without_running(env):
    _, base = env
    r = invoke(["stop"], base)
    assert r.exit_code != 0
    assert "没有运行中的压测" in r.output


def test_status_not_running(env):
    _, base = env
    r = invoke(["status"], base)
    assert r.exit_code == 0
    assert "没有运行中的压测" in r.output


def test_status_running_with_live_pid(env):
    tmp, base = env
    pid_file = tmp / "runs" / "loadtest.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    async def seed():
        async with CorpusStore(tmp / "corpus.db", RealClock()) as s:
            await s.create_run("run-live", "load", "{}")

    asyncio.run(seed())
    r = invoke(["status"], base)
    assert r.exit_code == 0
    assert "运行中" in r.output and "run-live" in r.output


def test_stop_with_live_pid_writes_stop_file(env, monkeypatch):
    tmp, base = env
    pid_file = tmp / "runs" / "loadtest.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text("424242", encoding="utf-8")
    monkeypatch.setattr("loadtest.load.runner.pid_alive", lambda pid: True)
    sent = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    r = invoke(["stop"], base)
    assert r.exit_code == 0
    assert (tmp / "runs" / "STOP").exists()
    if os.name != "nt":
        assert sent and sent[0][0] == 424242


def test_report_unknown_run(env):
    _, base = env
    r = invoke(["report", "nope"], base)
    assert r.exit_code != 0
    assert "nope" in r.output


def test_report_from_runs_table_and_md(env):
    tmp, base = env

    async def seed():
        async with CorpusStore(tmp / "corpus.db", RealClock()) as s:
            await s.create_run("run-a", "load", "{}")
            await s.finish_run("run-a", "duration", requests_ok=10)

    asyncio.run(seed())
    r = invoke(["report"], base)
    assert r.exit_code == 0 and "run-a" in r.output and "duration" in r.output
    d = tmp / "runs" / "run-a"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.md").write_text("# Load test report run-a\nhello", encoding="utf-8")
    r = invoke(["report", "run-a"], base)
    assert r.exit_code == 0 and "hello" in r.output


def test_budget_output(env):
    tmp, base = env

    async def seed():
        async with CorpusStore(tmp / "corpus.db", RealClock()) as s:
            await s.create_run("r1", "load", "{}")
            await s.write_ledger([LedgerEntry("r1", "azure", "gpt-4.1-nano", 1_000_000, 1000, 10.0)])
            await s.write_ledger([
                LedgerEntry("r1", "gemini", "gemini-3.1-flash-lite", 100, 100, 1.5, account="test-project"),
                LedgerEntry("r1", "gemini", "gemini-3.1-flash-lite", 100, 100, 240.0, account="old-project"),
            ])

    asyncio.run(seed())
    r = invoke(["budget"], base)
    assert r.exit_code == 0, r.output
    out = r.output
    assert "Azure" in out and "10.0000" in out and "180.00" in out
    assert "gpt-4.1-nano" in out and "gemini-3.1-flash-lite" in out
    # Gemini 只计当前账号；旧账号单列
    gem = next(l for l in out.splitlines() if l.startswith("Gemini"))
    assert "test-project" in gem and "1.5000" in gem and "250.00" in gem
    assert "old-project" in out and "240.0000" in out and "不计入当前上限" in out
    assert "剩余可运行次数" in out
    # (180 - 0.5 - 10) / ≈1.66 → ≈102
    line = next(l for l in out.splitlines() if "剩余可运行次数" in l)
    assert 95 <= int(line.split(":")[-1].strip()) <= 110


def test_pool_stats_output(env):
    _, base = env
    r = invoke(["pool-stats"], base)
    assert r.exit_code == 0, r.output
    assert "active=0" in r.output
    assert "refuse" in r.output
    assert "zh: 需要" in r.output and "en: 需要" in r.output


def test_run_passes_options(env, monkeypatch):
    _, base = env
    captured = {}

    class FakeController:
        def __init__(self, settings, opts):
            captured["opts"] = opts

        async def run(self):
            return RunResult(exit_code=0, stop_reason="duration", run_id="run-x")

    monkeypatch.setattr("loadtest.load.runner.RunController", FakeController)
    r = invoke(["run", "--dry-run", "--duration", "2m", "--tpm", "50000", "--no-ramp", "--seed", "7"], base)
    assert r.exit_code == 0, r.output
    o = captured["opts"]
    assert o.duration_s == 120 and o.dry_run and o.tpm == 50000 and o.no_ramp and o.seed == 7


def test_run_exit_code_propagates(env, monkeypatch):
    _, base = env

    class FakeController:
        def __init__(self, settings, opts):
            pass

        async def run(self):
            return RunResult(exit_code=5, stop_reason="supply_refused", run_id="run-y")

    monkeypatch.setattr("loadtest.load.runner.RunController", FakeController)
    r = invoke(["run"], base)
    assert r.exit_code == 5


def test_config_error_exits_nonzero(tmp_path, monkeypatch):
    for k in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "GOOGLE_APPLICATION_CREDENTIALS"):
        monkeypatch.delenv(k, raising=False)
    r = runner.invoke(app, ["--config", str(tmp_path / "missing.yaml"), "budget"])
    assert r.exit_code != 0
