"""loadtest doctor 自检测试（假 Azure 客户端 / 假 Gemini 后端，不联网）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from loadtest import cli
from loadtest.cli import app
from loadtest.clock import RealClock
from loadtest.corpus.gemini_client import BackendResult
from loadtest.corpus.store import CorpusStore
from loadtest.doctor import Doctor
from loadtest.load.azure_client import AzureResult
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import clear_registered_secrets, load_settings

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    clear_registered_secrets()
    cfg = yaml.safe_load((REPO / "config.example.yaml").read_text(encoding="utf-8"))
    cfg["db_path"] = str(tmp_path / "corpus.db")
    cfg["log_dir"] = str(tmp_path / "runs")
    cfg["load"]["pid_file"] = str(tmp_path / "runs" / "loadtest.pid")
    cfg["gemini"]["project"] = "proj-doctor"
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    cred = tmp_path / "sa.json"
    cred.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-doctor-test-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred))
    s = load_settings(cfg_path, tmp_path / "none.env")
    yield s, cfg_path
    clear_registered_secrets()


class FakeAzure:
    def __init__(self, result: AzureResult) -> None:
        self.result = result
        self.sent: list = []

    async def send(self, req):
        self.sent.append(req)
        return self.result

    async def aclose(self) -> None:  # pragma: no cover - 注入的客户端不由 doctor 关闭
        pass


def azure_ok(limit_tokens=1_000_000, limit_requests=1000) -> AzureResult:
    return AzureResult(
        "x", Outcome.OK, 200, 120.0, prompt_tokens=12, completion_tokens=1, finish_reason="stop",
        headers=RateLimitHeaders(limit_tokens=limit_tokens, limit_requests=limit_requests),
    )


class FakeGemini:
    """按模型返回结果；值为异常时抛出。grounding 调用单独返回。"""

    def __init__(self, per_model: dict | None = None, grounded=None) -> None:
        self.per_model = per_model or {}
        self.grounded = grounded
        self.calls: list[tuple[str, bool, dict]] = []

    async def generate(self, model, prompt, grounding, timeout_s, **kw):
        self.calls.append((model, grounding, kw))
        r = self.grounded if grounding else self.per_model.get(model)
        if r is None:
            r = BackendResult("OK", "STOP", 8, 3, ("https://news.example/1",) if grounding else (),
                              search_queries=("q1", "q2") if grounding else None)
        if isinstance(r, BaseException):
            raise r
        return r


class ClientError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def do(settings, azure_client, backend, **kw):
    store = await CorpusStore(settings.db_path, RealClock()).open()
    try:
        doc = Doctor(settings, store=store, azure_client=azure_client, gemini_backend=backend)
        checks = await doc.run(**kw)
        spent = await store.spent_usd(["gemini", "gemini_grounding"], "proj-doctor")
        az_spent = await store.spent_usd(["azure"])
        rec = await store.get_run(doc.run_id)
        return {c.name: c for c in checks}, spent, az_spent, rec, doc
    finally:
        await store.close()


async def test_all_ok_and_costs_ledgered(settings) -> None:
    s, _ = settings
    gem = FakeGemini()
    checks, g_spent, a_spent, rec, doc = await do(s, FakeAzure(azure_ok()), gem)
    assert checks["tokenizer"].status == "ok"
    assert checks["database"].status == "warn"  # 空库：提示 warmup
    assert checks["azure"].status == "ok"
    assert checks["azure:quota"].status == "ok"
    for m in s.gemini.model_chain:
        assert checks[f"gemini:{m}"].status == "ok", checks[f"gemini:{m}"].detail
    assert checks["grounding"].status == "ok"
    assert "计费查询 2 次" in checks["grounding"].detail
    # 每个模型一次 ping + 主模型一次 grounding
    assert [(m, g) for m, g, _ in gem.calls] == [(m, False) for m in s.gemini.model_chain] + [
        (s.gemini.model_chain[0], True)
    ]
    assert all(kw["temperature"] == 1.0 for *_, kw in gem.calls)
    # 花费写入账本：Gemini 记在当前账号下，grounding 按 2 次查询计
    assert g_spent > 2 * 0.014 - 1e-9 and a_spent > 0
    assert rec.kind == "generate" and rec.stop_reason == "doctor"
    assert rec.cost_usd == pytest.approx(g_spent + a_spent)
    assert "本次自检花费" in checks["cost"].detail


async def test_azure_404_and_quota_too_low(settings) -> None:
    s, _ = settings
    checks, *_ = await do(
        s, FakeAzure(AzureResult("x", Outcome.CLIENT_ERROR, 404, 50.0, error_code="DeploymentNotFound")),
        FakeGemini(), gemini=False,
    )
    assert checks["azure"].status == "fail" and "deployment" in checks["azure"].detail
    assert checks["gemini"].status == "skip"

    checks, *_ = await do(s, FakeAzure(azure_ok(limit_tokens=300_000, limit_requests=180)), None, gemini=False)
    q = checks["azure:quota"]
    assert q.status == "warn" and "300,000" in q.detail and "--tpm" in q.detail


async def test_azure_401_and_missing_headers(settings) -> None:
    s, _ = settings
    checks, *_ = await do(s, FakeAzure(AzureResult("x", Outcome.CLIENT_ERROR, 401, 5.0)), None, gemini=False)
    assert checks["azure"].status == "fail" and "AZURE_OPENAI_API_KEY" in checks["azure"].detail
    res = AzureResult("x", Outcome.OK, 200, 5.0, prompt_tokens=10, completion_tokens=1)
    checks, *_ = await do(s, FakeAzure(res), None, gemini=False)
    assert checks["azure:quota"].status == "warn"


async def test_gemini_model_unavailable_and_empty_output(settings) -> None:
    s, _ = settings
    primary, second, third = s.gemini.model_chain
    gem = FakeGemini(per_model={
        second: ClientError(404, "Publisher model not found"),
        third: BackendResult("", "MAX_TOKENS", 8, 512),
    })
    checks, *_ = await do(s, None, gem, azure=False)
    assert checks["azure"].status == "skip"
    assert checks[f"gemini:{primary}"].status == "ok"
    bad = checks[f"gemini:{second}"]
    assert bad.status == "fail" and "404" in bad.detail and "thinking_levels" in bad.detail
    empty = checks[f"gemini:{third}"]
    assert empty.status == "warn" and "思考" in empty.detail


async def test_grounding_without_sources_warns(settings) -> None:
    s, _ = settings
    gem = FakeGemini(grounded=BackendResult("text", "STOP", 10, 10, (), search_queries=()))
    checks, *_ = await do(s, None, gem, azure=False)
    assert checks["grounding"].status == "warn" and "未返回来源" in checks["grounding"].detail


async def test_gemini_transient_error_is_warn(settings) -> None:
    s, _ = settings
    primary = s.gemini.model_chain[0]
    checks, *_ = await do(s, None, FakeGemini(per_model={primary: TimeoutError()}), azure=False, grounding=False)
    assert checks[f"gemini:{primary}"].status == "warn"
    assert checks["grounding"].status == "skip"


def test_sdk_loggers_quiet_at_info() -> None:
    import logging

    cli._setup_logging("INFO")
    for name in ("httpx", "openai", "google_genai", "google"):
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


def test_cli_doctor_exit_codes(settings, monkeypatch) -> None:
    s, cfg_path = settings
    base = ["--config", str(cfg_path), "--env-file", str(cfg_path.parent / "none.env")]

    async def fake_ok(settings, **kw):
        from loadtest.doctor import Check
        return [Check("azure", "ok", "fine"), Check("grounding", "warn", "meh")]

    async def fake_fail(settings, **kw):
        from loadtest.doctor import Check
        assert kw == {"azure": False, "gemini": True, "grounding": False}
        return [Check("azure", "skip", "--skip-azure"), Check("gemini:x", "fail", "404")]

    import loadtest.doctor as doctor_mod

    monkeypatch.setattr(doctor_mod, "run_doctor", fake_ok)
    r = CliRunner().invoke(app, [*base, "doctor"])
    assert r.exit_code == 0, r.output
    assert "[OK  ] azure: fine" in r.output and "0 项失败，1 项警告" in r.output

    monkeypatch.setattr(doctor_mod, "run_doctor", fake_fail)
    r = CliRunner().invoke(app, [*base, "doctor", "--skip-azure", "--no-grounding"])
    assert r.exit_code == 1 and "[FAIL] gemini:x: 404" in r.output
    assert cli is not None
