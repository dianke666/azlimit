"""供应拒绝零请求测试（Property 25 的 run 部分；Requirements 8.3, 8.6, 8.7）。

refuse 且无 ``--allow-low-supply`` → Azure 请求数（含启动探测）为 0、stop_reason = supply_refused、
非零退出并输出各语言缺口与 warmup 命令；``--allow-low-supply`` → 启动并告警；warn → 启动并输出告警。
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loadtest.corpus.planner import check_for_run
from loadtest.load.runner import EXIT_OK, EXIT_SUPPLY
from tests.unit.test_runner import FakeStore, make_ctrl, make_passages, make_settings, opts, run_ctx


def test_refuse_sends_zero_azure_requests(tmp_path: Path) -> None:
    s = make_settings(tmp_path, corpus={"min_runnable": 2000})
    store = FakeStore(make_passages(zh=50, en=30))
    ctx = make_ctrl(s, store=store)
    res = run_ctx(ctx)
    assert res.exit_code == EXIT_SUPPLY and res.stop_reason == "supply_refused"
    assert res.azure_requests == 0 and ctx.client.sent == []  # 含 Probe_Request
    assert ctx.factory_calls == []  # Azure 客户端都没有创建
    assert store.runs["run-test"]["stop_reason"] == "supply_refused"
    assert "load_active_passages" not in store.calls
    text = "\n".join(res.messages)
    assert "zh 库存 50" in text and "en 库存 30" in text  # 各语言缺口
    assert "loadtest warmup --target" in text  # 建议命令
    assert not Path(s.load.pid_file).exists()


def test_refuse_per_language_gap_only(tmp_path: Path) -> None:
    # 总量足够，但 en 低于所需库存（60s 默认 TPM：en 需 79）
    s = make_settings(tmp_path)
    store = FakeStore(make_passages(zh=900, en=60))
    ctx = make_ctrl(s, store=store)
    res = run_ctx(ctx)
    assert res.supply is not None and res.supply.verdict == "refuse"
    assert res.exit_code == EXIT_SUPPLY and res.azure_requests == 0
    assert any("en 库存 60" in m for m in res.messages)


def test_allow_low_supply_starts_with_warning(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 20}, corpus={"min_runnable": 2000})
    store = FakeStore(make_passages(zh=300, en=200))
    ctx = make_ctrl(s, opts(allow_low_supply=True), store=store)
    res = run_ctx(ctx)
    assert res.supply is not None and res.supply.verdict == "refuse"
    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    assert res.startup_probe_outcomes == ["ok"] and len(ctx.client.load_sends) > 0
    assert any("corpus_exhausted" in m for m in res.messages)


def test_warn_starts_and_prints_all_warnings(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 20}, corpus={"healthy_size": 100_000})
    ctx = make_ctrl(s)
    res = run_ctx(ctx)
    sc = res.supply
    assert sc is not None and sc.verdict == "warn" and sc.messages
    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    assert res.azure_requests > 1
    for m in sc.messages:
        assert any(m in line for line in res.messages)
    assert any("loadtest warmup --target" in line for line in res.messages)


def test_ok_supply_no_warnings(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 10})
    res = run_ctx(make_ctrl(s))
    assert res.supply is not None and res.supply.verdict == "ok"
    assert not any("供应检查" in m for m in res.messages)


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    zh=st.integers(0, 400),
    en=st.integers(0, 300),
    min_runnable=st.integers(0, 600),
    duration=st.integers(3, 60),
    allow=st.booleans(),
)
def test_refuse_iff_zero_requests(zh: int, en: int, min_runnable: int, duration: int, allow: bool) -> None:
    """Property 25（run 部分）：refuse ∧ ¬allow ⟺ Azure 请求数为 0 且 EXIT_SUPPLY；否则启动探测已发送。"""
    with tempfile.TemporaryDirectory() as d:
        s = make_settings(Path(d), load={"duration_s": duration}, corpus={"min_runnable": min_runnable})
        store = FakeStore(make_passages(zh=zh, en=en))
        ctx = make_ctrl(s, opts(allow_low_supply=allow), store=store)
        res = run_ctx(ctx)
        expected = check_for_run(asyncio.run(store.stats()), duration, s)
        assert res.supply is not None and res.supply.verdict == expected.verdict
        if expected.verdict == "refuse" and not allow:
            assert res.exit_code == EXIT_SUPPLY and res.stop_reason == "supply_refused"
            assert res.azure_requests == 0 and ctx.client.sent == [] and ctx.factory_calls == []
        else:
            assert res.azure_requests >= 1 and ctx.client.probe_sends == [0.0]
            assert res.stop_reason in ("duration", "corpus_exhausted")
            if expected.verdict == "refuse":
                assert any("corpus_exhausted" in m for m in res.messages)
            elif expected.verdict == "warn":
                assert all(any(m in line for line in res.messages) for m in expected.messages)
