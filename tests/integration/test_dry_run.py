"""20.1 dry-run 集成测试（Requirements 22.7, 27.7）。

完整 RunController + 真实 RequestAssembler（tiktoken 精确拟合）+ 合成语料池，SimClock 驱动 2 分钟：
不调用 Azure、不发探测、不写 Azure 账本，输出预期 TPM/RPM、爬坡曲线、语言配比、复用与长度分布。
"""

from __future__ import annotations

import json
from pathlib import Path

from loadtest.load.runner import EXIT_OK, RunOptions
from tests.property.test_assembler_precision import catalog, synthetic_passages
from tests.unit.test_runner import FakeStore, make_ctrl, make_settings, run_ctx


def _real_assembler(index, settings):
    from loadtest.load.assembler import RequestAssembler
    from loadtest.tokens import default_counter

    return RequestAssembler(index, catalog(), default_counter(), cfg=settings.load)


def test_dry_run_two_minutes(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 120})
    store = FakeStore(list(synthetic_passages(600, seed=11)))
    ctx = make_ctrl(s, RunOptions(dry_run=True, no_replenish=True, seed=3), store=store,
                    assembler_factory=_real_assembler)
    res = run_ctx(ctx)

    assert res.exit_code == EXIT_OK and res.stop_reason == "duration"
    # 不调用 Azure、不发探测、不写 Azure 账本、不累加 use_count
    assert ctx.factory_calls == [] and res.azure_requests == 0 and res.startup_probe_outcomes == []
    assert store.ledger == [] and store.use_counts == {}

    msgs = "\n".join(res.messages)
    for key in ("[dry-run] 爬坡曲线", "[dry-run] 语言配比", "[dry-run] 复用分布"):
        assert key in msgs

    report = json.loads((res.run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["requests"]["dispatched"] > 100
    # 预期 TPM 接近目标（2 分钟内含 10s 爬坡与随机游走），RPM 在 200–500
    assert 600_000 <= report["avg_tpm"] <= 921_000
    assert 150 <= report["avg_rpm"] <= 500
    zh = report["lang_share_pct"]["zh"]
    assert 50.0 <= zh <= 70.0
    assert report["max_run_uses"] <= s.corpus.per_run_max_uses
    p50, p95 = report["tokens"]["input_p50"], report["tokens"]["input_p95"]
    assert 2500 <= p50 <= 3500 and p95 <= 6000 + 20
