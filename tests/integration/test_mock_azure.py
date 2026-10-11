"""20.2 Mock Azure 端到端集成测试（Requirements 18.1, 19.1, 19.4, 19.7, 20.1, 20.2, 20.5, 23.2, 24.1, 27.3, 28.4）。

完整 RunController（SimClock，无网络），脚本化 Azure：
t ≈ 30s 收到 429 → HOLD 期间零派发 → 恰好 1 个 Throttle_Probe → 探测再次 429 → 回到 HOLD →
再探测 OK → 10% → 25% → 50% → 75% → 100% 约 60s 回满；JSONL 中 is_probe 与探测一致；
报告中包含 ThrottleStats 与时间线。另测内容过滤隔离与熔断停止。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loadtest.load.runner import EXIT_CIRCUIT, RunOptions
from loadtest.outcome import Outcome
from loadtest.throttle import ThrottleState
from tests.property.test_global_window import SimClock
from tests.unit.test_runner import FakeClient, make_ctrl, make_settings, run_ctx

S = ThrottleState


def _throttle_script():
    st = {"trigger_at": None, "probes": 0, "recovered": False}

    def beh(planned, now, n):
        if st["trigger_at"] is None and now >= 30.0:
            st["trigger_at"] = now
            return 0.5, Outcome.THROTTLED, 429, 6.0
        if st["trigger_at"] is not None and not st["recovered"] and now >= st["trigger_at"] + 0.5:
            st["probes"] += 1  # HOLD 后只可能派发探测
            if st["probes"] == 1:
                return 0.5, Outcome.THROTTLED, 429, 10.0
            st["recovered"] = True
            return 0.5, Outcome.OK, 200, None
        return 1.0, Outcome.OK, 200, None

    return beh, st


def _hold_intervals(timeline, end):
    out, start = [], None
    for t, a, b in timeline:
        if b is S.HOLD and start is None:
            start = t
        elif a is S.HOLD and b is not S.HOLD and start is not None:
            out.append((start, t))
            start = None
    if start is not None:
        out.append((start, end))
    return out


def test_429_hold_single_probe_and_60s_recovery(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 200})
    clock = SimClock()
    beh, st = _throttle_script()
    client = FakeClient(clock, behavior=beh)
    ctx = make_ctrl(s, RunOptions(no_replenish=True, seed=5), client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.stop_reason == "duration"

    th = ctx.ctrl.throttle
    stats = th.stats
    assert stats.throttle_events == 1
    assert stats.probes_sent == 2 and stats.probe_throttled == 1 and stats.probe_ok == 1
    assert stats.max_consecutive_probe_fail == 1

    # 状态序列：HOLD → PROBE → HOLD → PROBE → RECOVER → NORMAL
    states = [b for _, _, b in stats.timeline]
    assert states == [S.HOLD, S.PROBE, S.HOLD, S.PROBE, S.RECOVER, S.NORMAL]

    # HOLD 期间零派发（19.4）
    sends = [t for t, _, task in client.sent if task != "startup_probe"]
    for a, b in _hold_intervals(stats.timeline, clock.now()):
        assert not [t for t in sends if a + 1e-9 < t < b - 1e-9]
        assert b - a >= 10.0 - 1e-9  # min_cooldown

    # 探测成功 → 100% 用时 ∈ [60, 61]（20.5）
    t_recover = next(t for t, _, b in stats.timeline if b is S.RECOVER)
    t_normal = next(t for t, _, b in stats.timeline if b is S.NORMAL)
    assert 60.0 - 1e-6 <= t_normal - t_recover <= 61.0 + 1e-6

    # PROBE 期间只派发探测：两次探测之间（第一次 HOLD 结束 → RECOVER）只有 2 个请求
    first_hold_end = _hold_intervals(stats.timeline, clock.now())[0][1]
    between = [t for t in sends if first_hold_end - 1e-9 <= t < t_recover]
    assert len(between) == 2

    # JSONL：is_probe 与实际探测一致（28.2）
    rows = [json.loads(l) for l in (res.run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l]
    probes = [r for r in rows if r["is_probe"]]
    assert len(probes) == 2
    assert [r["outcome"] for r in probes] == ["throttled", "ok"]

    # 报告包含 ThrottleStats 与时间线（28.4）
    rep = json.loads((res.run_dir / "report.json").read_text(encoding="utf-8"))
    assert rep["throttle_stats"]["probes_sent"] == 2
    ev = rep["throttle_timeline"]
    assert len(ev) == 1 and ev[0]["hold_cycles"] == 2 and ev[0]["full_at"] is not None
    md = (res.run_dir / "report.md").read_text(encoding="utf-8")
    assert "ThrottleStats" in md and "Throttle event timeline" in md

    # 启动爬坡：前 10s 速率约为满速一半（18.1）
    early = len([t for t in sends if t < 10.0])
    full = len([t for t in sends if 15.0 <= t < 25.0])
    assert early < full


def test_content_filter_quarantines_passages(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 60})
    clock = SimClock()
    flagged: list[str] = []

    def beh(planned, now, n):
        if n % 50 == 0:
            flagged.append(planned.request_id)
            return 0.5, Outcome.CONTENT_FILTER, 400, None
        return 1.0, Outcome.OK, 200, None

    client = FakeClient(clock, behavior=beh)
    ctx = make_ctrl(s, RunOptions(no_replenish=True, seed=2), client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.stop_reason == "duration"
    assert flagged
    bad = {pid for rid in flagged for pid in client.planned[rid].passage_ids}
    assert bad <= set(ctx.store.quarantined)
    # 隔离后同一运行内不再使用（24.2）
    for t, rid, task in client.sent:
        p = client.planned[rid]
        first_flag = min(client.done[r] for r in flagged if set(client.planned[r].passage_ids) & set(p.passage_ids)) \
            if set(p.passage_ids) & bad else None
        if first_flag is not None and rid not in flagged:
            assert t <= first_flag + 1e-9


def test_persistent_server_errors_trip_breaker_and_stop(tmp_path: Path) -> None:
    s = make_settings(tmp_path, load={"duration_s": 900})
    clock = SimClock()
    client = FakeClient(clock, behavior=lambda p, now, n: (0.5, Outcome.SERVER_ERROR, 500, None))
    ctx = make_ctrl(s, RunOptions(no_replenish=True, seed=4), client=client, clock=clock)
    res = run_ctx(ctx)
    assert res.stop_reason == "circuit_open" and res.exit_code == EXIT_CIRCUIT
    # 每次 open 期间不派发（23.3）：结束时间远早于 900s
    assert clock.now() < 400


@pytest.mark.parametrize("ra", [None, 3.0, 40.0])
def test_retry_after_honoured(tmp_path: Path, ra) -> None:
    s = make_settings(tmp_path, load={"duration_s": 150})
    clock = SimClock()
    st = {"done": False}

    def beh(planned, now, n):
        if not st["done"] and now >= 20.0:
            st["done"] = True
            return 0.2, Outcome.THROTTLED, 429, ra
        return 1.0, Outcome.OK, 200, None

    client = FakeClient(clock, behavior=beh)
    ctx = make_ctrl(s, RunOptions(no_replenish=True, seed=9), client=client, clock=clock)
    run_ctx(ctx)
    tl = ctx.ctrl.throttle.stats.timeline
    t_hold = next(t for t, _, b in tl if b is S.HOLD)
    t_probe = next(t for t, _, b in tl if b is S.PROBE)
    expected = min(max(ra or 0.0, 10.0), 120.0)
    assert t_probe - t_hold >= expected - 1e-6
