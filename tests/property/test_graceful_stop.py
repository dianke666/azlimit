"""Property 20: 优雅停止。

∀ 停止时刻（SIGINT/SIGTERM、runs/STOP 文件、duration 到期），停止信号后派发数为 0，所有在途请求
在 drain_timeout 内结算或按预留计入账本，``run_uses`` 已（恰好一次）累加回 ``use_count``。

**Validates: Requirements 27.3, 27.4, 27.5, 9.4**

SimClock 离散事件模拟下运行完整 RunController（FakeStore / FakeClient，无网络）。
"""

from __future__ import annotations

import signal
import tempfile
from pathlib import Path
from random import Random
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loadtest.cost.guard import CostGuard
from loadtest.cost.pricing import token_cost, worst_case_token_cost
from loadtest.outcome import Outcome
from loadtest.settings import CostSettings
from tests.property.test_global_window import SimClock
from tests.unit.test_runner import FakeClient, at, make_ctrl, make_settings, run_ctx

PRICES = CostSettings().prices


def _behavior(seed: int, lat_max: float, p_err: float, outcomes: dict[str, Outcome]):
    rng = Random(seed ^ 0xBEEF)

    def beh(planned: Any, now: float, n: int) -> tuple[float, Outcome, int | None, float | None]:
        lat = rng.uniform(0.05, lat_max)
        o = Outcome.SERVER_ERROR if rng.random() < p_err else Outcome.OK
        outcomes[planned.request_id] = o
        return lat, o, (500 if o is Outcome.SERVER_ERROR else 200), None

    return beh


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**31 - 1),
    stop_at=st.integers(2, 90),
    drain=st.sampled_from([0.0, 1.0, 5.0, 12.5, 30.0]),
    lat_max=st.sampled_from([0.5, 3.0, 20.0, 80.0]),
    p_err=st.sampled_from([0.0, 0.05]),
    how=st.sampled_from(["sigint", "sigterm", "stop_file", "duration"]),
)
def test_graceful_stop(seed: int, stop_at: int, drain: float, lat_max: float, p_err: float, how: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        duration = stop_at if how == "duration" else 600
        s = make_settings(root, load={"duration_s": duration, "drain_timeout_s": drain})
        clock = SimClock()
        outcomes: dict[str, Outcome] = {}
        client = FakeClient(clock, behavior=_behavior(seed, lat_max, p_err, outcomes))
        guard = CostGuard(1e9, 0.0, PRICES)
        ctx = make_ctrl(s, client=client, clock=clock, cost_guard=guard)
        stop_file = Path(s.load.pid_file).parent / "STOP"

        drivers = []
        if how == "sigint":
            drivers.append(at(stop_at, lambda c: c.ctrl.on_signal(signal.SIGINT)))
        elif how == "sigterm":
            drivers.append(at(stop_at, lambda c: c.ctrl.on_signal(signal.SIGTERM)))
        elif how == "stop_file":
            drivers.append(at(stop_at - 0.5, lambda c: stop_file.write_text("", encoding="utf-8")))

        res = run_ctx(ctx, drivers)
        ctrl = ctx.ctrl
        expected_reason = {"sigint": "signal", "sigterm": "signal", "stop_file": "stop_cmd", "duration": "duration"}
        assert res.stop_reason == expected_reason[how]
        t_stop = ctrl.stop_requested_at
        assert t_stop is not None
        if how == "stop_file":
            assert stop_at - 0.5 <= t_stop <= stop_at + 0.5 + 1e-9  # 1s 轮询
        else:
            assert t_stop == pytest.approx(stop_at)

        # 27.3：停止信号后派发数为 0
        assert all(t <= t_stop + 1e-9 for t in client.load_sends)
        # 调度循环在 ≤ tick_interval_s（1s）内观察到停止（等待并发名额时按 ≤1s 分段轮询）
        assert res.summary is not None and t_stop - 1e-9 <= res.summary.t_stop <= t_stop + 1.0 + 1e-9
        drain_start = res.summary.t_stop

        # 27.4：drain ≤ drain_timeout；在 drain 窗口内完成的请求正常结算，其余被取消并按预留结算
        assert clock.now() <= drain_start + drain + 1e-6
        load_ids = [rid for _, rid, task in client.sent if task != "startup_probe"]
        cancelled = [rid for rid in load_ids if rid not in client.done]
        assert res.summary.cancelled == len(cancelled)
        assert res.summary.drained == (not cancelled)
        assert all(client.done[rid] <= drain_start + drain + 1e-6 for rid in load_ids if rid in client.done)
        assert guard.outstanding == 0 and guard.reserved == 0
        th = ctrl.throttle
        assert th is not None and th.in_flight == 0

        price = PRICES[s.azure.model_family]
        expected = 0.0
        for _, rid, task in client.sent:
            p = client.planned[rid]
            if rid not in client.done:
                expected += worst_case_token_cost(price, p.input_tokens, p.max_tokens)
            elif task == "startup_probe" or outcomes.get(rid) is Outcome.OK:
                expected += token_cost(price, p.input_tokens, 1 if task == "startup_probe" else 30, 0)
            else:  # SERVER_ERROR 无 usage → 按预留
                expected += worst_case_token_cost(price, p.input_tokens, p.max_tokens)
        assert guard.run_spent == pytest.approx(expected, rel=1e-9, abs=1e-12)
        assert ctx.store.ledger_cost == pytest.approx(guard.run_spent, rel=1e-9, abs=1e-12)  # 已全部刷盘

        # 27.5 / 9.4：run_uses 恰好一次累加回 use_count；runs 记录与报告已写；pid 删除
        index = ctrl.index
        assert index is not None
        assert ctx.store.use_counts == index.run_uses
        assert sum(index.run_uses.values()) >= len(load_ids)
        run = ctx.store.runs["run-test"]
        assert run["stop_reason"] == res.stop_reason and run["ended"]
        assert "throttle_json" in run and "probes" in run
        assert res.run_dir is not None and (res.run_dir / "report.json").exists()
        assert not Path(s.load.pid_file).exists()
