"""Property 10（调度器层）: HOLD 零派发、PROBE 单请求；并补充 Property 11 的 tick 间隔部分。

∀ 模拟运行（FakeClock 离散事件模拟，任意 429 / 在途返回 / 探测结果交错）：
- HOLD 期间（[进入 HOLD, hold_until)）调度器派发 0 个新请求；
- PROBE 期间每次至多派发 1 个探测请求，探测结果返回前不派发任何其他请求；
- 探测只在在途请求清空或自进入 PROBE 起满 probe_inflight_wait_s（30s）后发出；
- 探测 SERVER_ERROR / TIMEOUT / CLIENT_ERROR 后下一个探测不早于 probe_error_retry_s（5s）；
- JSONL 记录中的 is_probe 与实际探测一致；
- 调度器调用 throttle.tick 的间隔（含 HOLD、PROBE 与泊松间隔内）≤ 1s。

**Validates: Requirements 19.4, 19.7, 20.2, 20.3, 20.6**
"""

from __future__ import annotations

from random import Random
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from loadtest.outcome import Outcome
from loadtest.settings import LoadSettings
from loadtest.throttle import ThrottleState
from tests.property.test_global_window import Harness, make_scheduler

_ERRORS = (Outcome.SERVER_ERROR, Outcome.TIMEOUT, Outcome.CLIENT_ERROR)


class RecordingJsonl:
    def __init__(self) -> None:
        self.rows: dict[str, bool] = {}

    def write_result(self, planned: Any, result: Any, now_wall: float | None = None, is_probe: bool = False) -> None:
        self.rows[planned.request_id] = bool(is_probe)


def _script(seed: int, p_throttle: float, p_error: float, p_cf: float, p_long: float):
    rng = Random(seed)

    def script(planned: Any, now: float):
        lat = rng.uniform(20.0, 45.0) if rng.random() < p_long else rng.uniform(0.2, 3.0)
        u = rng.random()
        if u < p_throttle:
            return lat, Outcome.THROTTLED, rng.choice([None, 0.0, 3.0, 10.0, 25.0, 200.0])
        u -= p_throttle
        if u < p_error:
            return lat, rng.choice(_ERRORS), None
        u -= p_error
        if u < p_cf:
            return lat, Outcome.CONTENT_FILTER, None
        return lat, Outcome.OK, None

    return script


class Tracker:
    """在 on_dispatch / on_result 钩子中在线检查 Property 10。"""

    def __init__(self, h: Harness) -> None:
        self.h = h
        self.outstanding_probes: set[str] = set()
        self.last_probe_error_at: float | None = None
        self.violations: list[str] = []
        self.probe_results: list[tuple[float, Outcome]] = []
        self.wait_cfg = h.throttle.cfg.probe_inflight_wait_s
        self.err_retry = h.throttle.cfg.probe_error_retry_s
        h.sched.ctx.on_result = self.on_result
        orig = h.sched.on_dispatch
        assert orig is not None

        def on_dispatch(now: float, planned: Any, ticket: Any) -> None:
            orig(now, planned, ticket)
            self.on_dispatch(now, planned, ticket)

        h.sched.on_dispatch = on_dispatch

    def on_dispatch(self, now: float, planned: Any, ticket: Any) -> None:
        th = self.h.throttle
        if th.state is ThrottleState.HOLD:
            self.violations.append(f"dispatch during HOLD at {now}")
        if self.outstanding_probes:
            self.violations.append(f"dispatch at {now} while probe {self.outstanding_probes} in flight")
        if ticket.is_probe != (th.state is ThrottleState.PROBE):
            self.violations.append(f"is_probe={ticket.is_probe} but state={th.state} at {now}")
        if ticket.is_probe:
            others = th.in_flight - 1
            if others > 0 and now - th.probe_phase_start < self.wait_cfg - 1e-9:
                self.violations.append(f"probe at {now} with {others} in flight, waited {now - th.probe_phase_start}")
            if self.last_probe_error_at is not None and now < self.last_probe_error_at + self.err_retry - 1e-9:
                self.violations.append(f"probe at {now} < error at {self.last_probe_error_at} + 5s")
            self.outstanding_probes.add(planned.request_id)

    def on_result(self, planned: Any, res: Any, ticket: Any, now: float) -> None:
        if ticket.is_probe:
            self.outstanding_probes.discard(planned.request_id)
            self.probe_results.append((now, res.outcome))
            self.last_probe_error_at = now if res.outcome in _ERRORS else None


def check_hold_intervals(h: Harness) -> None:
    """时间线中每个 HOLD 区间内部没有派发。"""
    tl = h.throttle.stats.timeline
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    for t, a, b in tl:
        if b is ThrottleState.HOLD and a is not ThrottleState.HOLD:
            start = t
        elif a is ThrottleState.HOLD and start is not None:
            intervals.append((start, t))
            start = None
    if start is not None:
        intervals.append((start, float("inf")))
    for d in h.dispatches:
        for lo, hi in intervals:
            assert not (lo < d.t < hi), f"dispatch at {d.t} inside HOLD [{lo}, {hi})"


def check_ticks(h: Harness) -> None:
    s = h.sched
    ticks = [t for t in h.ticks if s.t_start <= t <= (s.t_stop or float("inf"))]
    assert ticks, "no ticks recorded"
    pts = [s.t_start] + ticks + [s.t_stop]
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    assert max(gaps) <= h.sched.tick_s + 1e-9, f"tick gap {max(gaps)}"


def run_checked(h: Harness, duration: float) -> Tracker:
    jsonl = RecordingJsonl()
    h.sched.ctx.jsonl = jsonl
    tr = Tracker(h)
    h.run(duration)
    assert not tr.violations, tr.violations[:5]
    check_hold_intervals(h)
    check_ticks(h)
    # JSONL is_probe 与实际派发一致（drain 超时被取消的请求没有结果行）
    sent = {d.request_id: d.is_probe for d in h.dispatches}
    assert all(sent[rid] == p for rid, p in jsonl.rows.items())
    assert len(jsonl.rows) == len(sent) - h.sched.cancelled
    assert h.sched.probes_dispatched == sum(d.is_probe for d in h.dispatches) == h.throttle.stats.probes_sent
    assert h.cost_guard.outstanding == 0
    return tr


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    seed=st.integers(0, 2**31 - 1),
    p_throttle=st.sampled_from([0.01, 0.05, 0.2, 0.6]),
    p_error=st.sampled_from([0.0, 0.02, 0.1]),
    p_cf=st.sampled_from([0.0, 0.01, 0.05]),
    p_long=st.sampled_from([0.0, 0.05, 0.2]),
    ramp=st.booleans(),
)
def test_hold_zero_probe_single(seed: int, p_throttle: float, p_error: float, p_cf: float, p_long: float,
                                ramp: bool) -> None:
    load = LoadSettings(max_concurrency=64)
    h = make_scheduler(seed=seed, load=load, script=_script(seed, p_throttle, p_error, p_cf, p_long),
                       ramp_enabled=ramp, max_breaker_opens=None)
    run_checked(h, 240.0)


def test_throttle_storm_probe_then_recover() -> None:
    """前 20s 全部 429：进入 HOLD → PROBE（单探测）→ 探测 OK → RECOVER → 60s 后回满 NORMAL。"""

    def script(planned: Any, now: float):
        if now < 20.0:
            return 0.5, Outcome.THROTTLED, 10.0
        return 0.5, Outcome.OK, None

    h = make_scheduler(seed=3, script=script)
    tr = run_checked(h, 200.0)
    stats = h.throttle.stats
    assert stats.throttle_events >= 1 and stats.probes_sent >= 1 and stats.probe_ok == 1
    ok_t = next(t for t, o in tr.probe_results if o is Outcome.OK)
    normal_t = next(t for t, a, b in stats.timeline if b is ThrottleState.NORMAL and t > ok_t)
    assert 60.0 - 1e-6 <= normal_t - ok_t <= 61.0 + 1e-6
    assert h.throttle.state is ThrottleState.NORMAL
    # 探测期间无泊松等待：每次 PROBE 进入后，探测在 ≤ 1s tick + 令牌桶等待内发出
    probes = [d for d in h.dispatches if d.is_probe]
    for p in probes:
        assert p.t - p.probe_phase_start <= 6.0


def test_probe_waits_for_inflight_up_to_30s() -> None:
    """HOLD 前派发的长耗时在途请求：探测在 probe_inflight_wait_s 后发出（不无限等待）。"""

    def script(planned: Any, now: float):
        if planned.request_id == "r1":
            return 1.5, Outcome.THROTTLED, 10.0
        if now < 1.5:
            return 120.0, Outcome.OK, None  # 长在途
        return 0.5, Outcome.OK, None

    h = make_scheduler(seed=5, script=script, ramp_enabled=False)
    run_checked(h, 150.0)
    probes = [d for d in h.dispatches if d.is_probe]
    assert probes
    first = probes[0]
    assert first.throttle_in_flight - 1 > 0  # 仍有旧在途请求
    assert first.t - first.probe_phase_start >= 30.0 - 1e-9
