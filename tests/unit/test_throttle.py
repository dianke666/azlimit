"""ThrottleController 单元测试（设计时间线示例、level 模式、探测结果分支、主动减速）。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from loadtest.clock import FakeClock
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import GEMINI_THROTTLE_DEFAULT_LEVELS, ThrottleSettings
from loadtest.throttle import ThrottleController, ThrottleState


@dataclass
class Res:
    outcome: Outcome
    retry_after_s: float | None = None
    headers: RateLimitHeaders | None = None


def run_until(th: ThrottleController, clock: FakeClock, t_end: float, dt: float = 1.0) -> None:
    while clock() < t_end - 1e-12:
        clock.advance(min(dt, t_end - clock()))
        th.tick(clock())


def azure(clock: FakeClock) -> ThrottleController:
    return ThrottleController(ThrottleSettings(), clock)


def gemini(clock: FakeClock) -> ThrottleController:
    cfg = ThrottleSettings(mode="level", levels=list(GEMINI_THROTTLE_DEFAULT_LEVELS), recover_full_s=20.0)
    return ThrottleController(cfg, clock)


def test_step_s() -> None:
    c = FakeClock(0.0)
    assert azure(c).recover_step_s == 15.0
    assert gemini(c).recover_step_s == 10.0


def test_design_timeline_example() -> None:
    clock = FakeClock(0.0)
    th = azure(clock)
    old = th.note_dispatch(-1.0, is_probe=False)  # HOLD 前派发的在途请求
    new = th.note_dispatch(-0.5, is_probe=False)

    th.on_result(new, Res(Outcome.THROTTLED, 6.0), 0.0)
    assert th.state is ThrottleState.HOLD and th.hold_until == 10.0
    assert th.allowed_dispatch(0.0) == (0, 0.0)
    th.note_done(new)

    run_until(th, clock, 3.0)
    th.on_result(old, Res(Outcome.THROTTLED, 10.0), 3.0)  # stale → 只延长
    th.note_done(old)
    assert th.hold_until == 13.0 and th.stats.stale_throttles == 1 and th.stats.throttle_events == 1
    assert th.is_holding(3.0)

    run_until(th, clock, 12.0)
    assert th.state is ThrottleState.HOLD
    run_until(th, clock, 13.0)
    assert th.state is ThrottleState.PROBE
    assert th.allowed_dispatch(13.0) == (1, 0.10)
    probe = th.note_dispatch(13.0, is_probe=True)
    assert th.allowed_dispatch(13.0) == (0, 0.0)

    run_until(th, clock, 15.0)
    th.on_result(probe, Res(Outcome.OK), 15.0)
    th.note_done(probe)
    assert th.state is ThrottleState.RECOVER and th.level == 0.10

    seen: dict[float, float] = {}
    while th.state is not ThrottleState.NORMAL:
        clock.advance(1.0)
        th.tick(clock())
        if th.level > 0.10:
            seen.setdefault(th.level, clock())
    assert seen == {0.25: 30.0, 0.50: 45.0, 0.75: 60.0, 1.0: 75.0}
    assert th.allowed_dispatch(75.0) == (None, 1.0)
    states = [b for _, _, b in th.stats.timeline]
    assert states == [ThrottleState.HOLD, ThrottleState.PROBE, ThrottleState.RECOVER, ThrottleState.NORMAL]
    assert th.stats.time_in_state[ThrottleState.HOLD] == pytest.approx(13.0)
    assert th.stats.time_in_state[ThrottleState.PROBE] == pytest.approx(2.0)
    assert th.stats.time_in_state[ThrottleState.RECOVER] == pytest.approx(60.0)


@pytest.mark.parametrize(("t_ok", "dt"), [(10.0, 1.0), (10.4, 1.0), (10.3, 0.5), (10.0, 0.25)])
def test_probe_ok_to_full_within_bounds(t_ok: float, dt: float) -> None:
    clock = FakeClock(0.0)
    th = azure(clock)
    th.on_throttle(None, 0.0)
    run_until(th, clock, 10.0)
    p = th.note_dispatch(10.0, True)
    clock.set(t_ok)
    th.on_result(p, Res(Outcome.OK), t_ok)
    while th.state is not ThrottleState.NORMAL:
        clock.advance(dt)
        th.tick(clock())
    assert 60.0 - 1e-6 <= clock() - t_ok <= 61.0 + 1e-6


def test_level_mode_gemini() -> None:
    clock = FakeClock(0.0)
    th = gemini(clock)
    th.on_throttle(None, 0.0)
    assert th.state is ThrottleState.HOLD and th.level == 0.10 and th.is_holding(0.0)
    run_until(th, clock, 10.0)
    assert th.state is ThrottleState.RECOVER and th.level == 0.10 and not th.is_holding(10.0)
    run_until(th, clock, 19.0)
    assert th.level == 0.10
    run_until(th, clock, 20.0)
    assert th.level == 0.50
    run_until(th, clock, 30.0)
    assert th.level == 1.0 and th.state is ThrottleState.NORMAL
    assert ThrottleState.PROBE not in [b for _, _, b in th.stats.timeline]


def test_cooldown_clamped_and_monotone() -> None:
    th = azure(FakeClock(0.0))
    th.on_throttle(500.0, 0.0)
    assert th.hold_until == 120.0
    th.on_throttle(1.0, 5.0)  # 新事件不缩短 hold
    assert th.hold_until == 120.0


def test_stale_ignored_in_probe_and_recover() -> None:
    clock = FakeClock(0.0)
    th = azure(clock)
    old = th.note_dispatch(0.0, False)
    clock.set(1.0)
    th.on_throttle(None, 1.0)
    run_until(th, clock, 11.0)
    assert th.state is ThrottleState.PROBE
    th.on_result(old, Res(Outcome.THROTTLED, 30.0), 11.0)
    assert th.state is ThrottleState.PROBE and th.stats.stale_throttles == 1
    # 在途请求未清空：等待 probe_inflight_wait_s
    assert th.allowed_dispatch(20.0) == (0, 0.0)
    assert th.allowed_dispatch(41.0) == (1, 0.10)
    th.note_done(old)
    p = th.note_dispatch(12.0, True)
    th.on_result(p, Res(Outcome.OK), 12.0)
    th.note_done(p)
    th.on_throttle(30.0, 13.0, dispatched_at=0.5)  # RECOVER 中 stale → 忽略
    assert th.state is ThrottleState.RECOVER
    th.on_throttle(None, 14.0, dispatched_at=12.5)  # RECOVER 中新节流 → HOLD
    assert th.state is ThrottleState.HOLD and th.hold_until == 24.0 and th.stats.throttle_events == 2


def test_probe_failure_branches() -> None:
    clock = FakeClock(0.0)
    th = azure(clock)
    th.on_throttle(None, 0.0)
    run_until(th, clock, 10.0)

    p = th.note_dispatch(10.0, True)
    th.on_result(p, Res(Outcome.THROTTLED, 20.0), 11.0)
    th.note_done(p)
    assert th.state is ThrottleState.HOLD and th.hold_until == 31.0
    assert th.stats.consecutive_probe_fail == 1
    run_until(th, clock, 31.0)
    assert th.state is ThrottleState.PROBE

    p = th.note_dispatch(31.0, True)
    th.on_result(p, Res(Outcome.SERVER_ERROR), 32.0)
    th.note_done(p)
    assert th.state is ThrottleState.PROBE
    assert th.allowed_dispatch(36.9) == (0, 0.0)
    assert th.allowed_dispatch(37.0) == (1, 0.10)

    p = th.note_dispatch(37.0, True)
    th.on_result(p, Res(Outcome.CONTENT_FILTER), 38.0)
    th.note_done(p)
    assert th.state is ThrottleState.PROBE and th.allowed_dispatch(38.0) == (1, 0.10)

    p = th.note_dispatch(38.0, True)
    th.on_result(p, Res(Outcome.THROTTLED, None), 39.0)
    th.note_done(p)
    s = th.stats
    assert s.probes_sent == 4 and s.probe_throttled == 2 and s.probe_error == 1
    assert s.probe_content_filter == 1 and s.consecutive_probe_fail == 2 and s.max_consecutive_probe_fail == 2


def test_soft_down() -> None:
    th = azure(FakeClock(0.0))
    low = RateLimitHeaders(remaining_tokens=5, limit_tokens=100)
    th.on_success(low, 0.0)
    assert th.level == 0.75 and th.state is ThrottleState.RECOVER and th.stats.soft_downs == 1
    th.on_success(low, 4.9)  # 冷却期内
    assert th.level == 0.75
    th.on_success(low, 5.0)
    assert th.level == 0.50
    th.on_success(low, 10.0)  # 不低于 floor
    assert th.level == 0.50 and th.stats.soft_downs == 2 and th.last_soft == 10.0
    # 缺失 limit 头 → 不判定
    th2 = azure(FakeClock(0.0))
    th2.on_success(RateLimitHeaders(remaining_tokens=1, remaining_requests=1), 0.0)
    assert th2.level == 1.0 and th2.state is ThrottleState.NORMAL
    # HOLD 中不触发
    th2.on_throttle(None, 1.0)
    th2.on_success(low, 2.0)
    assert th2.stats.soft_downs == 0 and th2.state is ThrottleState.HOLD


def test_note_done_unblocks_probe_without_result() -> None:
    clock = FakeClock(0.0)
    th = azure(clock)
    th.on_throttle(None, 0.0)
    run_until(th, clock, 10.0)
    p = th.note_dispatch(10.0, True)
    th.note_done(p)  # worker 异常，未调用 on_result
    assert not th.probe_in_flight and th.in_flight == 0
    assert th.allowed_dispatch(15.0) == (1, 0.10)


def _assert_timeline_consistent(th: ThrottleController) -> None:
    tl = th.stats.timeline
    if tl:
        assert tl[0][1] is ThrottleState.NORMAL
        assert tl[-1][2] is th.state
    for (t_a, _, to_a), (t_b, from_b, _) in zip(tl, tl[1:]):
        assert t_a <= t_b and to_a is from_b
    assert all(a is not b for _, a, b in tl)


def test_stats_counters_time_and_timeline() -> None:
    import json

    start = 100.0
    clock = FakeClock(start)
    th = azure(clock)
    low = RateLimitHeaders(remaining_requests=1, limit_requests=100)

    th.on_success(low, clock())  # NORMAL → RECOVER（主动减速）
    old = th.note_dispatch(clock(), False)
    run_until(th, clock, 101.0)
    th.on_throttle(None, 101.0)  # 新节流 → HOLD
    run_until(th, clock, 105.0)
    th.on_result(old, Res(Outcome.THROTTLED, 30.0), 105.0)  # stale → 延长到 135
    th.note_done(old)
    assert th.hold_until == 135.0
    run_until(th, clock, 135.0)
    assert th.state is ThrottleState.PROBE

    for outcome, dt in [
        (Outcome.TIMEOUT, 5.0),
        (Outcome.CLIENT_ERROR, 5.0),
        (Outcome.CONTENT_FILTER, 0.0),
        (Outcome.THROTTLED, 0.0),
    ]:
        p = th.note_dispatch(clock(), True)
        run_until(th, clock, clock() + 1.0)
        th.on_result(p, Res(outcome), clock())
        th.note_done(p)
        run_until(th, clock, clock() + dt)
    assert th.state is ThrottleState.HOLD
    run_until(th, clock, th.hold_until)
    p = th.note_dispatch(clock(), True)
    th.on_result(p, Res(Outcome.OK), clock())
    th.note_done(p)
    run_until(th, clock, clock() + 61.0)
    assert th.state is ThrottleState.NORMAL

    s = th.stats
    assert (s.throttle_events, s.stale_throttles, s.soft_downs) == (1, 1, 1)
    assert (s.probes_sent, s.probe_ok, s.probe_throttled, s.probe_error, s.probe_content_filter) == (5, 1, 1, 2, 1)
    assert s.consecutive_probe_fail == 0 and s.max_consecutive_probe_fail == 1
    # 各状态停留时长之和 == 总经过时间
    assert sum(s.time_in_state.values()) == pytest.approx(clock() - start)
    assert all(v >= 0 for v in s.time_in_state.values())
    # timeline 记录每次转移
    assert [b for _, _, b in s.timeline] == [
        ThrottleState.RECOVER,
        ThrottleState.HOLD,
        ThrottleState.PROBE,
        ThrottleState.HOLD,
        ThrottleState.PROBE,
        ThrottleState.RECOVER,
        ThrottleState.NORMAL,
    ]
    _assert_timeline_consistent(th)
    # to_dict JSON 可序列化，且与字段一致
    d = json.loads(json.dumps(th.stats.to_dict()))
    assert d["probes_sent"] == 5 and d["probe_error"] == 2 and d["stale_throttles"] == 1
    assert set(d["time_in_state"]) == {x.value for x in ThrottleState}
    assert sum(d["time_in_state"].values()) == pytest.approx(clock() - start)
    assert len(d["timeline"]) == len(s.timeline)
    assert d["timeline"][0][1:] == ["normal", "recover"]


def test_stats_level_mode_time_and_timeline() -> None:
    import json

    clock = FakeClock(0.0)
    th = gemini(clock)
    th.on_throttle(5.0, 0.0)
    run_until(th, clock, 35.0)
    assert th.state is ThrottleState.NORMAL
    s = th.stats
    assert [(t, a, b) for t, a, b in s.timeline] == [
        (0.0, ThrottleState.NORMAL, ThrottleState.HOLD),
        (10.0, ThrottleState.HOLD, ThrottleState.RECOVER),
        (30.0, ThrottleState.RECOVER, ThrottleState.NORMAL),
    ]
    assert s.time_in_state[ThrottleState.HOLD] == pytest.approx(10.0)
    assert s.time_in_state[ThrottleState.RECOVER] == pytest.approx(20.0)
    assert s.time_in_state[ThrottleState.PROBE] == 0.0
    assert sum(s.time_in_state.values()) == pytest.approx(35.0)
    assert s.throttle_events == 1 and s.probes_sent == 0
    json.dumps(s.to_dict())
    _assert_timeline_consistent(th)
