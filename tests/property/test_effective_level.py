"""Feature: azure-gpt-load-tester, Property 14: Effective_Level 合成（Validates: Requirements 18.4）。

随机交错 节流 / 探测结果 / 主动减速 / tick / 爬坡时间，检查：
- throttle.level ∈ levels 且 ≤ 1.0；
- NORMAL / RECOVER → effective_level == min(ramp, level)；HOLD → 0；PROBE → levels[0]。
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.load.ramp import StartupRamp, effective_level
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import GEMINI_THROTTLE_DEFAULT_LEVELS, ThrottleSettings
from loadtest.throttle import ThrottleController, ThrottleState

ra = st.one_of(st.none(), st.floats(min_value=-10.0, max_value=300.0, allow_nan=False))
op = st.one_of(
    st.tuples(st.just("advance"), st.floats(min_value=0.0, max_value=40.0, allow_nan=False)),
    st.tuples(st.just("throttle"), ra, st.floats(min_value=-50.0, max_value=0.0)),
    st.tuples(st.just("probe"), st.sampled_from(list(Outcome)), ra),
    st.tuples(st.just("soft"), st.integers(0, 100), st.integers(1, 100)),
)


def _check(th: ThrottleController, ramp: StartupRamp, elapsed: float) -> None:
    assert th.level in th.levels and th.level <= 1.0
    r = ramp.factor(elapsed)
    eff = effective_level(r, th)
    if th.state is ThrottleState.HOLD:
        assert eff == 0.0
    elif th.state is ThrottleState.PROBE:
        assert eff == th.levels[0]
    else:
        assert eff == min(r, th.level)
    assert 0.0 <= eff <= 1.0


@settings(deadline=None, max_examples=300)
@given(
    ops=st.lists(op, max_size=60),
    mode=st.sampled_from(["probe", "level"]),
    ramp_on=st.booleans(),
)
def test_effective_level_composition(ops: list[tuple], mode: str, ramp_on: bool) -> None:
    clock = FakeClock(0.0)
    cfg = (
        ThrottleSettings()
        if mode == "probe"
        else ThrottleSettings(mode="level", levels=list(GEMINI_THROTTLE_DEFAULT_LEVELS), recover_full_s=20.0)
    )
    th = ThrottleController(cfg, clock)
    ramp = StartupRamp(enabled=ramp_on)
    _check(th, ramp, 0.0)
    for o in ops:
        now = clock()
        kind = o[0]
        if kind == "advance":
            clock.advance(o[1])
            th.tick(clock())
        elif kind == "throttle":
            th.on_throttle(o[1], now, dispatched_at=now + o[2])
        elif kind == "probe":
            if th.state is ThrottleState.PROBE:
                t = th.note_dispatch(now, True)
                th.on_probe_result(o[1], o[2], now)
                th.note_done(t)
        else:
            th.on_success(RateLimitHeaders(remaining_tokens=o[1], limit_tokens=o[2]), now)
        _check(th, ramp, clock())
