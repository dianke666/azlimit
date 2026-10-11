"""Feature: azure-gpt-load-tester, Property 9: 节流立即生效
（Validates: Requirements 19.1, 19.2, 19.3, 19.6）。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.load.azure_client import parse_retry_after
from loadtest.load.ramp import StartupRamp, effective_level
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.throttle import ThrottleController, ThrottleState
from loadtest.settings import ThrottleSettings

ra_st = st.one_of(st.none(), st.floats(min_value=-100.0, max_value=1000.0, allow_nan=False))


def cooldown(ra: float | None) -> float:
    v = ra if ra is not None and ra > 0 else 0.0
    return min(max(v, 10.0), 120.0)


# 将控制器驱动到某个状态的随机前缀操作
prefix_op = st.one_of(
    st.tuples(st.just("advance"), st.floats(min_value=0.0, max_value=30.0)),
    st.tuples(st.just("throttle"), ra_st),
    st.tuples(st.just("probe_ok")),
    st.tuples(st.just("soft")),
)


def _drive(th: ThrottleController, clock: FakeClock, ops: list[tuple]) -> None:
    for o in ops:
        now = clock()
        if o[0] == "advance":
            clock.advance(o[1])
            th.tick(clock())
        elif o[0] == "throttle":
            th.on_throttle(o[1], now, dispatched_at=now)
        elif o[0] == "probe_ok":
            if th.state is ThrottleState.PROBE and th.allowed_dispatch(now)[0]:
                t = th.note_dispatch(now, True)
                th.on_probe_result(Outcome.OK, None, now)
                th.note_done(t)
        else:
            th.on_success(RateLimitHeaders(remaining_tokens=1, limit_tokens=100), now)


@settings(deadline=None, max_examples=400)
@given(
    prefix=st.lists(prefix_op, max_size=30),
    ra=ra_st,
    dt=st.floats(min_value=0.0, max_value=5.0),
    lag=st.floats(min_value=0.0, max_value=20.0),
    ramp_on=st.booleans(),
)
def test_fresh_throttle_enters_hold(
    prefix: list[tuple], ra: float | None, dt: float, lag: float, ramp_on: bool
) -> None:
    clock = FakeClock(0.0)
    th = ThrottleController(ThrottleSettings(), clock)
    _drive(th, clock, prefix)
    clock.advance(dt)
    now = clock()
    th.tick(now)
    before_until = th.hold_until
    # 非 stale：派发时刻不早于当前 HOLD 周期起点
    dispatched_at = max(th.hold_started_at, now - lag) if math.isfinite(th.hold_started_at) else now - lag
    th.on_throttle(ra, now, dispatched_at=dispatched_at)
    assert th.state is ThrottleState.HOLD
    assert th.allowed_dispatch(now) == (0, 0.0)
    assert th.hold_until >= now + cooldown(ra) - 1e-9
    assert th.hold_until >= before_until
    # 爬坡期间同样生效
    assert effective_level(StartupRamp(enabled=ramp_on).factor(now), th) == 0.0


@settings(deadline=None, max_examples=400)
@given(
    ops=st.lists(
        st.one_of(
            st.tuples(st.just("advance"), st.floats(min_value=0.0, max_value=30.0)),
            st.tuples(st.just("throttle"), ra_st, st.floats(min_value=0.0, max_value=60.0)),
            st.tuples(st.just("probe_ok")),
        ),
        max_size=60,
    )
)
def test_hold_until_never_decreases_and_stale_rules(ops: list[tuple]) -> None:
    clock = FakeClock(0.0)
    th = ThrottleController(ThrottleSettings(), clock)
    for o in ops:
        now = clock()
        prev_until, prev_state, prev_start = th.hold_until, th.state, th.hold_started_at
        if o[0] == "advance":
            clock.advance(o[1])
            th.tick(clock())
        elif o[0] == "throttle":
            dispatched_at = now - o[2]
            stale = dispatched_at < th.hold_started_at
            prev_stale = th.stats.stale_throttles
            th.on_throttle(o[1], now, dispatched_at=dispatched_at)
            if stale:
                assert th.stats.stale_throttles == prev_stale + 1
                assert th.state is prev_state  # 状态不变
                assert th.hold_started_at == prev_start
                if prev_state is ThrottleState.HOLD:
                    assert th.hold_until >= now + cooldown(o[1]) - 1e-9
                else:  # PROBE / RECOVER / NORMAL：忽略
                    assert th.hold_until == prev_until
            else:
                assert th.state is ThrottleState.HOLD
                assert th.hold_until >= now + cooldown(o[1]) - 1e-9
        else:
            if th.state is ThrottleState.PROBE and th.allowed_dispatch(now)[0]:
                t = th.note_dispatch(now, True)
                th.on_probe_result(Outcome.OK, None, now)
                th.note_done(t)
        assert th.hold_until >= prev_until
        if th.state is ThrottleState.HOLD:
            assert th.allowed_dispatch(clock()) == (0, 0.0)


# ---------------------------------------------------------------- parse_retry_after
NOW = 1_700_000_000.0
_now_dt = datetime.fromtimestamp(NOW, tz=timezone.utc)

num_text = st.one_of(
    st.integers(min_value=-10**9, max_value=10**9).map(str),
    st.floats(allow_nan=True, allow_infinity=True).map(repr),
    st.floats(min_value=-1e6, max_value=1e6).map(lambda f: f"{f:.3f}"),
)
date_text = st.integers(min_value=-10**6, max_value=10**6).map(
    lambda s: format_datetime(_now_dt + timedelta(seconds=s), usegmt=True)
)
header_val = st.one_of(num_text, date_text, st.text(max_size=40), st.just(""), st.just("  "))


@settings(deadline=None, max_examples=500)
@given(
    ms=st.one_of(st.none(), header_val),
    sec=st.one_of(st.none(), header_val),
    upper=st.booleans(),
    now_dt=st.booleans(),
)
def test_parse_retry_after_bounded(ms: str | None, sec: str | None, upper: bool, now_dt: bool) -> None:
    h: dict[str, str] = {}
    if ms is not None:
        h["Retry-After-Ms" if upper else "retry-after-ms"] = ms
    if sec is not None:
        h["Retry-After" if upper else "retry-after"] = sec
    v = parse_retry_after(h, _now_dt if now_dt else NOW)
    assert v is None or (isinstance(v, float) and 0.0 <= v <= 120.0)
    if not h:
        assert v is None


def test_parse_retry_after_examples() -> None:
    assert parse_retry_after({"retry-after-ms": "1500", "retry-after": "9"}, NOW) == 1.5
    assert parse_retry_after({"retry-after-ms": "junk", "retry-after": "9"}, NOW) == 9.0
    assert parse_retry_after({"retry-after": "-5"}, NOW) is None
    assert parse_retry_after({"retry-after": "500"}, NOW) == 120.0
    past = format_datetime(_now_dt - timedelta(seconds=30), usegmt=True)
    assert parse_retry_after({"retry-after": past}, NOW) == 0.0
    fut = format_datetime(_now_dt + timedelta(seconds=30), usegmt=True)
    assert abs(parse_retry_after({"retry-after": fut}, NOW) - 30.0) < 1e-6  # type: ignore[operator]
    assert parse_retry_after(None, NOW) is None
    assert parse_retry_after({}, NOW) is None
