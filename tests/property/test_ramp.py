"""Feature: azure-gpt-load-tester, Property 13: 启动爬坡与初始存量
（Validates: Requirements 18.1, 18.2, 18.3, 18.5, 17.5）。"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.bucket import DualTokenBucket
from loadtest.clock import FakeClock
from loadtest.load.ramp import StartupRamp, ramp_factor

finite = st.floats(min_value=-1e6, max_value=1e9, allow_nan=False, allow_infinity=False)


@settings(deadline=None, max_examples=300)
@given(t=st.floats(min_value=0.0, max_value=10.0, exclude_max=True))
def test_default_ramp_half_before_10s(t: float) -> None:
    assert StartupRamp().factor(t) == 0.5


@settings(deadline=None, max_examples=300)
@given(t=st.floats(min_value=10.0, max_value=1e9, allow_nan=False))
def test_default_ramp_full_after_10s(t: float) -> None:
    r = StartupRamp()
    assert r.factor(t) == 1.0
    assert r.done(t)


@settings(deadline=None, max_examples=300)
@given(t=finite)
def test_no_ramp_always_full(t: float) -> None:
    r = StartupRamp(enabled=False)
    assert r.factor(t) == 1.0
    assert r.levels == frozenset({1.0})


@st.composite
def ramp_steps(draw: st.DrawFn) -> list[tuple[float, float]]:
    n = draw(st.integers(min_value=1, max_value=6))
    gaps = draw(st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=n - 1, max_size=n - 1))
    times = [0.0]
    for g in gaps:
        times.append(times[-1] + g)
    lv = sorted(draw(st.lists(st.floats(min_value=0.01, max_value=1.0), min_size=n, max_size=n)))
    lv[-1] = 1.0
    return list(zip(times, lv))


@settings(deadline=None, max_examples=300)
@given(steps=ramp_steps(), ts=st.lists(finite, min_size=2, max_size=20))
def test_ramp_monotone_in_unit_interval(steps: list[tuple[float, float]], ts: list[float]) -> None:
    r = StartupRamp(steps)
    vals = [r.factor(t) for t in sorted(ts)]
    assert all(0.0 < v <= 1.0 for v in vals)
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert all(v in r.levels for v in vals)
    assert r.factor(steps[-1][0]) == 1.0
    assert ramp_factor(r.steps, -1.0) == steps[0][1]


@pytest.mark.parametrize(
    "bad",
    [[(1.0, 0.5), (10.0, 1.0)], [(0.0, 0.5), (10.0, 0.9)], [(0.0, 0.5), (0.0, 1.0)], [(0.0, 0.0), (5.0, 1.0)]],
)
def test_invalid_steps_rejected(bad: list[tuple[float, float]]) -> None:
    with pytest.raises(ValueError):
        StartupRamp(bad)


@settings(deadline=None, max_examples=300)
@given(
    tpm=st.floats(min_value=0.0, max_value=5e6, allow_nan=False),
    rpm=st.floats(min_value=0.0, max_value=1e5, allow_nan=False),
)
def test_bucket_initial_fill_two_seconds(tpm: float, rpm: float) -> None:
    b = DualTokenBucket(tpm, rpm, clock=FakeClock(0.0))
    assert b.tok == pytest.approx(min(b.tok_cap, 2.0 * tpm / 60.0))
    assert b.req == pytest.approx(min(b.req_cap, max(1.0, 2.0 * rpm / 60.0)))
    # 默认 burst 5s > 2s：初始存量不会被容量截断（token 维度）
    assert b.tok == pytest.approx(2.0 * tpm / 60.0)
