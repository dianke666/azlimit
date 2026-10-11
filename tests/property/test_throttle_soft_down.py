"""Property 12: 主动减速下限（ThrottleController.on_success）。

∀ 响应头序列（无 429 / 403-quota / 503）：state 永不变为 HOLD 或 PROBE，level ≥ 0.50；
每次主动减速只降一级（1.0 → 0.75 → 0.50）；任意 5 秒内最多一次；
remaining 或 limit 头缺失（或 limit ≤ 0）的维度不触发减速。

Validates: Requirements 21.1, 21.2, 21.3, 21.4
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.outcome import RateLimitHeaders
from loadtest.settings import GEMINI_THROTTLE_DEFAULT_LEVELS, ThrottleSettings
from loadtest.throttle import ThrottleController, ThrottleState

S = ThrottleState
CONFIGS = {
    "azure": ThrottleSettings(),
    "gemini": ThrottleSettings(mode="level", levels=list(GEMINI_THROTTLE_DEFAULT_LEVELS), recover_full_s=20.0),
}

REMAINING = st.one_of(st.none(), st.integers(0, 30), st.integers(0, 2000))
LIMIT = st.one_of(st.none(), st.just(0), st.integers(1, 2000))
HEADERS = st.one_of(
    st.none(),
    st.builds(
        RateLimitHeaders,
        remaining_tokens=REMAINING,
        limit_tokens=LIMIT,
        remaining_requests=REMAINING,
        limit_requests=LIMIT,
    ),
)
STEP = st.tuples(st.floats(min_value=0.0, max_value=4.0, allow_nan=False), HEADERS)


def dim_low(rem: int | None, lim: int | None, ratio: float) -> bool | None:
    """None = 该维度缺失（不判定）。"""
    if rem is None or lim is None or lim <= 0:
        return None
    return rem / lim < ratio


@pytest.mark.parametrize("name", list(CONFIGS))
@settings(max_examples=200, deadline=None)
@given(start=st.floats(0.0, 1e5, allow_nan=False), steps=st.lists(STEP, min_size=1, max_size=150))
def test_soft_down_floor(name: str, start: float, steps: list[tuple[float, RateLimitHeaders | None]]) -> None:
    cfg = CONFIGS[name]
    levels = list(cfg.levels)
    floor_idx = levels.index(cfg.soft_down_floor)
    clock = FakeClock(start)
    th = ThrottleController(cfg, clock)
    last_soft = -math.inf  # 模型：最近一次满足减速条件的时刻
    drop_times: list[float] = []

    def tick_until(t_end: float) -> None:
        # 调度器以 ≤ 1s 间隔 tick
        while clock() < t_end:
            clock.set(min(clock() + 1.0, t_end))
            i0 = levels.index(th.level)
            th.tick(clock())
            assert th.state in (S.NORMAL, S.RECOVER)
            assert levels.index(th.level) >= i0  # tick 只升不降

    for dt, h in steps:
        tick_until(clock() + dt)
        now = clock()
        s0, i0, soft0 = th.state, levels.index(th.level), th.stats.soft_downs
        th.on_success(h, now)
        s1, i1 = th.state, levels.index(th.level)

        # 永不 HOLD / PROBE，不低于 floor
        assert s1 in (S.NORMAL, S.RECOVER)
        assert th.level >= cfg.soft_down_floor - 1e-9
        assert th.allowed_dispatch(now)[0] is None

        dims = [] if h is None else [
            dim_low(h.remaining_tokens, h.limit_tokens, cfg.soft_down_ratio),
            dim_low(h.remaining_requests, h.limit_requests, cfg.soft_down_ratio),
        ]
        low = any(d is True for d in dims)
        if all(d is None for d in dims):
            # 头缺失：完全无影响（含冷却计时）
            assert (s1, i1) == (s0, i0) and th.stats.soft_downs == soft0
            continue

        triggered = low and now - last_soft >= cfg.soft_down_cooldown_s
        if triggered:
            last_soft = now
            exp_i = i0 - 1 if i0 - 1 >= floor_idx else i0
            assert i1 == exp_i
            if exp_i < i0:
                drop_times.append(now)
                assert th.stats.soft_downs == soft0 + 1
                assert s1 is (S.RECOVER if s0 is S.NORMAL else s0)
            else:
                assert (s1, th.stats.soft_downs) == (s0, soft0)
        else:
            assert (s1, i1, th.stats.soft_downs) == (s0, i0, soft0)

    # 任意 5 秒内最多一次
    for a, b in zip(drop_times, drop_times[1:]):
        assert b - a >= cfg.soft_down_cooldown_s
    assert th.stats.soft_downs == len(drop_times)
