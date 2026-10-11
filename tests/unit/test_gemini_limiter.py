"""GeminiRateLimiter 单元测试（Requirements 10.1, 11.1, 11.2, 11.7）。"""

from __future__ import annotations

import pytest

from loadtest.clock import FakeClock
from loadtest.corpus.gemini_limiter import GeminiRateLimiter
from loadtest.throttle import ThrottleState


def test_default_rates_half_quota() -> None:
    lim = GeminiRateLimiter(clock=FakeClock())
    assert lim.rpm == pytest.approx(300.0)
    assert lim.tpm == pytest.approx(400_000.0)
    assert lim.throttle.mode == "level"


def test_rpm_cap_for_replenisher() -> None:
    lim = GeminiRateLimiter(clock=FakeClock(), rpm_cap=30)
    assert lim.rpm == pytest.approx(30.0)
    assert lim.tpm == pytest.approx(400_000.0)


def test_rates_scale_with_throttle_level_and_recover() -> None:
    clock = FakeClock()
    lim = GeminiRateLimiter(clock=clock)
    lim.on_throttle(retry_after=None)
    assert lim.throttle.state is ThrottleState.HOLD
    assert lim.rpm == pytest.approx(30.0)  # 300 × 10%
    assert lim.tpm == pytest.approx(40_000.0)
    clock.advance(10.0)  # min_cooldown 10s → RECOVER @ 10%
    lim._sync()
    assert lim.rpm == pytest.approx(30.0)
    clock.advance(10.0)
    lim._sync()
    assert lim.rpm == pytest.approx(150.0)  # 50%
    clock.advance(10.0)
    lim._sync()
    assert lim.rpm == pytest.approx(300.0)
    assert lim.throttle.state is ThrottleState.NORMAL


async def test_hold_blocks_acquire_until_cooldown() -> None:
    clock = FakeClock()
    lim = GeminiRateLimiter(clock=clock)
    lim.on_throttle(retry_after=30.0)
    assert lim.is_holding()
    assert not lim.try_acquire(100)
    await lim.acquire(100)
    assert clock.now() >= 30.0
    # HOLD 期间睡眠步长 ≤ 1s
    assert clock.sleeps and max(clock.sleeps) <= 1.0


async def test_acquire_rate_bounded_and_cost_clamped() -> None:
    clock = FakeClock()
    lim = GeminiRateLimiter(clock=clock, rpm_cap=30)
    granted = 0
    while clock.now() < 60.0:
        await lim.acquire(3000)
        granted += 1
    # ≤ req_cap + rate × t（30 RPM, burst 5s → cap 2.5；外加跨越 60s 的最后一次）
    assert granted <= lim.bucket.req_cap + 30 * clock.now() / 60.0 + 1e-6
    assert granted >= 25
    assert max(clock.sleeps) <= 0.25
    # 超过 tok_cap 的估计被截断，不抛错
    cost = await lim.acquire(10**7)
    assert cost == int(lim.bucket.tok_cap)


async def test_level_change_applies_while_waiting() -> None:
    clock = FakeClock()
    lim = GeminiRateLimiter(clock=clock)
    lim.on_throttle(retry_after=None)  # hold 10s → 10%
    await lim.acquire(10)
    # hold 结束后从 10% 恢复，最终升至 100%
    while clock.now() < 40.0:
        await lim.acquire(10)
    assert lim.rpm == pytest.approx(300.0)
