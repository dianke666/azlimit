import asyncio
import time

import pytest

from loadtest.clock import Clock, FakeClock, RealClock


def test_clocks_satisfy_protocol():
    assert isinstance(RealClock(), Clock)
    assert isinstance(FakeClock(), Clock)


def test_fake_clock_advance_and_set():
    c = FakeClock(start=5.0, wall_offset=1000.0)
    assert c() == c.now() == 5.0
    assert c.wall() == 1005.0
    c.advance(2.5)
    assert c.now() == 7.5
    c.set(10.0)
    assert c.now() == 10.0


def test_fake_clock_rejects_going_backwards():
    c = FakeClock(start=3.0)
    with pytest.raises(ValueError):
        c.advance(-1)
    with pytest.raises(ValueError):
        c.set(2.0)


async def test_fake_clock_sleep_advances_without_real_wait():
    c = FakeClock()
    t0 = time.monotonic()
    await c.sleep(3600)
    await c.sleep(-1)  # 负值按 0 处理
    assert c.now() == 3600
    assert c.sleeps == [3600, 0.0]
    assert time.monotonic() - t0 < 1.0


async def test_fake_clock_injected_sleep_is_delegated():
    calls: list[float] = []

    async def my_sleep(s: float) -> None:
        calls.append(s)
        c.advance(s / 2)  # 注入的 sleep 自行决定如何推进时间

    c = FakeClock(sleep_fn=my_sleep)
    await c.sleep(4)
    assert calls == [4.0]
    assert c.now() == 2.0


async def test_real_clock_monotonic_and_sleep():
    c = RealClock()
    a = c()
    await c.sleep(0.01)
    assert c.now() >= a
    assert abs(c.wall() - time.time()) < 1.0
    await asyncio.sleep(0)
