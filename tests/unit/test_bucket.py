import asyncio

import pytest

from loadtest.bucket import MAX_SINGLE_REQUEST, DualTokenBucket
from loadtest.clock import FakeClock


def test_caps_and_initial_fill():
    c = FakeClock()
    b = DualTokenBucket(600_000, 300, 5.0, 2.0, c)
    assert b.tok_cap == 50_000 and b.req_cap == 25
    assert b.tok == 20_000 and b.req == 10
    # 低速率：容量下限与请求初始存量 >= 1
    low = DualTokenBucket(60, 6, 5.0, 2.0, c)
    assert low.tok_cap == MAX_SINGLE_REQUEST and low.req_cap == 1
    assert low.tok == 2 and low.req == 1


def test_configurable_min_cap_and_burst():
    c = FakeClock()
    b = DualTokenBucket(6000, 60, burst_s=10.0, initial_fill_s=0.0, clock=c, min_tok_cap=100)
    assert b.tok_cap == 1000 and b.req_cap == 10
    assert b.tok == 0 and b.req == 1


def test_try_acquire_and_refill():
    c = FakeClock()
    b = DualTokenBucket(60_000, 60, 5.0, 2.0, c)  # 1000 tok/s, 1 req/s; tok=2000, req=2
    assert b.try_acquire(1500)
    assert not b.try_acquire(1000)  # tok 500 不足
    c.advance(0.5)
    assert b.try_acquire(1000)
    assert not b.try_acquire(1)  # req 用尽
    c.advance(1.0)
    assert b.try_acquire(1)


def test_set_rates_downshift_truncates():
    c = FakeClock()
    b = DualTokenBucket(1_200_000, 600, 5.0, 2.0, c)
    c.advance(10)  # 充满：tok=100_000, req=50
    b.set_rates(120_000, 60)
    assert b.tok == 10_000 and b.req == 5


def test_cost_over_cap_rejected():
    b = DualTokenBucket(60, 60, 5.0, 2.0, FakeClock())
    with pytest.raises(ValueError):
        b.try_acquire(MAX_SINGLE_REQUEST + 1)


async def test_acquire_waits_in_small_steps():
    c = FakeClock()
    b = DualTokenBucket(60_000, 600, 5.0, 0.0, c)  # 初始 tok=0
    await b.acquire(3000)  # 需 3s
    assert c.now() == pytest.approx(3.0, abs=1e-6)
    assert c.sleeps and max(c.sleeps) <= 0.25


async def test_zero_rate_waits_without_error_until_rate_restored():
    c = FakeClock()
    b = DualTokenBucket(0, 0, 5.0, 2.0, c)  # tok=0, req=1
    task = asyncio.create_task(b.acquire(100))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not task.done()
    assert all(s <= 0.25 for s in c.sleeps)
    b.set_rates(60_000, 60)
    await asyncio.wait_for(task, timeout=1)
    assert b.granted_requests == 1


async def test_fifo_order_and_try_acquire_respects_waiters():
    c = FakeClock()
    b = DualTokenBucket(60_000, 6000, 5.0, 0.0, c)
    order: list[int] = []

    async def worker(i: int) -> None:
        await b.acquire(1000)
        order.append(i)

    tasks = [asyncio.create_task(worker(i)) for i in range(5)]
    await asyncio.sleep(0)
    assert not b.try_acquire(0)  # 有等待者持锁
    await asyncio.gather(*tasks)
    assert order == [0, 1, 2, 3, 4]
    assert b.granted_tokens == 5000
