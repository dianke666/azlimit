"""Property 6: 限流不超预算（DualTokenBucket）。

∀ 成本序列 c_i ∈ [1540, 6040]、∀ 速率配置、∀ 由 FakeClock 推进驱动的放行时刻序列，
任意区间 [t0, t1] 内放行的 Σcost ≤ tok_cap + tok_rate × (t1 − t0)，
放行请求数 ≤ req_cap + req_rate × (t1 − t0)；等待者按 FIFO 顺序放行。

Validates: Requirements 17.1, 17.2, 17.6, 11.1
"""

from __future__ import annotations

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.bucket import DualTokenBucket
from loadtest.clock import FakeClock

COST = st.integers(min_value=1540, max_value=6040)
TPM = st.integers(min_value=1_000, max_value=2_000_000)
RPM = st.integers(min_value=1, max_value=5_000)

# 每步：(推进秒数, 成本, 贪婪模式)。贪婪模式下反复 try_acquire 直到被拒，压满突发容量。
STEP = st.tuples(st.floats(min_value=0.0, max_value=3.0, allow_nan=False), COST, st.booleans())


def assert_window_bound(
    grants: list[tuple[float, int]], tok_cap: float, tok_rate: float, req_cap: float, req_rate: float
) -> None:
    """对所有以放行时刻为端点的区间检查上界（端点取放行时刻时上界最紧）。"""
    if not grants:
        return
    times: list[float] = []
    toks: list[int] = []
    reqs: list[int] = []
    for t, c in grants:
        if times and t == times[-1]:
            toks[-1] += c
            reqs[-1] += 1
        else:
            assert not times or t > times[-1], "grant times must be non-decreasing"
            times.append(t)
            toks.append(c)
            reqs.append(1)
    ptok = [0]
    preq = [0]
    for a, b in zip(toks, reqs):
        ptok.append(ptok[-1] + a)
        preq.append(preq[-1] + b)
    tol_tok = 1e-6 * max(1.0, tok_cap)
    tol_req = 1e-6
    n = len(times)
    for i in range(n):
        for j in range(i, n):
            dur = times[j] - times[i]
            sum_tok = ptok[j + 1] - ptok[i]
            sum_req = preq[j + 1] - preq[i]
            assert sum_tok <= tok_cap + tok_rate * dur + tol_tok, (times[i], times[j], sum_tok)
            assert sum_req <= req_cap + req_rate * dur + tol_req, (times[i], times[j], sum_req)


def drive(bucket: DualTokenBucket, clock: FakeClock, steps: list[tuple[float, int, bool]]) -> list[tuple[float, int]]:
    grants: list[tuple[float, int]] = []
    for dt, cost, greedy in steps:
        clock.advance(dt)
        for _ in range(1000 if greedy else 1):
            if not bucket.try_acquire(cost):
                break
            grants.append((clock(), cost))
        assert 0.0 <= bucket.tok <= bucket.tok_cap + 1e-9
        assert 0.0 <= bucket.req <= bucket.req_cap + 1e-9
    return grants


@settings(max_examples=200, deadline=None)
@given(tpm=TPM, rpm=RPM, start=st.floats(0.0, 1e6, allow_nan=False), steps=st.lists(STEP, min_size=1, max_size=120))
def test_grants_within_budget(tpm: int, rpm: int, start: float, steps: list[tuple[float, int, bool]]) -> None:
    clock = FakeClock(start)
    bucket = DualTokenBucket(tpm, rpm, clock=clock)
    # 容量公式（Req 17.2）
    assert bucket.tok_cap == max(tpm * 5 / 60, 6040)
    assert bucket.req_cap == max(rpm * 5 / 60, 1)
    grants = drive(bucket, clock, steps)
    assert bucket.granted_tokens == sum(c for _, c in grants)
    assert bucket.granted_requests == len(grants)
    assert_window_bound(grants, bucket.tok_cap, bucket.tok_rate, bucket.req_cap, bucket.req_rate)


@settings(max_examples=100, deadline=None)
@given(tpm=TPM, rpm=RPM, costs=st.lists(COST, min_size=2, max_size=25))
def test_acquire_fifo_and_bounded(tpm: int, rpm: int, costs: list[int]) -> None:
    """异步等待者按 FIFO 放行，重查步长 ≤ 0.25s，且放行序列同样满足预算上界（Req 17.6）。"""
    clock = FakeClock(0.0)

    async def main() -> tuple[list[int], list[tuple[float, int]], DualTokenBucket]:
        bucket = DualTokenBucket(tpm, rpm, clock=clock)
        order: list[int] = []
        grants: list[tuple[float, int]] = []

        async def worker(i: int, c: int) -> None:
            await bucket.acquire(c)
            order.append(i)
            grants.append((clock(), c))

        tasks = []
        for i, c in enumerate(costs):
            tasks.append(asyncio.create_task(worker(i, c)))
            await asyncio.sleep(0)  # 保证按创建顺序排队
        await asyncio.gather(*tasks)
        return order, grants, bucket

    order, grants, bucket = asyncio.run(main())
    assert order == list(range(len(costs)))
    assert all(s <= 0.25 + 1e-12 for s in clock.sleeps)
    assert_window_bound(grants, bucket.tok_cap, bucket.tok_rate, bucket.req_cap, bucket.req_rate)
