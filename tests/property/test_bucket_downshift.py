"""Property 8: 降档生效（DualTokenBucket.set_rates）。

∀ 运行时刻 t 调用 set_rates(r × 0.1)，此后任一区间放行量 ≤ 新容量 + 新速率 × 时长。

Validates: Requirements 17.4
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.bucket import DualTokenBucket
from loadtest.clock import FakeClock
from tests.property.test_bucket_bound import COST, RPM, STEP, TPM, assert_window_bound, drive


@settings(max_examples=200, deadline=None)
@given(
    tpm=TPM,
    rpm=RPM,
    before=st.lists(STEP, min_size=0, max_size=60),
    switch_dt=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    burst_at_switch=st.tuples(COST, st.booleans()),
    after=st.lists(STEP, min_size=1, max_size=80),
)
def test_downshift_bound(
    tpm: int,
    rpm: int,
    before: list[tuple[float, int, bool]],
    switch_dt: float,
    burst_at_switch: tuple[int, bool],
    after: list[tuple[float, int, bool]],
) -> None:
    clock = FakeClock(0.0)
    bucket = DualTokenBucket(tpm, rpm, clock=clock)
    old = (bucket.tok_cap, bucket.tok_rate, bucket.req_cap, bucket.req_rate)
    pre = drive(bucket, clock, before)
    assert_window_bound(pre, *old)

    clock.advance(switch_dt)
    bucket.set_rates(tpm * 0.1, rpm * 0.1)
    # 新容量 / 速率（Req 17.2 公式），存量立即截断
    assert bucket.tok_rate == tpm * 0.1 / 60 and bucket.req_rate == rpm * 0.1 / 60
    assert bucket.tok_cap == max(tpm * 0.1 * 5 / 60, 6040)
    assert bucket.req_cap == max(rpm * 0.1 * 5 / 60, 1)
    assert bucket.tok <= bucket.tok_cap and bucket.req <= bucket.req_cap
    new = (bucket.tok_cap, bucket.tok_rate, bucket.req_cap, bucket.req_rate)

    # 切换时刻立即尝试放行（区间起点 = 切换时刻），随后继续推进
    post = drive(bucket, clock, [(0.0, *burst_at_switch), *after])
    assert_window_bound(post, *new)
