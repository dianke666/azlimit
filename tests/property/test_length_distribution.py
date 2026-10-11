"""Feature: azure-gpt-load-tester, Property 2: 长度分布形状（Validates: Requirements 13.2, 13.3）。"""

from __future__ import annotations

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.load.distribution import sample_target_length
from loadtest.settings import LoadSettings

CFG = LoadSettings()
N = 20_000


@settings(deadline=None, max_examples=10)
@given(seed=st.integers(min_value=0, max_value=2**63 - 1))
def test_core_and_tail_shares(seed: int) -> None:
    rng = Random(seed)
    core = low = high = 0
    for _ in range(N):
        x = sample_target_length(rng, CFG)
        if 2500 <= x <= 3500:
            core += 1
        elif x < 2500:
            low += 1
        else:
            high += 1
    assert 0.78 <= core / N <= 0.82
    assert 0.09 <= low / N <= 0.11
    assert 0.09 <= high / N <= 0.11
