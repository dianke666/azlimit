"""Feature: azure-gpt-load-tester, Property 1: 目标长度范围（Validates: Requirements 13.1）。"""

from __future__ import annotations

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.load.distribution import LengthDistribution, sample_target_length
from loadtest.settings import LoadSettings

CFG = LoadSettings()


@settings(deadline=None, max_examples=300)
@given(seed=st.integers(min_value=0, max_value=2**63 - 1))
def test_sample_in_range_and_int(seed: int) -> None:
    rng = Random(seed)
    for _ in range(50):
        x = sample_target_length(rng, CFG)
        assert type(x) is int
        assert 1500 <= x <= 6000


@settings(deadline=None, max_examples=50)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_wrapper_matches_function(seed: int) -> None:
    a, b = Random(seed), Random(seed)
    dist = LengthDistribution()
    assert [dist.sample(a) for _ in range(20)] == [sample_target_length(b, CFG) for _ in range(20)]
