"""Feature: azure-gpt-load-tester, Property 27: 模型链切换
（Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5）。"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.corpus.model_chain import DEFAULT_MODEL_CHAIN, ModelChain

FLASH = "gemini-3.8-flash"
PRIMARY = DEFAULT_MODEL_CHAIN[0]

op = st.one_of(
    st.tuples(st.just("fail")),
    st.tuples(st.just("ok")),
    st.tuples(st.just("adv"), st.floats(min_value=0.0, max_value=400.0)),
    st.tuples(st.just("budget"), st.sampled_from([1.0, 0.5, 0.25, 0.2, 0.19, 0.05, 0.0])),
)


@settings(deadline=None, max_examples=400)
@given(ops=st.lists(op, min_size=1, max_size=120))
def test_model_chain_property(ops: list[tuple]) -> None:
    budget = {"r": 1.0}
    chain = ModelChain(budget_ratio_fn=lambda: budget["r"])
    now = 0.0
    consec_fail = {m: 0 for m in DEFAULT_MODEL_CHAIN}  # 测试侧观测的连续失败
    probe_ok_run = 0  # 探测期间主模型的连续成功
    off_primary_at: float | None = None  # 活动模型离开主模型 / 探测失败的时刻
    prev_active = chain.active
    prev_cur = chain.current(now)

    for o in ops:
        kind = o[0]
        was_probing = chain.probing
        if kind == "adv":
            now += o[1]
        elif kind == "budget":
            budget["r"] = o[1]
        else:
            m = chain.current(now)
            was_probing = chain.probing
            prev_active = chain.active
            if kind == "fail":
                consec_fail[m] += 1
                chain.report_failure(m, now)
                if was_probing and m == PRIMARY:
                    probe_ok_run = 0
                    # 探测失败 → 回到备用并重新计时
                    assert not chain.probing
                    assert chain.active == prev_active != PRIMARY
                    off_primary_at = now
            else:
                consec_fail[m] = 0
                chain.report_success(m, now)
                assert chain.failures[m] == 0  # 12.3
                if was_probing and m == PRIMARY:
                    probe_ok_run += 1
                    if probe_ok_run >= 3:  # 12.4：连续 3 次成功恢复主模型
                        assert chain.active == PRIMARY and not chain.probing
                        probe_ok_run = 0
                    else:
                        assert chain.active == prev_active

        cur = chain.current(now)
        idx_prev = DEFAULT_MODEL_CHAIN.index(prev_active)
        idx_new = DEFAULT_MODEL_CHAIN.index(chain.active)

        # 12.1：始终 ∈ 列表
        assert cur in DEFAULT_MODEL_CHAIN and chain.active in DEFAULT_MODEL_CHAIN
        # 12.5：预算 < 20% 不选 2.5-flash
        if budget["r"] < 0.2:
            assert cur != FLASH and chain.active != FLASH
        # 12.2：活动模型前进只发生在该模型连续失败 ≥ 5 之后，且只前进一位
        if idx_new > idx_prev:
            assert kind == "fail"
            assert idx_new == idx_prev + 1
            assert consec_fail[prev_active] >= 5
            assert not (budget["r"] < 0.2 and chain.active == FLASH)
        if prev_active == PRIMARY and chain.active != PRIMARY:
            off_primary_at = now
        # 12.4：探测（下一次尝试发往主模型）只在离开主模型 ≥ 600s 后发生
        if chain.active != PRIMARY and cur == PRIMARY and not was_probing:
            assert off_primary_at is not None and now - off_primary_at >= 600.0
        if chain.active != PRIMARY and not chain.probing:
            probe_ok_run = 0
        # 恢复主模型只能经由 3 次探测成功（或预算回退，默认链中不会回退到主模型）
        if chain.active == PRIMARY and prev_active != PRIMARY:
            assert kind == "ok" and was_probing
        prev_active = chain.active
        prev_cur = cur

    assert prev_cur in DEFAULT_MODEL_CHAIN


def test_failover_sequence_example() -> None:
    chain = ModelChain()
    for _ in range(4):
        chain.report_failure(PRIMARY, 0.0)
    assert chain.current(0.0) == PRIMARY
    chain.report_failure(PRIMARY, 1.0)
    assert chain.current(1.0) == "gemini-3.5-flash-lite"
    assert chain.warnings
    # 未到 600s 不探测
    assert chain.current(600.9) == "gemini-3.5-flash-lite"
    assert chain.current(601.0) == PRIMARY and chain.probing
    chain.report_failure(PRIMARY, 601.0)  # 探测失败 → 回备用并重新计时
    assert chain.current(1200.0) == "gemini-3.5-flash-lite"
    assert chain.current(1201.0) == PRIMARY
    for _ in range(3):
        chain.report_success(PRIMARY, 1202.0)
    assert chain.active == PRIMARY and not chain.probing


def test_success_resets_failures_and_last_model_stays() -> None:
    chain = ModelChain(failback_after_s=1e9)
    for _ in range(4):
        chain.report_failure(PRIMARY, 0.0)
    chain.report_success(PRIMARY, 0.0)
    for _ in range(4):
        chain.report_failure(PRIMARY, 0.0)
    assert chain.active == PRIMARY
    for m in DEFAULT_MODEL_CHAIN:
        for _ in range(5):
            chain.report_failure(chain.current(0.0), 0.0)
    assert chain.active == FLASH
    n = len(chain.warnings)
    for _ in range(5):
        chain.report_failure(FLASH, 0.0)
    assert chain.active == FLASH and len(chain.warnings) == n + 1


def test_low_budget_never_switches_to_flash() -> None:
    chain = ModelChain(budget_ratio_fn=lambda: 0.1, failback_after_s=1e9)
    for _ in range(5):
        chain.report_failure(PRIMARY, 0.0)
    assert chain.active == "gemini-3.5-flash-lite"
    for _ in range(10):
        chain.report_failure("gemini-3.5-flash-lite", 0.0)
    assert chain.current(0.0) == "gemini-3.5-flash-lite"
    assert any("expensive" in w for w in chain.warnings)
