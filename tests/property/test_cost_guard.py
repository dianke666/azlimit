"""Property 15: 成本守卫（Validates: Requirements 25.2, 25.4, 25.6, 11.5）。

"reserve 返回 None 后调度器不再派发" 部分在调度器测试（15.6/15.7）中验证。
"""

from __future__ import annotations

import asyncio
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.corpus.store import LedgerWriteError
from loadtest.cost import (
    AZURE_PROVIDERS,
    GEMINI_PROVIDERS,
    GROUNDING_PRICE_KEY,
    CostGuard,
    Usage,
    cost_of,
    worst_case_token_cost,
)
from loadtest.settings import CostSettings, ModelPrice

PRICES = CostSettings().prices
AZURE_MODELS = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"]
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
TOL = 1e-9

# 每个请求: (模型下标, 输入 token, max_tokens, grounding 次数, 结局, usage 比例 a/b/c)
OUTCOMES = ["settle", "settle_none", "settle_zero", "release", "keep"]
request = st.tuples(
    st.integers(0, 2),
    st.integers(0, 200_000),
    st.integers(0, 20_000),
    st.integers(0, 2),
    st.sampled_from(OUTCOMES),
    st.floats(0, 1),
    st.floats(0, 1),
    st.floats(0, 1),
)


def _usage_within(r, fp: float, fc: float, fcache: float) -> Usage:
    """生成 ≤ 预留量的 usage：prompt ≤ in_tok × 1.02，completion ≤ max_out，cached ≤ prompt。"""
    prompt = math.floor(r.in_tok * 1.02 * fp)
    completion = math.floor(r.max_out * fc)
    return Usage(prompt, completion, math.floor(prompt * fcache))


def _check_invariants(g: CostGuard) -> None:
    assert g.reserved >= 0
    assert g.spent + g.reserved <= g.cap - g.margin + TOL
    assert g.spent <= g.cap + TOL
    if g.per_run_cap is not None:
        assert g.run_spent + g.reserved <= g.per_run_cap + TOL
    # 各 (provider, model) 花费之和 == 总花费；且只出现自身 provider
    assert math.isclose(sum(g.spent_by_model.values()), g.spent, rel_tol=1e-9, abs_tol=TOL)
    assert all(p in g.providers for p, _ in g.spent_by_model)
    # 待刷写账本与内存花费一致（本测试中历史花费为 0 且未刷盘）
    assert math.isclose(sum(e.cost_usd for e in g.pending_entries("r")), g.spent, rel_tol=1e-9, abs_tol=TOL)


async def _drive(g: CostGuard, models: list[str], reqs, grounding: bool) -> None:
    held = []
    for mi, in_tok, max_out, units, outcome, fp, fc, fcache in reqs:
        model = models[mi]
        before = g.reserved
        r = await g.reserve(model, in_tok, max_out, units if grounding else 0)
        if r is None:
            # 拒绝时状态不变，且确实会超出 cap − margin 或 per_run_cap
            assert g.reserved == before
            worst = g.worst_case(model, in_tok, max_out, units if grounding else 0)
            over_cap = g.spent + g.reserved + worst > g.cap - g.margin
            over_run = g.per_run_cap is not None and g.run_spent + g.reserved + worst > g.per_run_cap
            assert over_cap or over_run
        else:
            assert r.amount >= 0
            if outcome == "settle":
                charged = await g.settle(r, _usage_within(r, fp, fc, fcache))
                assert charged <= r.amount + TOL
            elif outcome == "settle_none":
                charged = await g.settle(r, None)
                assert math.isclose(charged, r.amount, rel_tol=1e-12, abs_tol=1e-15)
            elif outcome == "settle_zero":
                s0 = g.spent
                await g.settle_zero(r)
                assert g.spent == s0
            elif outcome == "release":
                s0 = g.spent
                await g.release(r)
                assert g.spent == s0
            else:
                held.append((r, fp, fc, fcache))
        _check_invariants(g)
    # 在途请求全部结算（按 usage ≤ 预留）
    for r, fp, fc, fcache in held:
        await g.settle(r, _usage_within(r, fp, fc, fcache))
        _check_invariants(g)
    assert g.outstanding == 0 and g.reserved == 0


@settings(max_examples=200, deadline=None)
@given(
    reqs=st.lists(request, max_size=40),
    cap=st.floats(0.6, 5.0),
    margin=st.sampled_from([0.0, 0.5]),
    per_run=st.one_of(st.none(), st.floats(0.01, 3.0)),
)
def test_azure_guard_never_exceeds_cap(reqs, cap, margin, per_run):
    g = CostGuard(cap, margin, PRICES, AZURE_PROVIDERS, per_run)
    asyncio.run(_drive(g, AZURE_MODELS, reqs, grounding=False))
    assert all(p == "azure" for p, _ in g.spent_by_model)


@settings(max_examples=200, deadline=None)
@given(
    reqs=st.lists(request, max_size=40),
    cap=st.floats(0.6, 5.0),
    per_run=st.one_of(st.none(), st.floats(0.01, 3.0)),
)
def test_gemini_guard_per_model_sums(reqs, cap, per_run):
    g = CostGuard(cap, 0.5, PRICES, GEMINI_PROVIDERS, per_run)
    asyncio.run(_drive(g, GEMINI_MODELS, reqs, grounding=True))
    # grounding 次数全部记在 gemini_grounding 下
    grounding_cost = g.spent_by_model.get(("gemini_grounding", GROUNDING_PRICE_KEY), 0.0)
    units = sum(e.units for e in g.pending_entries("r") if e.provider == "gemini_grounding")
    assert math.isclose(grounding_cost, units * 0.014, rel_tol=1e-9, abs_tol=TOL)
    assert all(e.units == 0 for e in g.pending_entries("r") if e.provider == "gemini")


# --------------------------------------------------------------------------- 示例测试


def test_cost_of_and_worst_case():
    p = ModelPrice(input_per_m=0.10, output_per_m=0.40, cached_input_per_m=0.025)
    assert cost_of(p, 1_000_000, 0) == pytest.approx(0.10)
    assert cost_of(p, 1_000_000, 1_000_000, cached_tokens=1_000_000) == pytest.approx(0.025 + 0.40)
    assert worst_case_token_cost(p, 1_000_000, 1_000_000) == pytest.approx(0.102 + 0.40)
    nocache = ModelPrice(input_per_m=0.10, output_per_m=0.40)
    assert cost_of(nocache, 1_000_000, 0, cached_tokens=500_000) == pytest.approx(0.10)
    assert cost_of(PRICES[GROUNDING_PRICE_KEY], 0, 0, units=3) == pytest.approx(0.042)


async def test_reserve_settle_release_basics():
    g = CostGuard(1.0, 0.5, PRICES, AZURE_PROVIDERS)
    r = await g.reserve("gpt-4.1-nano", 1_000_000, 0)
    assert r is not None and r.amount == pytest.approx(0.102)
    assert g.reserved == pytest.approx(0.102)
    assert await g.reserve("gpt-4.1-nano", 4_000_000, 0) is None  # 0.102 + 0.408 > 0.5
    await g.settle(r, Usage(prompt_tokens=1_000_000, completion_tokens=0, cached_tokens=400_000))
    assert g.spent == pytest.approx(0.06 + 0.01)
    assert g.reserved == 0
    await g.settle(r, None)  # 重复结算被忽略
    assert g.spent == pytest.approx(0.07)
    r2 = await g.reserve("gpt-4.1-nano", 100, 10)
    await g.release(r2)
    assert g.spent == pytest.approx(0.07) and g.reserved == 0
    assert g.remaining_ratio() == pytest.approx(0.93)


async def test_settle_none_charges_reservation_and_openai_usage_shape():
    g = CostGuard(10.0, 0.5, PRICES, AZURE_PROVIDERS)
    r = await g.reserve("gpt-4o-mini", 1000, 100)
    assert await g.settle(r, None) == pytest.approx(r.amount)

    class Details:
        cached_tokens = 1000

    class OAUsage:
        prompt_tokens = 1000
        completion_tokens = 10
        prompt_tokens_details = Details()

    r = await g.reserve("gpt-4o-mini", 1000, 100)
    charged = await g.settle(r, OAUsage())
    assert charged == pytest.approx((1000 * 0.075 + 10 * 0.60) / 1e6)


async def test_per_run_cap():
    g = CostGuard(100.0, 0.5, PRICES, AZURE_PROVIDERS, per_run_cap=0.2, spent=50.0)
    r = await g.reserve("gpt-4.1-nano", 1_000_000, 0)
    assert r is not None
    assert await g.reserve("gpt-4.1-nano", 1_000_000, 0) is None  # 0.102 × 2 > 0.2
    await g.settle(r, None)
    assert g.run_spent == pytest.approx(0.102) and g.spent == pytest.approx(50.102)


async def test_gemini_grounding_split():
    g = CostGuard(250.0, 0.5, PRICES, GEMINI_PROVIDERS)
    r = await g.reserve("gemini-3.1-flash-lite", 1000, 1000, units=1)
    assert r.unit_amount == pytest.approx(0.014)
    await g.settle(r, Usage(1000, 500))
    entries = {(e.provider, e.model): e for e in g.pending_entries("run1")}
    assert entries[("gemini", "gemini-3.1-flash-lite")].cost_usd == pytest.approx((1000 * 0.25 + 500 * 1.5) / 1e6)
    gr = entries[("gemini_grounding", GROUNDING_PRICE_KEY)]
    assert gr.units == 1 and gr.cost_usd == pytest.approx(0.014)


def test_estimate_run_cost_matches_design_example():
    g = CostGuard(180.0, 0.5, PRICES, AZURE_PROVIDERS)
    # 设计文档: 800k TPM × 20 分钟，≈ 5,300 次请求 → ≈ $1.66
    est = g.estimate_run_cost(800_000, 265, 20, "gpt-4.1-nano")
    assert est == pytest.approx(1.66, abs=0.01)
    assert g.remaining_runs(est) == math.floor(179.5 / est)


class _FailingStore:
    def __init__(self) -> None:
        self.fail = True
        self.written = []

    async def write_ledger(self, entries):
        if self.fail:
            raise LedgerWriteError("boom")
        self.written.extend(entries)


async def test_flush_keeps_pending_on_failure():
    g = CostGuard(10.0, 0.5, PRICES, AZURE_PROVIDERS)
    for _ in range(3):
        r = await g.reserve("gpt-4.1-nano", 1000, 10)
        await g.settle(r, Usage(1000, 10))
    store = _FailingStore()
    with pytest.raises(LedgerWriteError):
        await g.flush(store, "run1")
    assert len(g.pending_entries("run1")) == 1
    assert g.pending_entries("run1")[0].input_tokens == 3000
    store.fail = False
    assert await g.flush(store, "run1") == 1
    assert g.pending_entries("run1") == []
    assert store.written[0].cost_usd == pytest.approx(g.spent)
    assert await g.flush(store, "run1") == 0
