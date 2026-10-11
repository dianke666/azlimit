"""Property 16: 账本持久化（Validates: Requirements 26.1, 26.2, 26.3）。

∀ 运行序列：ledger_totals 累计值 == Σ runs.cost_usd（误差 < 1e-6）；
新运行的 CostGuard.from_store 从持久化花费开始累计。
"""

from __future__ import annotations

import asyncio
import math
import tempfile
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.corpus.store import CorpusStore
from loadtest.cost import AZURE_PROVIDERS, GEMINI_PROVIDERS, CostGuard, Usage
from loadtest.settings import CostSettings

PRICES = CostSettings().prices
AZURE_MODELS = ["gpt-4.1-nano", "gpt-4o-mini"]
GEMINI_MODELS = ["gemini-3.1-flash-lite", "gemini-3.8-flash"]

# (侧: 0=azure / 1=gemini, 模型下标, 输入, 输出, cached, grounding 次数, 无 usage?)
req = st.tuples(
    st.integers(0, 1),
    st.integers(0, 1),
    st.integers(0, 50_000),
    st.integers(0, 2_000),
    st.integers(0, 50_000),
    st.integers(0, 1),
    st.booleans(),
)
# 每个运行：请求列表 + 每隔多少个请求刷盘一次
run = st.tuples(st.lists(req, max_size=12), st.integers(1, 5))


async def _scenario(db: Path, runs) -> None:
    clock = FakeClock()
    for i, (reqs, flush_every) in enumerate(runs):
        store = await CorpusStore(db, clock).open()
        try:
            persisted_az = await store.spent_usd(AZURE_PROVIDERS)
            persisted_gm = await store.spent_usd(GEMINI_PROVIDERS)
            az = await CostGuard.from_store(store, 1000.0, 0.5, PRICES, AZURE_PROVIDERS)
            gm = await CostGuard.from_store(store, 1000.0, 0.5, PRICES, GEMINI_PROVIDERS)
            # 26.2：从持久化值开始
            assert math.isclose(az.spent, persisted_az, abs_tol=1e-9)
            assert math.isclose(gm.spent, persisted_gm, abs_tol=1e-9)
            assert math.isclose(sum(az.spent_by_model.values()), az.spent, abs_tol=1e-9)
            assert math.isclose(sum(gm.spent_by_model.values()), gm.spent, abs_tol=1e-9)

            run_id = f"run-{i}"
            await store.create_run(run_id, ("load", "warmup", "generate")[i % 3], "{}")
            for n, (side, mi, in_tok, out_tok, cached, units, no_usage) in enumerate(reqs, 1):
                g, model = (az, AZURE_MODELS[mi]) if side == 0 else (gm, GEMINI_MODELS[mi])
                r = await g.reserve(model, in_tok, out_tok, units if side == 1 else 0)
                assert r is not None
                await g.settle(r, None if no_usage else Usage(in_tok, out_tok, min(cached, in_tok)))
                if n % flush_every == 0:
                    await az.flush(store, run_id)
                    await gm.flush(store, run_id)
            await az.flush(store, run_id)
            await gm.flush(store, run_id)

            # 刷盘后：持久化值 == 内存权威值
            assert math.isclose(await store.spent_usd(AZURE_PROVIDERS), az.spent, abs_tol=1e-6)
            assert math.isclose(await store.spent_usd(GEMINI_PROVIDERS), gm.spent, abs_tol=1e-6)
            # 26.3：ledger_totals 累计 == Σ runs.cost_usd
            ledger_total = sum(t.total_usd for t in await store.ledger_totals())
            assert abs(ledger_total - await store.runs_cost_total()) < 1e-6
            rec = await store.get_run(run_id)
            assert abs(rec.cost_usd - (az.run_spent + gm.run_spent)) < 1e-6
            # 26.1：provider 只有三类，grounding 按次单列
            for t in await store.ledger_totals():
                assert t.provider in ("azure", "gemini", "gemini_grounding")
                if t.provider != "gemini_grounding":
                    assert t.units == 0
        finally:
            await store.close()


@settings(max_examples=25, deadline=None)
@given(runs=st.lists(run, min_size=1, max_size=4))
def test_ledger_totals_equal_runs_cost_and_guard_resumes(runs):
    with tempfile.TemporaryDirectory() as d:
        asyncio.run(_scenario(Path(d) / "corpus.db", runs))
