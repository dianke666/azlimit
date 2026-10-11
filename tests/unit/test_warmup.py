"""WarmupRunner / CorpusGenerator 单元测试（Requirements 3.5, 3.7, 4.5, 4.6, 5.1, 7.6, 26.1）。

使用假 GeminiClient（合成段落，不发起网络调用）、临时 CorpusStore 与 FakeClock。
"""

from __future__ import annotations

import re
from random import Random
from types import SimpleNamespace

import pytest

from loadtest.clock import FakeClock
from loadtest.corpus.gemini_client import GeminiBudgetExhausted, GeminiCallFailed, GeneratedText
from loadtest.corpus.planner import SupplyPlanner
from loadtest.corpus.prompts import load_blocklist
from loadtest.corpus.safety import SafetyFilter
from loadtest.corpus.store import CorpusStore
from loadtest.corpus.taxonomy import TopicSpec, TopicTaxonomy
from loadtest.corpus.warmup import CorpusGenerator, WarmupRunner
from loadtest.cost import GEMINI_PROVIDERS, CostGuard, Usage
from loadtest.settings import CorpusSettings, CostSettings, LoadSettings, ModelPrice

MODEL = "gemini-3.1-flash-lite"
# 固定测试单价（与默认价格解耦）：输入 0.10 / 输出 0.40 USD per 1M
PRICES = {**CostSettings().prices, MODEL: ModelPrice(input_per_m=0.10, output_per_m=0.40)}
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z0-9]+")


class UnitCounter:
    """测试用 token 计数：1 个 CJK 字符或 1 个英文/数字词 = 1 token。"""

    def count(self, text: str) -> int:
        return len(_CJK.findall(text)) + len(_WORD.findall(text))


def synth_text(spec: TopicSpec, rng: Random, *, lang: str | None = None) -> str:
    """合成满足过滤规则的段落：语言正确、4 个自然段、长度 = target_len、随机内容（不重复）。"""
    lang = lang or spec.lang
    n = spec.target_len
    date = "2024-05-01"
    units: list[str] = []
    if lang == "zh":
        chars = [chr(rng.randint(0x4E00, 0x9FA5)) for _ in range(n - (1 if spec.grounding else 0))]
        units = chars
    else:
        words = ["".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
                 for _ in range(n - (1 if spec.grounding else 0))]
        units = words
    k = len(units) // 4
    paras = [units[i * k:(i + 1) * k] if i < 3 else units[3 * k:] for i in range(4)]
    sep = "" if lang == "zh" else " "
    end = "。" if lang == "zh" else "."
    out = [sep.join(p) + end for p in paras]
    if spec.grounding:
        out[0] = date + sep + out[0]
    return "\n\n".join(out)


class FakeGeminiClient:
    """可编程假客户端：按脚本返回合成段落 / 错误，并在 CostGuard 中预留与结算。"""

    def __init__(self, guard: CostGuard, rng: Random, *, script=None, budget_after: int | None = None,
                 cost_tokens: tuple[int, int] = (300, 1000)) -> None:
        self.guard = guard
        self.rng = rng
        self.script = script  # callable(call_index, spec) -> kind
        self.budget_after = budget_after
        self.cost_tokens = cost_tokens
        self.calls = 0
        self.specs: list[TopicSpec] = []
        self.ok_texts: dict[str, str] = {}
        self.store = None

    async def generate(self, spec: TopicSpec, prompt: str) -> GeneratedText:
        assert prompt
        if self.budget_after is not None and self.calls >= self.budget_after:
            raise GeminiBudgetExhausted()
        r = await self.guard.reserve(MODEL, 400, 2048, units=int(spec.grounding))
        if r is None:
            raise GeminiBudgetExhausted()
        idx = self.calls
        self.calls += 1
        self.specs.append(spec)
        kind = self.script(idx, spec) if self.script else "ok"
        if kind == "error":
            await self.guard.settle(r, None)
            raise GeminiCallFailed("boom", kind="server", model=MODEL, attempts=4)
        await self.guard.settle(r, Usage(*self.cost_tokens))
        finish = "STOP"
        if kind == "safety":
            text, finish = "", "SAFETY"
        elif kind == "language":
            text = synth_text(spec, self.rng, lang="en" if spec.lang == "zh" else "zh")
        elif kind == "duplicate":
            text = self.ok_texts.get(spec.lang)
            if text is None:  # 该语言尚无已入库正文：先直接入库一份，再返回同一正文
                text = synth_text(spec, self.rng)
                assert self.store is not None
                await self.store.insert_text(
                    category=spec.category, subtopic=spec.subtopic, genre=spec.genre, lang=spec.lang,
                    token_len=UnitCounter().count(text), length_bucket=spec.length_bucket, text=text,
                    gen_model=MODEL, grounded=spec.grounding, sources=("https://news.example/1",),
                )
        else:
            text = synth_text(spec, self.rng)
            self.ok_texts[spec.lang] = text
        return GeneratedText(
            text=text, model=MODEL, input_tokens=self.cost_tokens[0], output_tokens=self.cost_tokens[1],
            finish_reason=finish, grounded=spec.grounding,
            sources=["https://news.example/1"] if spec.grounding else [], attempts=1,
            grounding_calls=int(spec.grounding),
        )


def make_cfg(target: int = 40) -> SimpleNamespace:
    return SimpleNamespace(
        load=LoadSettings(),
        corpus=CorpusSettings(target_size=target),
        gemini=SimpleNamespace(grounding_daily_limit=300),
    )


@pytest.fixture
async def store(tmp_path):
    clock = FakeClock()
    s = CorpusStore(tmp_path / "corpus.db", clock=clock)
    await s.open()
    try:
        yield s
    finally:
        await s.close()


def build(store, *, target=40, kind="warmup", concurrency=4, max_cost=10.0, cap=1e6, script=None,
          budget_after=None, categories=None, langs=None, limit=None, run_id="r1", seed=7):
    clock = store.clock
    tax = TopicTaxonomy.load()
    if kind == "generate":
        planner = SupplyPlanner(tax, make_cfg(target), categories=categories, langs=langs, limit=limit)
    else:
        planner = SupplyPlanner(tax, make_cfg(target), target_size=target)
    guard = CostGuard(cap, 0.0, PRICES, GEMINI_PROVIDERS)
    client = FakeGeminiClient(guard, Random(seed + 1), script=script, budget_after=budget_after)
    client.store = store
    safety = SafetyFilter(load_blocklist(), counter=UnitCounter())
    runner = WarmupRunner(
        None, store, client, planner, tax, safety, guard,
        clock=clock, rng=Random(seed), run_id=run_id, concurrency=concurrency, max_cost=max_cost,
        kind=kind, config_json="{}",
    )
    return runner, client, planner, guard, tax


async def rejection_rows(store) -> dict[str, int]:
    async with store.db.execute("SELECT reason, COUNT(*) AS n FROM rejections GROUP BY reason") as cur:
        return {r["reason"]: r["n"] async for r in cur}


# ---------------------------------------------------------------------------


async def test_fills_deficit_then_stops_filled(store) -> None:
    runner, client, planner, guard, _ = build(store, target=40, concurrency=6)
    rep = await runner.run()
    assert rep.stop_reason == "filled"
    assert rep.inserted == 40
    assert rep.deficit_at_start == 40
    assert client.calls == 40
    stats = await store.stats()
    assert stats.active == 40
    assert rep.inventory.active == 40
    assert sum(rep.by_lang.values()) == 40
    assert rep.by_lang["zh"] == 24 and rep.by_lang["en"] == 16  # 60/40
    run = await store.get_run("r1")
    assert run.kind == "warmup" and run.stop_reason == "filled" and run.passages_added == 40
    # 账本已刷写，runs.cost_usd == 本次花费
    assert run.cost_usd == pytest.approx(guard.run_spent) and rep.cost_usd == pytest.approx(guard.run_spent)
    assert guard.pending_entries("r1") == []
    # gen_model 记录实际模型（12.6）
    async with store.db.execute("SELECT DISTINCT gen_model FROM passages") as cur:
        assert [r[0] async for r in cur] == [MODEL]


async def test_zero_deficit_makes_no_calls(store) -> None:
    r1, *_ = build(store, target=40, run_id="r1")
    await r1.run()
    r2, client2, *_ = build(store, target=40, run_id="r2")
    rep = await r2.run()
    assert client2.calls == 0
    assert rep.stop_reason == "filled" and rep.inserted == 0 and rep.deficit_at_start == 0
    assert rep.inventory.active == 40
    assert (await store.get_run("r2")).passages_added == 0


async def test_budget_exhausted_stops(store) -> None:
    runner, client, planner, *_ = build(store, target=40, concurrency=3, budget_after=5)
    rep = await runner.run()
    assert rep.stop_reason == "budget"
    assert client.calls == 5
    assert rep.inserted == 5
    assert planner.inflight == 0
    assert (await store.get_run("r1")).stop_reason == "budget"


async def test_cost_guard_refusal_stops_with_budget(store) -> None:
    # 每次预留 ≈ 0.00086，cap 很小：CostGuard 很快拒绝预留
    runner, client, *_ = build(store, target=40, concurrency=1, cap=0.0027)
    rep = await runner.run()
    assert rep.stop_reason == "budget"
    # 实际结算 0.00043/次：0.00043·n + 0.00086 ≤ 0.0027 ⟹ 最多 5 次
    assert 1 <= client.calls <= 5
    assert client.calls < 40


async def test_max_cost_stops(store) -> None:
    # 每次实际花费 = 300 × 0.1/1M + 1000 × 0.4/1M = 0.00043
    runner, client, _, guard, _ = build(store, target=40, concurrency=1, max_cost=0.002)
    rep = await runner.run()
    assert rep.stop_reason == "max_cost"
    assert client.calls == 5  # 第 5 次后 ≥ 0.002
    assert guard.run_spent >= 0.002
    assert rep.inserted == 5


async def test_stop_event_stops(store) -> None:
    runner, client, *_ = build(store, target=40, concurrency=1)
    orig = client.generate

    async def gen(spec, prompt):
        out = await orig(spec, prompt)
        if client.calls == 3:
            runner.stop_event.set()
        return out

    client.generate = gen  # type: ignore[method-assign]
    rep = await runner.run()
    assert rep.stop_reason == "stopped" and client.calls == 3


async def test_generate_mode_only_given_categories_and_count(store) -> None:
    cats = ["cooking", "nature"]
    runner, client, planner, *_ = build(
        store, kind="generate", categories=cats, limit=7, concurrency=3, run_id="g1"
    )
    rep = await runner.run()
    assert rep.stop_reason == "limit"
    assert rep.inserted == 7
    assert {s.category for s in client.specs} <= set(cats)
    async with store.db.execute("SELECT DISTINCT category FROM passages") as cur:
        assert {r[0] async for r in cur} <= set(cats)
    assert (await store.stats()).active == 7
    run = await store.get_run("g1")
    assert run.kind == "generate" and run.passages_added == 7


async def test_generate_mode_with_grounding_category_and_lang(store) -> None:
    runner, client, planner, *_ = build(
        store, kind="generate", categories=["daily_news"], langs=["en"], limit=3, concurrency=2, run_id="g2"
    )
    rep = await runner.run()
    assert rep.inserted == 3
    assert all(s.grounding and s.lang == "en" for s in client.specs)
    assert planner.grounding_used == 3
    async with store.db.execute("SELECT grounded, sources, expires_at FROM passages") as cur:
        rows = [tuple(r) async for r in cur]
    assert len(rows) == 3 and all(r[0] == 1 and r[2] is not None for r in rows)


async def test_rejections_recorded(store) -> None:
    kinds = ["ok", "safety", "language", "duplicate", "error"]
    script = lambda i, spec: kinds[i] if i < len(kinds) else "ok"  # noqa: E731
    runner, client, planner, guard, tax = build(store, target=10, concurrency=1, script=script)
    first_safety_sub: list[str] = []
    orig = client.generate

    async def gen(spec, prompt):
        if client.calls == 1:
            first_safety_sub.append(spec.subtopic)
            first_safety_sub.append(tax.subtopic_weight(spec.subtopic))
        return await orig(spec, prompt)

    client.generate = gen  # type: ignore[method-assign]
    rep = await runner.run()
    assert rep.stop_reason == "filled"
    assert rep.inserted == 10
    assert rep.rejected_by_reason == {"safety_block": 1, "language": 1, "duplicate": 1}
    assert rep.errors == 1
    assert client.calls == 14
    assert rep.rejection_rate == pytest.approx(3 / 13)
    rows = await rejection_rows(store)
    assert rows == {"safety_block": 1, "language": 1, "duplicate": 1}  # error 不写 rejections
    # SAFETY → 子话题权重 × 0.5（5.1）
    sub, w0 = first_safety_sub
    assert tax.subtopic_weight(sub) == pytest.approx(w0 * 0.5)
    assert planner.inflight == 0


async def test_generator_returns_outcomes(store) -> None:
    tax = TopicTaxonomy.load()
    planner = SupplyPlanner(tax, make_cfg(10), target_size=10)
    guard = CostGuard(1e6, 0.0, PRICES, GEMINI_PROVIDERS)
    client = FakeGeminiClient(guard, Random(3), budget_after=1)
    gen = CorpusGenerator(client, store, planner, tax, SafetyFilter(load_blocklist(), counter=UnitCounter()),
                          run_id=None, clock=store.clock)
    planner.reset(await store.stats())
    spec = planner.next_spec(Random(1))
    out = await gen.generate_and_store(spec)
    assert out.status == "inserted" and out.passage_id is not None and out.model == MODEL
    spec2 = planner.next_spec(Random(2))
    with pytest.raises(GeminiBudgetExhausted):
        await gen.generate_and_store(spec2)
    assert planner.inflight == 0 and planner.inserted == 1


async def test_expired_passages_marked_at_start(store) -> None:
    # 先生成 grounded 段落，再把时钟推进 73h：warmup 开始时置为 expired（4.5）
    g, *_ = build(store, kind="generate", categories=["daily_news"], limit=2, concurrency=1, run_id="g0")
    await g.run()
    store.clock.advance(73 * 3600)
    r, client, *_ = build(store, target=5, concurrency=1, run_id="w1")
    await r.run()
    async with store.db.execute("SELECT COUNT(*) FROM passages WHERE status = 'expired'") as cur:
        (n,) = await cur.fetchone()
    assert n == 2
