"""20.3 Mock Gemini warmup 集成测试（Requirements 3.7, 5.1, 6.4, 11.2, 11.3, 12.2）。

真实 build_corpus_pipeline（GeminiClient + 限流 + ModelChain + CostGuard + SafetyFilter + 去重 + CorpusStore），
后端为可编程假实现：注入 429、503、超时、SAFETY、语言错误、过短文本与重复文本。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from random import Random

import pytest
from google.genai import errors as genai_errors

from loadtest.clock import FakeClock
from loadtest.corpus.gemini_client import BackendResult
from loadtest.corpus.store import CorpusStore
from loadtest.corpus.warmup import build_corpus_pipeline
from loadtest.tokens import default_counter
from tests.unit.test_runner import make_settings

_ZH_POOL = "".join(chr(c) for c in range(0x4E00, 0x4E00 + 2500))


def _text(rng: Random, lang: str, target: int = 800, grounding: bool = False) -> str:
    c = default_counter()
    paras: list[str] = []
    total = 0
    while total < target or len(paras) < 3:
        if lang == "zh":
            p = "".join(rng.choice(_ZH_POOL) for _ in range(60)) + "。"
        else:
            p = " ".join("".join(rng.choice("abcdefghijklmnoprstuvw") for _ in range(rng.randint(3, 8)))
                         for _ in range(45)) + "."
        paras.append(p)
        total += c.count(p)
    if grounding:
        paras[0] = ("2026-10-10 " if lang == "en" else "2026年10月10日，") + paras[0]
    return "\n\n".join(paras)


class FakeBackend:
    def __init__(self, script=None, fail_model: str | None = None) -> None:
        self.rng = Random(42)
        self.n = 0
        self.script = script or {}
        self.fail_model = fail_model
        self.models: list[str] = []
        self.last_ok: dict[tuple[str, bool], str] = {}

    async def generate(self, model, prompt, grounding, timeout_s, **kw):
        self.n += 1
        self.models.append(model)
        lang = "zh" if "简体中文" in prompt else "en"
        if self.fail_model == model:
            raise genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
        kind = self.script.get(self.n, "ok")
        src = ("https://example.com/news",) if grounding else ()
        if kind == "429":
            raise genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}})
        if kind == "503":
            raise genai_errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
        if kind == "timeout":
            raise asyncio.TimeoutError()
        if kind == "safety":
            return BackendResult("", "SAFETY", 100, 0)
        if kind == "wrong_lang":
            return BackendResult(_text(self.rng, "en" if lang == "zh" else "zh"), "STOP", 100, 900, src)
        if kind == "short":
            return BackendResult("太短。\n\n很短。\n\nshort.", "STOP", 100, 10, src)
        key = (lang, grounding)
        if kind == "dup" and key in self.last_ok:
            return BackendResult(self.last_ok[key], "STOP", 100, 900, src)
        if kind == "dup":
            self.script[self.n + 1] = "dup"  # 该语言尚无可复用文本：推迟到下一次调用
        text = _text(self.rng, lang, grounding=grounding)
        self.last_ok[key] = text
        return BackendResult(text, "STOP", 100, 900, src)


async def _rejections(store) -> dict[str, int]:
    async with store.db.execute("SELECT reason, COUNT(*) AS n FROM rejections GROUP BY reason") as cur:
        return {r["reason"]: r["n"] async for r in cur}


@pytest.fixture
async def store(tmp_path):
    s = await CorpusStore(tmp_path / "corpus.db", FakeClock()).open()
    try:
        yield s
    finally:
        await s.close()


async def test_warmup_with_injected_faults(tmp_path: Path, store) -> None:
    settings = make_settings(tmp_path)
    script = {1: "429", 2: "503", 3: "timeout", 5: "safety", 6: "wrong_lang", 7: "short", 9: "dup"}
    backend = FakeBackend(script)
    runner = await build_corpus_pipeline(
        settings, store, store.clock, run_id="w1", kind="warmup", rng=Random(1),
        target=24, concurrency=1, max_cost=5.0, backend=backend,
    )
    rep = await runner.run()
    assert rep.stop_reason == "filled", rep
    assert rep.inserted == 24
    st = await store.stats()
    assert st.active == 24 and st.by_lang.get("zh", 0) > st.by_lang.get("en", 0)

    rej = await _rejections(store)
    assert rej.get("safety_block", 0) >= 1
    assert rej.get("language", 0) >= 1
    assert rej.get("length", 0) + rej.get("structure", 0) >= 1
    assert rej.get("duplicate", 0) >= 1
    # 429 / 503 / 超时被重试（同一任务 4 次尝试内成功），不产生拒绝
    assert backend.n >= 24 + 7
    # 429 后 Gemini 节流器进入过 HOLD
    assert runner.client.limiter.throttle.stats.throttle_events >= 1
    # 账本写入且 runs 记录完成
    run = await store.get_run("w1")
    assert run is not None and run.stop_reason == "filled" and run.cost_usd > 0


async def test_primary_model_failover(tmp_path: Path, store) -> None:
    settings = make_settings(tmp_path)
    backend = FakeBackend(fail_model="gemini-3.1-flash-lite")
    runner = await build_corpus_pipeline(
        settings, store, store.clock, run_id="w2", kind="warmup", rng=Random(2),
        target=6, concurrency=1, max_cost=5.0, backend=backend,
    )
    rep = await runner.run()
    assert rep.inserted == 6
    # 主模型连续 5 次（重试耗尽后）失败 → 切换到 gemini-3.5-flash-lite（12.2）
    assert backend.models.count("gemini-3.1-flash-lite") == 5 * 4
    async with store.db.execute("SELECT DISTINCT gen_model FROM passages") as cur:
        models = {r[0] async for r in cur}
    assert models == {"gemini-3.5-flash-lite"}
    assert any("switching to gemini-3.5-flash-lite" in w for w in runner.client.chain.warnings)
