"""Property 18: 隔离。

∀ 段落 p，收到 content_filter 一次后 p.status == 'quarantined'，quarantine 表记录原因
azure_content_filter，且此后同一运行（同一索引）及之后运行（从库重新加载的索引）的任何组装
结果都不包含 p。

**Validates: Requirements 24.1, 24.2**
"""
from __future__ import annotations

import asyncio
import hashlib
from random import Random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import CorpusStore, NewPassage
from loadtest.load.quarantine import QUARANTINE_REASON, extract_filter_categories, quarantine_passages
from tests.property.test_assembler_precision import make_assembler, synthetic_passages

_POOL = [p for p in synthetic_passages() if p.lang == "zh"][:30] + [p for p in synthetic_passages() if p.lang == "en"][:30]


async def _seed_store() -> CorpusStore:
    store = await CorpusStore(":memory:", clock=FakeClock()).open()
    for i, p in enumerate(_POOL):
        await store.insert(NewPassage(
            category=p.category, subtopic="s", genre="g", lang=p.lang, token_len=p.token_len,
            length_bucket=p.length_bucket, text=p.text,
            sha256=hashlib.sha256(p.text.encode()).hexdigest() + str(i), simhash=i * 7919,
            gen_model="test",
        ))
    return store


async def _scenario(seed: int, n_hits: int, detail: str | None) -> None:
    store = await _seed_store()
    try:
        idx = PassageIndex(await store.load_active_passages(), per_run_max_uses=50)
        asm = make_assembler(idx)
        rng = Random(seed)
        quarantined: set[int] = set()
        for _ in range(n_hits):
            req = asm.assemble(rng.randint(1500, 3500), rng)
            assert not (set(req.passage_ids) & quarantined)
            await quarantine_passages(store, idx, req.passage_ids, detail=detail, run_id="run-1")
            quarantined |= set(req.passage_ids)

        for pid in quarantined:
            row = await store.get_passage(pid)
            assert row["status"] == "quarantined"
            assert pid not in idx
        async with store.db.execute("SELECT passage_id, reason, detail, run_id FROM quarantine") as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        assert {r["passage_id"] for r in rows} == quarantined
        assert all(r["reason"] == QUARANTINE_REASON and r["detail"] == detail and r["run_id"] == "run-1"
                   for r in rows)

        # 同一运行：后续组装不包含隔离段落
        for _ in range(10):
            try:
                req = asm.assemble(rng.randint(1500, 3000), rng)
            except Exception:  # 语料可能被隔离到不足；不影响性质
                break
            assert not (set(req.passage_ids) & quarantined)

        # 之后运行：从库重新加载的索引不含隔离段落
        reloaded = await store.load_active_passages()
        assert not ({p.id for p in reloaded} & quarantined)
        assert len(reloaded) == len(_POOL) - len(quarantined)
    finally:
        await store.close()


@settings(max_examples=20, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n_hits=st.integers(min_value=1, max_value=4),
    detail=st.sampled_from([None, "hate", "violence,jailbreak"]),
)
def test_quarantine_property(seed: int, n_hits: int, detail: str | None) -> None:
    asyncio.run(_scenario(seed, n_hits, detail))


class _FailingStore:
    async def quarantine(self, *a, **kw):
        raise RuntimeError("db down")


def test_index_removed_even_if_db_write_fails() -> None:
    idx = PassageIndex(_POOL)
    ids = (_POOL[0].id, _POOL[1].id)
    with pytest.raises(RuntimeError):
        asyncio.run(quarantine_passages(_FailingStore(), idx, ids))
    assert all(pid not in idx for pid in ids)
    # 已移除的 id 同一运行内不可再加入
    assert idx.ingest(_POOL[0]) is False


def test_extract_filter_categories() -> None:
    err = {"error": {"code": "content_filter", "innererror": {
        "code": "ResponsibleAIPolicyViolation",
        "content_filter_result": {
            "hate": {"filtered": False, "severity": "safe"},
            "jailbreak": {"filtered": True, "detected": True},
            "self_harm": {"filtered": False, "severity": "safe"},
            "sexual": {"filtered": False, "severity": "safe"},
            "violence": {"filtered": True, "severity": "medium"},
        }}}}
    assert extract_filter_categories(err) == "violence,jailbreak"
    ok_body = {"choices": [{"finish_reason": "content_filter",
                            "content_filter_results": {"sexual": {"filtered": True, "severity": "high"}}}],
               "prompt_filter_results": [{"prompt_index": 0,
                                          "content_filter_results": {"hate": {"filtered": True}}}]}
    assert extract_filter_categories(ok_body) == "hate,sexual"
    import json
    assert extract_filter_categories(json.dumps(err)) == "violence,jailbreak"
    assert extract_filter_categories(json.dumps(err).encode()) == "violence,jailbreak"
    assert extract_filter_categories(None) is None
    assert extract_filter_categories("not json") is None
    assert extract_filter_categories({"error": {"code": "content_filter"}}) is None

    class R:
        body = err
    assert extract_filter_categories(R()) == "violence,jailbreak"
