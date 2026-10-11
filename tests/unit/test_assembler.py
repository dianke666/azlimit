"""RequestAssembler 单元测试（Requirements 9.3, 9.5, 14.x, 15.x）。"""
from __future__ import annotations

from random import Random

import pytest

from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage
from loadtest.load.assembler import (
    SEP,
    CorpusExhausted,
    RequestAssembler,
    TargetUnreachable,
    build_messages,
    choose_lang,
    choose_passage_count,
)
from loadtest.load.models import PlannedRequest
from loadtest.settings import LoadSettings
from loadtest.tokens import default_counter
from tests.property.test_assembler_precision import catalog, make_assembler, synthetic_passages


def test_choose_passage_count_tiers() -> None:
    rng = Random(0)
    assert {choose_passage_count(1999, rng) for _ in range(200)} == {1, 2}
    assert {choose_passage_count(2000, rng) for _ in range(200)} == {2, 3}
    assert {choose_passage_count(3999, rng) for _ in range(200)} == {2, 3}
    assert {choose_passage_count(4000, rng) for _ in range(200)} == {3, 4}


def test_choose_lang_extremes() -> None:
    rng = Random(0)
    assert choose_lang(rng, {"zh": 1.0, "en": 0.0}) == "zh"
    assert choose_lang(rng, {"zh": 0.0, "en": 1.0}) == "en"


def test_build_messages_positions() -> None:
    task = catalog().for_lang("zh")[0]
    before = build_messages(task, ["A", "B"], "before")
    after = build_messages(task, ["A", "B"], "after")
    assert before[0] == {"role": "system", "content": task.system_prompt}
    assert before[1]["content"] == f"{task.instruction}{SEP}A{SEP}B"
    assert after[1]["content"] == f"A{SEP}B{SEP}{task.instruction}"


def test_planned_request_fields_and_marks() -> None:
    idx = PassageIndex(synthetic_passages())
    asm = RequestAssembler(idx, catalog(), default_counter(), LoadSettings())
    req = asm.assemble(3000, Random(42))
    assert isinstance(req, PlannedRequest)
    assert len(req.request_id) == 36
    assert req.max_tokens == 40
    assert req.target_tokens == 3000
    assert req.task_id in {t.id for t in catalog().for_lang(req.lang)}
    assert idx.run_uses == {pid: 1 for pid in req.passage_ids}


def test_rejects_bad_lang_ratio() -> None:
    with pytest.raises(ValueError):
        RequestAssembler(PassageIndex(), catalog(), lang_ratio={"zh": 1.0})


def test_empty_index_raises_corpus_exhausted() -> None:
    asm = make_assembler(PassageIndex())
    with pytest.raises(CorpusExhausted):
        asm.assemble(2000, Random(0))


def test_filler_appended_when_single_passage_short() -> None:
    """只有一个 L 段落能被初始选中时（k=1），不足部分由补足片段填充。"""
    zh = [p for p in synthetic_passages() if p.lang == "zh"]
    pool = [p for p in zh if p.token_len < 800][:5] + [p for p in zh if p.token_len >= 1350][:1]
    found = False
    for s in range(40):
        asm = make_assembler(PassageIndex(pool), lang_ratio={"zh": 1.0, "en": 0.0})
        req = asm.assemble(1900, Random(s))
        assert abs(req.input_tokens - 1900) <= 20
        if req.filler_count > 0:
            found = True
    assert found


def test_overflow_trims_last_part_at_sentence_or_token_boundary() -> None:
    # 只有 L 段落，目标 1500 时一段即超出预算，需要裁剪
    pool = [p for p in synthetic_passages() if p.lang == "en" and p.token_len >= 1350][:10]
    asm = make_assembler(PassageIndex(pool), lang_ratio={"zh": 0.0, "en": 1.0})
    req = asm.assemble(1500, Random(9))
    assert abs(req.input_tokens - 1500) <= 20
    src = next(p for p in pool if p.id == req.passage_ids[-1]).text
    user = req.messages[1]["content"]
    parts = user.split(SEP)
    parts = parts[1:] if req.instruction_position == "before" else parts[:-1]
    assert len(parts) == len(req.passage_ids)
    assert src.startswith(parts[-1]) and len(parts[-1]) < len(src)  # 最后一段被裁剪


def test_target_unreachable_is_corpus_exhausted_subclass_and_no_marks() -> None:
    pool = [Passage(id=i, category="c", lang="zh", token_len=600, use_count=0, expires_at=None,
                    text="测试句子。" * 200, length_bucket="S") for i in range(1, 4)]
    idx = PassageIndex(pool)
    asm = make_assembler(idx, lang_ratio={"zh": 1.0, "en": 0.0})
    with pytest.raises(TargetUnreachable) as ei:
        asm.assemble(6000, Random(0))
    assert isinstance(ei.value, CorpusExhausted)
    assert ei.value.lang == "zh" and ei.value.target == 6000
    assert idx.run_uses == {}


def test_expired_passages_never_used() -> None:
    pool = list(synthetic_passages())
    expired_ids = {p.id for p in pool[::3]}
    pool = [Passage(**{**p.__dict__, "expires_at": 100.0}) if p.id in expired_ids else p for p in pool]
    idx = PassageIndex(pool, clock=lambda: 50.0)
    asm = make_assembler(idx)
    rng = Random(2)
    for _ in range(20):
        req = asm.assemble(rng.randint(1500, 6000), rng, now=200.0)
        assert not (set(req.passage_ids) & expired_ids)
