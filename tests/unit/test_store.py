"""CorpusStore 与 Deduplicator 单元测试（Requirements 4.1, 4.5, 5.10, 6.1–6.4, 9.4, 26.5, 28.4）。"""
from __future__ import annotations

import enum
import json
import random
from dataclasses import dataclass, field
from datetime import timedelta

import pytest

from loadtest.clock import FakeClock
from loadtest.corpus.dedup import (
    Deduplicator,
    hamming,
    normalize,
    sha256_of,
    simhash64,
)
from loadtest.corpus.store import (
    LEDGER_MAX_RETRIES,
    CorpusStore,
    LedgerEntry,
    LedgerWriteError,
    NewPassage,
    from_iso,
    to_iso,
)

EN_VOCAB = (
    "market data cloud model server latency budget team policy river city energy network "
    "report growth signal update future design history science health travel music garden "
    "engine planet window coffee winter summer language memory system ocean forest bridge"
).split()


def en_text(seed: int, n_words: int = 600) -> str:
    rng = random.Random(seed)
    paras = []
    for _ in range(4):
        words = [rng.choice(EN_VOCAB) for _ in range(n_words // 4)]
        paras.append(" ".join(words) + ".")
    return "\n\n".join(paras)


def zh_text(seed: int, n_chars: int = 1200) -> str:
    rng = random.Random(seed)
    chars = [chr(rng.randint(0x4E00, 0x4E00 + 800)) for _ in range(n_chars)]
    out = []
    for i, ch in enumerate(chars, 1):
        out.append(ch)
        if i % 37 == 0:
            out.append("，")
        if i % 300 == 0:
            out.append("。\n")
    return "".join(out)


def text_kwargs(text: str, lang: str = "en", **over):
    kw = dict(
        category="tech",
        subtopic="cloud",
        genre="news",
        lang=lang,
        token_len=800,
        length_bucket="M",
        text=text,
        gen_model="gemini-2.5-flash",
    )
    kw.update(over)
    return kw


def new_passage(text: str, *, grounded: bool = False, lang: str = "en") -> NewPassage:
    return NewPassage(
        category="tech",
        subtopic="cloud",
        genre="news",
        lang=lang,  # type: ignore[arg-type]
        token_len=800,
        length_bucket="M",
        text=text,
        sha256=sha256_of(text),
        simhash=simhash64(text, lang),  # type: ignore[arg-type]
        gen_model="gemini-2.5-flash",
        grounded=grounded,
        sources=("https://example.com/a",) if grounded else (),
    )


@pytest.fixture
async def store(tmp_path, fake_clock: FakeClock):
    s = CorpusStore(tmp_path / "corpus.db", fake_clock, ledger_retry_delay_s=0.5)
    await s.open()
    try:
        yield s
    finally:
        await s.close()


# --------------------------------------------------------------------------- expires_at / expire_due


async def test_grounded_expires_72h_after_created(store: CorpusStore, fake_clock: FakeClock):
    pid = await store.insert(new_passage(en_text(1), grounded=True))
    row = await store.get_passage(pid)
    assert row["grounded"] == 1
    assert row["created_at"] == to_iso(fake_clock.wall())
    assert from_iso(row["expires_at"]) - from_iso(row["created_at"]) == pytest.approx(72 * 3600)
    assert json.loads(row["sources"]) == ["https://example.com/a"]
    assert row["gen_model"] == "gemini-2.5-flash"


async def test_non_grounded_expires_null(store: CorpusStore):
    pid = await store.insert(new_passage(en_text(2)))
    row = await store.get_passage(pid)
    assert row["grounded"] == 0
    assert row["expires_at"] is None


async def test_expire_due_marks_expired(store: CorpusStore, fake_clock: FakeClock):
    g = await store.insert(new_passage(en_text(3), grounded=True))
    n = await store.insert(new_passage(en_text(4)))

    fake_clock.advance(timedelta(hours=72).total_seconds() - 1)
    assert await store.expire_due() == 0
    assert {p.id for p in await store.load_active_passages()} == {g, n}

    fake_clock.advance(1)
    # 到期但尚未置位：已从 active 索引中排除，stats 计为 expired
    assert {p.id for p in await store.load_active_passages()} == {n}
    st = await store.stats()
    assert (st.active, st.expired) == (1, 1)

    assert await store.expire_due() == 1
    assert (await store.get_passage(g))["status"] == "expired"
    assert (await store.get_passage(n))["status"] == "active"
    assert await store.expire_due() == 0


# --------------------------------------------------------------------------- rejections


async def test_rejections_table_has_no_text_column(store: CorpusStore):
    async with store.db.execute("PRAGMA table_info(rejections)") as cur:
        cols = {r["name"] async for r in cur}
    assert cols == {"id", "run_id", "category", "lang", "model", "reason", "created_at"}


async def test_rejection_does_not_store_body(store: CorpusStore):
    body = en_text(5)
    await store.insert_text(**text_kwargs(body))
    res = await store.insert_text(**text_kwargs(body), run_id=None)
    assert res.status == "duplicate"
    await store.record_rejection("tech", "en", "gemini-2.5-pro", "length", run_id="r1")
    async with store.db.execute("SELECT * FROM rejections") as cur:
        rows = [dict(r) async for r in cur]
    assert len(rows) == 2
    snippet = body[:40]
    for r in rows:
        assert all(snippet not in str(v) for v in r.values())
    counts = await store.rejection_counts(0)
    assert counts == {"duplicate": 1, "length": 1}


# --------------------------------------------------------------------------- use_count


async def test_flush_use_counts_accumulates(store: CorpusStore):
    a = await store.insert(new_passage(en_text(6)))
    b = await store.insert(new_passage(en_text(7)))
    await store.flush_use_counts({a: 3, b: 0})
    await store.flush_use_counts({a: 2, b: 5})
    await store.flush_use_counts({})
    assert (await store.get_passage(a))["use_count"] == 5
    assert (await store.get_passage(b))["use_count"] == 5
    st = await store.stats()
    assert st.by_use_count == {5: 2}


# --------------------------------------------------------------------------- ledger


async def test_ledger_write_fails_after_3_retries(store: CorpusStore, fake_clock: FakeClock):
    entry = LedgerEntry("no-such-run", "azure", "gpt-4o", 100, 10, 0.01)
    with pytest.raises(LedgerWriteError):
        await store.write_ledger([entry])
    # 首次失败 + 3 次重试，重试前各 sleep 一次
    assert fake_clock.sleeps == [0.5] * LEDGER_MAX_RETRIES
    async with store.db.execute("SELECT COUNT(*) FROM ledger") as cur:
        (n,) = await cur.fetchone()
    assert n == 0
    # 连接仍可用
    await store.create_run("ok", "load", "{}")
    await store.write_ledger([LedgerEntry("ok", "azure", "gpt-4o", 1, 1, 0.5)])
    assert await store.runs_cost_total() == pytest.approx(0.5)


async def test_ledger_totals_equal_runs_cost(store: CorpusStore):
    await store.create_run("r1", "load", "{}")
    await store.create_run("r2", "warmup", "{}")
    await store.write_ledger(
        [
            LedgerEntry("r1", "azure", "gpt-4o", 1000, 100, 0.125),
            LedgerEntry("r1", "azure", "gpt-4o", 500, 50, 0.0625),
            LedgerEntry("r2", "gemini", "gemini-2.5-flash", 200, 2000, 0.3),
        ]
    )
    await store.write_ledger([LedgerEntry("r2", "gemini_grounding", "gemini-2.5-flash", 0, 0, 0.07, units=2)])
    totals = await store.ledger_totals()
    assert sum(t.total_usd for t in totals) == pytest.approx(await store.runs_cost_total())
    assert (await store.get_run("r1")).cost_usd == pytest.approx(0.1875)
    assert (await store.get_run("r2")).cost_usd == pytest.approx(0.37)
    assert await store.spent_usd(["gemini", "gemini_grounding"]) == pytest.approx(0.37)
    assert await store.grounding_calls_today() == 2


async def test_ledger_scoped_by_account(store: CorpusStore):
    """换 GCP 账号：按 account 分别累计；不传 account 时为全部历史。"""
    from loadtest.cost import GEMINI_PROVIDERS, CostGuard
    from loadtest.settings import CostSettings

    await store.create_run("w1", "warmup", "{}")
    await store.write_ledger([
        LedgerEntry("w1", "gemini", "gemini-3.1-flash-lite", 100, 100, 200.0, account="proj-a"),
        LedgerEntry("w1", "gemini_grounding", "google-search-grounding", 0, 0, 1.0, units=10, account="proj-a"),
        LedgerEntry("w1", "gemini", "gemini-3.1-flash-lite", 100, 100, 5.0, account="proj-b"),
        LedgerEntry("w1", "azure", "gpt-4.1-nano", 100, 10, 3.0),
    ])
    gp = list(GEMINI_PROVIDERS)
    assert await store.spent_usd(gp) == pytest.approx(206.0)
    assert await store.spent_usd(gp, "proj-a") == pytest.approx(201.0)
    assert await store.spent_usd(gp, "proj-c") == 0.0
    assert await store.ledger_accounts(gp) == [("proj-a", pytest.approx(201.0)), ("proj-b", pytest.approx(5.0))]

    cs = CostSettings()
    a = await CostGuard.gemini_from_settings(store, cs, "proj-a")
    c = await CostGuard.gemini_from_settings(store, cs, "proj-c")  # 新账号：从 0 计
    assert a.spent == pytest.approx(201.0) and a.account == "proj-a"
    assert c.spent == 0.0 and c.account == "proj-c"
    # 显式 gemini_budget_account 优先于 project
    o = await CostGuard.gemini_from_settings(store, CostSettings(gemini_budget_account="proj-b"), "proj-a")
    assert o.spent == pytest.approx(5.0)

    # 新账号的结算写入自己的 account
    r = await c.reserve("gemini-3.1-flash-lite", 1000, 1000)
    from loadtest.cost import Usage
    await c.settle(r, Usage(1000, 500))
    await c.flush(store, "w1")
    assert await store.spent_usd(gp, "proj-c") == pytest.approx(c.spent)
    assert await store.spent_usd(gp, "proj-a") == pytest.approx(201.0)


async def test_old_ledger_migrated_with_account_column(tmp_path, fake_clock: FakeClock):
    """旧库（ledger 无 account 列）打开时自动迁移，旧行归属 ''。"""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        """CREATE TABLE runs (run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, started_at TEXT NOT NULL,
               ended_at TEXT, stop_reason TEXT, config_json TEXT NOT NULL, requests_ok INTEGER DEFAULT 0,
               requests_err INTEGER DEFAULT 0, throttled INTEGER DEFAULT 0, probes INTEGER DEFAULT 0,
               throttle_json TEXT, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0,
               passages_added INTEGER DEFAULT 0, cost_usd REAL DEFAULT 0);
           CREATE TABLE ledger (id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
               provider TEXT NOT NULL, model TEXT NOT NULL, input_tokens INTEGER NOT NULL,
               output_tokens INTEGER NOT NULL, units INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL,
               flushed_at TEXT NOT NULL);
           INSERT INTO runs (run_id, kind, started_at, config_json, cost_usd) VALUES ('r0','warmup','x','{}',2.0);
           INSERT INTO ledger (run_id, provider, model, input_tokens, output_tokens, cost_usd, flushed_at)
               VALUES ('r0','gemini','gemini-2.5-flash-lite',1,1,2.0,'x');"""
    )
    con.commit()
    con.close()
    s = await CorpusStore(db, fake_clock).open()
    try:
        assert await s.spent_usd(["gemini"]) == pytest.approx(2.0)
        assert await s.spent_usd(["gemini"], "") == pytest.approx(2.0)
        assert await s.spent_usd(["gemini"], "proj-new") == 0.0
        await s.write_ledger([LedgerEntry("r0", "gemini", "gemini-3.1-flash-lite", 1, 1, 1.0, account="proj-new")])
        assert await s.spent_usd(["gemini"], "proj-new") == pytest.approx(1.0)
    finally:
        await s.close()


# --------------------------------------------------------------------------- runs


class _State(enum.Enum):
    NORMAL = "normal"
    HOLD = "hold"


@dataclass
class _Stats:
    throttle_events: int = 2
    probes_sent: int = 3
    time_in_state: dict = field(default_factory=lambda: {_State.NORMAL: 10.5, _State.HOLD: 4.0})
    timeline: list = field(default_factory=lambda: [(0.0, _State.HOLD), (13.0, _State.NORMAL)])


async def test_runs_probes_and_throttle_json_roundtrip(store: CorpusStore):
    await store.create_run("r1", "load", '{"tpm": 1000}')
    await store.finish_run(
        "r1", "duration", throttle_stats=_Stats(), requests_ok=10, throttled=2, probes=3
    )
    run = await store.get_run("r1")
    assert run.stop_reason == "duration"
    assert run.ended_at is not None
    assert (run.requests_ok, run.throttled, run.probes) == (10, 2, 3)
    assert run.throttle_stats == {
        "throttle_events": 2,
        "probes_sent": 3,
        "time_in_state": {"normal": 10.5, "hold": 4.0},
        "timeline": [[0.0, "hold"], [13.0, "normal"]],
    }
    # dict 快照原样往返
    snap = {"probe_ok": 1, "timeline": [[1.5, "probe"]]}
    await store.update_run("r1", throttle_stats=snap)
    assert (await store.get_run("r1")).throttle_stats == snap
    assert (await store.latest_run("load")).run_id == "r1"
    with pytest.raises(ValueError):
        await store.update_run("r1", kind="warmup")


# --------------------------------------------------------------------------- dedup helpers


def test_normalize_strips_whitespace_and_unicode_punct():
    assert normalize(" Hello,\tworld!\n“引号”，句号。 ") == "Helloworld引号句号"
    assert sha256_of("a b, c.") == sha256_of("abc")
    assert sha256_of("abc") != sha256_of("abd")


def test_simhash_properties():
    t = en_text(10)
    assert simhash64(t) == simhash64(t)
    assert 0 <= simhash64(t) < 1 << 64
    assert hamming(simhash64(t), simhash64(t.replace(" ", "  ").replace(".", "!"))) == 0
    assert hamming(simhash64(en_text(10)), simhash64(en_text(11))) > 3
    assert hamming(simhash64(zh_text(10), "zh"), simhash64(zh_text(11), "zh")) > 3
    assert hamming(-1, (1 << 64) - 1) == 0


def test_in_memory_deduplicator():
    dd = Deduplicator()
    a = en_text(20)
    dd.add(a, passage_id=1)
    assert dd.is_duplicate(a)
    assert dd.check("  " + a.replace(".", ";") + "\n").kind == "exact"
    near = dd.check(a + " garden")
    assert near is not None and near.kind == "near" and near.passage_id == 1
    assert not dd.is_duplicate(en_text(21))


# --------------------------------------------------------------------------- dedup in store


async def test_dedup_exact(store: CorpusStore):
    t = en_text(30)
    first = await store.insert_text(**text_kwargs(t))
    assert first.inserted and first.id is not None
    again = await store.insert_text(**text_kwargs(t, gen_model="gemini-2.5-pro"), run_id=None)
    assert again.status == "duplicate" and again.id is None
    assert again.match.kind == "exact" and again.match.passage_id == first.id
    assert (await store.stats()).active == 1
    async with store.db.execute("SELECT category, lang, model, reason FROM rejections") as cur:
        rows = [tuple(r) async for r in cur]
    assert rows == [("tech", "en", "gemini-2.5-pro", "duplicate")]


async def test_dedup_whitespace_punctuation_variants(store: CorpusStore):
    t = zh_text(31)
    assert (await store.insert_text(**text_kwargs(t, lang="zh"))).inserted
    variant = "  " + t.replace("，", ", ").replace("。\n", "！\n\n\t") + " …"
    res = await store.insert_text(**text_kwargs(variant, lang="zh"))
    assert res.status == "duplicate" and res.match.kind == "exact"
    assert await store.is_duplicate(variant, "zh")


async def test_dedup_near_duplicate(store: CorpusStore):
    t = en_text(32)
    first = await store.insert_text(**text_kwargs(t))
    near_text = t + " garden"
    assert sha256_of(near_text) != sha256_of(t)
    assert hamming(simhash64(near_text, "en"), simhash64(t, "en")) <= 3
    res = await store.insert_text(**text_kwargs(near_text))
    assert res.status == "duplicate"
    assert res.match.kind == "near" and res.match.passage_id == first.id

    z = zh_text(33)
    zf = await store.insert_text(**text_kwargs(z, lang="zh"))
    zres = await store.insert_text(**text_kwargs(z + "新", lang="zh"))
    assert zres.status == "duplicate" and zres.match.passage_id == zf.id


async def test_dedup_distinct_texts_inserted(store: CorpusStore):
    ids = []
    for i in range(5):
        r = await store.insert_text(**text_kwargs(en_text(100 + i)))
        assert r.inserted
        ids.append(r.id)
    for i in range(3):
        assert (await store.insert_text(**text_kwargs(zh_text(200 + i), lang="zh"))).inserted
    assert (await store.stats()).active == 8
    assert await store.rejection_counts(0) == {}


async def test_dedup_applies_to_all_statuses(store: CorpusStore, fake_clock: FakeClock):
    q = en_text(40)
    e = en_text(41)
    rq = await store.insert_text(**text_kwargs(q))
    re_ = await store.insert_text(**text_kwargs(e, grounded=True, sources=["https://example.com"]))
    await store.quarantine([rq.id], "azure_content_filter", "hate")
    fake_clock.advance(73 * 3600)
    assert await store.expire_due() == 1
    assert (await store.get_passage(re_.id))["status"] == "expired"

    assert (await store.insert_text(**text_kwargs(q))).status == "duplicate"
    assert (await store.insert_text(**text_kwargs(e + " ocean"))).status == "duplicate"
    dd = await store.load_deduplicator()
    assert len(dd) == 2 and dd.is_duplicate(q) and dd.is_duplicate(e)


async def test_dedup_across_connections(tmp_path, fake_clock: FakeClock):
    path = tmp_path / "shared.db"
    async with CorpusStore(path, fake_clock) as a, CorpusStore(path, fake_clock) as b:
        t = en_text(50)
        assert (await a.insert_text(**text_kwargs(t))).inserted
        assert (await b.insert_text(**text_kwargs(t + " forest"))).status == "duplicate"


async def test_legacy_insert_returns_none_on_sha_conflict(store: CorpusStore):
    p = new_passage(en_text(60))
    assert await store.insert(p) is not None
    assert await store.insert(p) is None
