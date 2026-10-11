"""CorpusStore：SQLite（aiosqlite，WAL）访问层。

职责：
- 段落入库（grounded → ``expires_at = created_at + 72h``，否则 NULL；记录 ``gen_model``）
- 隔离（``quarantine``）、过期置位（``expire_due``）、运行结束累加复用计数（``flush_use_counts``）
- 库存统计 ``stats()`` → :class:`PoolStats`
- 拒绝记录（只记类别、语言、模型、原因、时间，不记正文）
- runs 记录读写（含 ``throttled``、``probes`` 与 ``throttle_json``）
- ledger 批量写入（首次失败后最多重试 3 次，仍失败抛出 :class:`LedgerWriteError`）、``ledger_totals`` 读取
- 当日 UTC grounding 调用计数
- ``load_active_passages(now)``：返回 active 且未过期的段落，供 PassageIndex 构建

时间：所有时间戳经注入的 ``Clock.wall()``（UTC epoch 秒）获得，以固定格式的 ISO8601 UTC 字符串存储，
因此可直接按字典序比较。
"""
from __future__ import annotations

import dataclasses
import enum
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import aiosqlite

from loadtest.clock import Clock, RealClock
from loadtest.corpus.dedup import (
    HAMMING_THRESHOLD,
    Deduplicator,
    DupMatch,
    bands,
    hamming,
    sha256_of,
    simhash64,
)

Lang = Literal["zh", "en"]
RunKind = Literal["warmup", "generate", "load"]
Provider = Literal["azure", "gemini", "gemini_grounding"]

GROUNDED_TTL = timedelta(hours=72)
LEDGER_MAX_RETRIES = 3  # 首次失败后的最大重试次数

# runs 表中可通过 update_run 修改的列
_RUN_MUTABLE_COLUMNS = frozenset(
    {
        "ended_at",
        "stop_reason",
        "config_json",
        "requests_ok",
        "requests_err",
        "throttled",
        "probes",
        "throttle_json",
        "input_tokens",
        "output_tokens",
        "passages_added",
        "cost_usd",
    }
)


# --------------------------------------------------------------------------- 时间辅助


def to_iso(ts: float | datetime) -> str:
    """epoch 秒或 datetime → 固定格式 ISO8601 UTC 字符串（可字典序比较）。"""
    if isinstance(ts, datetime):
        dt = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.isoformat(timespec="microseconds")


def from_iso(s: str) -> float:
    """ISO8601 字符串 → epoch 秒。"""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def utc_day_start(ts: float) -> float:
    """``ts`` 所在 UTC 日的 00:00 时刻（epoch 秒）。"""
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


# --------------------------------------------------------------------------- 数据类


@dataclass(frozen=True)
class Passage:
    """内存中的段落（PassageIndex 使用）。``expires_at`` 为 epoch 秒；非 grounded 为 None。"""

    id: int
    category: str
    lang: Lang
    token_len: int  # o200k_base
    use_count: int  # 历史累计复用次数（用于加权采样）
    expires_at: float | None
    text: str
    length_bucket: str | None = None  # S / M / L；由入库时的长度桶给出


@dataclass(frozen=True)
class NewPassage:
    """待入库段落。sha256 / simhash 由调用方（Deduplicator，任务 4.3）计算后传入。"""

    category: str
    subtopic: str
    genre: str
    lang: Lang
    token_len: int
    length_bucket: Literal["S", "M", "L"]
    text: str
    sha256: str
    simhash: int  # 有符号 64 位整数（无符号值需先转换，见 :func:`simhash_to_signed`）
    gen_model: str
    grounded: bool = False
    sources: Sequence[str] = ()


@dataclass(frozen=True)
class InsertResult:
    """:meth:`CorpusStore.insert_text` 的结果。duplicate 时 ``id`` 为 None，``match`` 给出命中信息。"""

    status: Literal["inserted", "duplicate"]
    id: int | None = None
    match: DupMatch | None = None

    @property
    def inserted(self) -> bool:
        return self.status == "inserted"


@dataclass
class PoolStats:
    """库存统计。``by_*`` 只统计 status='active' 且未过期的段落。"""

    active: int = 0  # active 且未过期
    quarantined: int = 0
    expired: int = 0  # status='expired' + status='active' 但已到期（尚未置位）
    by_lang: dict[str, int] = field(default_factory=dict)
    by_bucket: dict[tuple[str, str], int] = field(default_factory=dict)  # (lang, bucket)
    by_category: dict[tuple[str, str], int] = field(default_factory=dict)  # (lang, category)
    by_cell: dict[tuple[str, str, str], int] = field(default_factory=dict)  # (lang, category, bucket)
    by_use_count: dict[int, int] = field(default_factory=dict)  # use_count → 段落数
    quarantined_cells: dict[tuple[str, str, str], int] = field(default_factory=dict)
    expired_cells: dict[tuple[str, str, str], int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.active


@dataclass(frozen=True)
class LedgerEntry:
    run_id: str
    provider: Provider
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    units: int = 0
    account: str = ""  # 预算归属账号（Gemini 为 GCP 项目或 cost.gemini_budget_account）


@dataclass(frozen=True)
class LedgerTotal:
    provider: str
    model: str
    total_usd: float
    in_tok: int
    out_tok: int
    units: int


@dataclass
class RunRecord:
    run_id: str
    kind: str
    started_at: str
    config_json: str
    ended_at: str | None = None
    stop_reason: str | None = None
    requests_ok: int = 0
    requests_err: int = 0
    throttled: int = 0
    probes: int = 0
    throttle_json: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    passages_added: int = 0
    cost_usd: float = 0.0

    @property
    def throttle_stats(self) -> dict[str, Any] | None:
        """``throttle_json`` 解析后的字典（ThrottleStats 快照）。"""
        return json.loads(self.throttle_json) if self.throttle_json else None


class LedgerWriteError(RuntimeError):
    """账本写入在首次失败 + 3 次重试后仍失败：致命错误（Run_Controller 以 stop_reason=error 退出）。"""


# --------------------------------------------------------------------------- 序列化辅助


def simhash_to_signed(h: int) -> int:
    """无符号 64 位 → SQLite 可存储的有符号 64 位。"""
    h &= (1 << 64) - 1
    return h - (1 << 64) if h >= (1 << 63) else h


def _jsonable(obj: Any) -> Any:
    """把 ThrottleStats 等 dataclass（含枚举键/值、元组）转换为可 JSON 序列化的结构。"""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, enum.Enum):
        return _jsonable(obj.value) if isinstance(obj.value, (str, int, float)) else obj.name
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = k.value if isinstance(k, enum.Enum) else k
            out[str(key)] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return obj


def throttle_to_json(stats: Any) -> str | None:
    """ThrottleStats（dataclass）/ dict / 已序列化字符串 → JSON 字符串。"""
    if stats is None:
        return None
    if isinstance(stats, str):
        return stats
    return json.dumps(_jsonable(stats), ensure_ascii=False, sort_keys=True)


def _load_schema() -> str:
    return resources.files("loadtest.db").joinpath("schema.sql").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- CorpusStore


class CorpusStore:
    """SQLite 访问层。每个协程组（调度器 / Replenisher）应使用各自的实例（独立连接）。"""

    def __init__(
        self,
        db_path: str | Path,
        clock: Clock | None = None,
        *,
        grounded_ttl: timedelta = GROUNDED_TTL,
        ledger_retry_delay_s: float = 0.5,
    ) -> None:
        self.db_path = str(db_path)
        self.clock: Clock = clock or RealClock()
        self.grounded_ttl = grounded_ttl
        self.ledger_retry_delay_s = ledger_retry_delay_s
        self._db: aiosqlite.Connection | None = None

    # ---------------------------------------------------------------- 生命周期

    async def open(self) -> "CorpusStore":
        if self._db is not None:
            return self
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.executescript(_load_schema())
        await self._migrate(db)
        await db.commit()
        self._db = db
        return self

    @staticmethod
    async def _migrate(db: aiosqlite.Connection) -> None:
        """旧库迁移：ledger 补 account 列（旧行归属空账号 ''），并建立按账号查询的索引。"""
        async with db.execute("PRAGMA table_info(ledger)") as cur:
            cols = {r[1] async for r in cur}
        if "account" not in cols:
            await db.execute("ALTER TABLE ledger ADD COLUMN account TEXT NOT NULL DEFAULT ''")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_ledger_account ON ledger(provider, account)")

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "CorpusStore":
        return await self.open()

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("CorpusStore is not open")
        return self._db

    def _now_iso(self) -> str:
        return to_iso(self.clock.wall())

    def _resolve_now(self, now: float | datetime | None) -> str:
        return to_iso(self.clock.wall() if now is None else now)

    # ---------------------------------------------------------------- 段落

    async def insert(self, p: NewPassage) -> int | None:
        """插入段落并返回 id；sha256 已存在（UNIQUE 冲突）时返回 None 且不插入。

        不做 SimHash 近似去重、不写 rejections；生成流程应使用 :meth:`insert_text`。
        """
        try:
            pid = await self._insert_row(p)
        except sqlite3.IntegrityError as e:
            await self.db.rollback()
            if "sha256" in str(e):
                return None
            raise
        await self.db.commit()
        return pid

    async def insert_text(
        self,
        *,
        category: str,
        subtopic: str,
        genre: str,
        lang: Lang,
        token_len: int,
        length_bucket: Literal["S", "M", "L"],
        text: str,
        gen_model: str,
        grounded: bool = False,
        sources: Sequence[str] = (),
        run_id: str | None = None,
    ) -> InsertResult:
        """去重后入库（Requirements 6.1–6.4）。

        计算规范化 SHA256 与 64 位 SimHash，对所有状态（active / quarantined / expired）段落检查
        精确匹配与汉明距离 ≤ 3；重复则不插入，并以原因 ``duplicate`` 写入 rejections（不含正文）。
        检查与插入在同一个 ``BEGIN IMMEDIATE`` 事务中完成，多连接并发入库时也不会漏判。
        """
        sha = sha256_of(text)
        sh = simhash64(text, lang)
        p = NewPassage(
            category=category,
            subtopic=subtopic,
            genre=genre,
            lang=lang,
            token_len=token_len,
            length_bucket=length_bucket,
            text=text,
            sha256=sha,
            simhash=sh,
            gen_model=gen_model,
            grounded=grounded,
            sources=sources,
        )
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            match = await self._find_duplicate(sha, sh)
            pid: int | None = None
            if match is None:
                try:
                    pid = await self._insert_row(p)
                except sqlite3.IntegrityError as e:
                    if "sha256" not in str(e):
                        raise
                    match = DupMatch("exact", 0, None)
        except BaseException:
            await self.db.rollback()
            raise
        if match is not None:
            await self.db.rollback()
            await self.record_rejection(category, lang, gen_model, "duplicate", run_id=run_id)
            return InsertResult("duplicate", None, match)
        await self.db.commit()
        return InsertResult("inserted", pid, None)

    async def find_duplicate(self, text: str, lang: Lang | None = None) -> DupMatch | None:
        """对库中所有状态段落检查 ``text`` 是否重复（只读）。"""
        return await self._find_duplicate(sha256_of(text), simhash64(text, lang))

    async def is_duplicate(self, text: str, lang: Lang | None = None) -> bool:
        return await self.find_duplicate(text, lang) is not None

    async def load_deduplicator(self) -> Deduplicator:
        """把库中全部段落（所有状态）的指纹载入内存 :class:`Deduplicator`。"""
        dd = Deduplicator()
        async with self.db.execute("SELECT id, sha256, simhash FROM passages") as cur:
            async for r in cur:
                dd.add_fingerprint(r["sha256"], r["simhash"], r["id"])
        return dd

    async def _find_duplicate(self, sha: str, simhash: int) -> DupMatch | None:
        async with self.db.execute("SELECT id FROM passages WHERE sha256 = ?", (sha,)) as cur:
            row = await cur.fetchone()
        if row:
            return DupMatch("exact", 0, row["id"])
        b = bands(simhash)
        # 表达式须与 schema.sql 中的 band 索引逐字一致
        sql = """SELECT id, simhash FROM passages WHERE (simhash & 65535) = ?
                 UNION SELECT id, simhash FROM passages WHERE ((simhash >> 16) & 65535) = ?
                 UNION SELECT id, simhash FROM passages WHERE ((simhash >> 32) & 65535) = ?
                 UNION SELECT id, simhash FROM passages WHERE ((simhash >> 48) & 65535) = ?"""
        best: DupMatch | None = None
        async with self.db.execute(sql, b) as cur:
            async for r in cur:
                d = hamming(simhash, r["simhash"])
                if d <= HAMMING_THRESHOLD and (best is None or d < best.distance):
                    best = DupMatch("near", d, r["id"])
        return best

    async def _insert_row(self, p: NewPassage) -> int | None:
        created = self.clock.wall()
        expires = to_iso(created + self.grounded_ttl.total_seconds()) if p.grounded else None
        cur = await self.db.execute(
            """INSERT INTO passages
               (category, subtopic, genre, lang, token_len, length_bucket, text,
                sha256, simhash, grounded, sources, gen_model, created_at, expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                p.category,
                p.subtopic,
                p.genre,
                p.lang,
                p.token_len,
                p.length_bucket,
                p.text,
                p.sha256,
                simhash_to_signed(p.simhash),
                1 if p.grounded else 0,
                json.dumps(list(p.sources), ensure_ascii=False) if p.sources else None,
                p.gen_model,
                to_iso(created),
                expires,
            ),
        )
        return cur.lastrowid

    async def get_passage(self, pid: int) -> dict[str, Any] | None:
        async with self.db.execute("SELECT * FROM passages WHERE id = ?", (pid,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def quarantine(
        self,
        passage_ids: Iterable[int],
        reason: str,
        detail: str | None = None,
        *,
        run_id: str | None = None,
    ) -> None:
        """把段落置为 quarantined 并逐条写入 quarantine 表。"""
        ids = list(dict.fromkeys(int(i) for i in passage_ids))
        if not ids:
            return
        now = self._now_iso()
        await self.db.executemany(
            "UPDATE passages SET status = 'quarantined', filter_hits = filter_hits + 1 WHERE id = ?",
            [(i,) for i in ids],
        )
        await self.db.executemany(
            "INSERT INTO quarantine (passage_id, run_id, reason, detail, created_at) VALUES (?,?,?,?,?)",
            [(i, run_id, reason, detail, now) for i in ids],
        )
        await self.db.commit()

    async def expire_due(self, now: float | datetime | None = None) -> int:
        """把 active 且 ``expires_at <= now`` 的段落置为 expired，返回受影响行数。"""
        cur = await self.db.execute(
            """UPDATE passages SET status = 'expired'
               WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at <= ?""",
            (self._resolve_now(now),),
        )
        await self.db.commit()
        return cur.rowcount

    async def flush_use_counts(self, counts: Mapping[int, int]) -> None:
        """把本次运行的 Run_Uses 累加到 use_count。调用方保证每次运行只调用一次（或传增量）。"""
        rows = [(int(n), int(pid)) for pid, n in counts.items() if n]
        if not rows:
            return
        await self.db.executemany("UPDATE passages SET use_count = use_count + ? WHERE id = ?", rows)
        await self.db.commit()

    async def load_active_passages(self, now: float | datetime | None = None) -> list[Passage]:
        """返回 status='active' 且未过期（expires_at 为空或 > now）的全部段落。"""
        sql = """SELECT id, category, lang, token_len, use_count, expires_at, text, length_bucket
                 FROM passages
                 WHERE status = 'active' AND (expires_at IS NULL OR expires_at > ?)
                 ORDER BY id"""
        out: list[Passage] = []
        async with self.db.execute(sql, (self._resolve_now(now),)) as cur:
            async for r in cur:
                out.append(
                    Passage(
                        id=r["id"],
                        category=r["category"],
                        lang=r["lang"],
                        token_len=r["token_len"],
                        use_count=r["use_count"],
                        expires_at=from_iso(r["expires_at"]) if r["expires_at"] else None,
                        text=r["text"],
                        length_bucket=r["length_bucket"],
                    )
                )
        return out

    async def stats(self, now: float | datetime | None = None) -> PoolStats:
        now_iso = self._resolve_now(now)
        st = PoolStats()
        sql = """SELECT lang, category, length_bucket, use_count,
                        CASE WHEN status = 'active' AND expires_at IS NOT NULL AND expires_at <= ?
                             THEN 'expired' ELSE status END AS eff_status,
                        COUNT(*) AS n
                 FROM passages
                 GROUP BY lang, category, length_bucket, use_count, eff_status"""
        async with self.db.execute(sql, (now_iso,)) as cur:
            async for r in cur:
                lang, cat, b, uc, s, n = (
                    r["lang"], r["category"], r["length_bucket"], r["use_count"], r["eff_status"], r["n"],
                )
                cell = (lang, cat, b)
                if s == "active":
                    st.active += n
                    st.by_lang[lang] = st.by_lang.get(lang, 0) + n
                    st.by_bucket[(lang, b)] = st.by_bucket.get((lang, b), 0) + n
                    st.by_category[(lang, cat)] = st.by_category.get((lang, cat), 0) + n
                    st.by_cell[cell] = st.by_cell.get(cell, 0) + n
                    st.by_use_count[uc] = st.by_use_count.get(uc, 0) + n
                elif s == "quarantined":
                    st.quarantined += n
                    st.quarantined_cells[cell] = st.quarantined_cells.get(cell, 0) + n
                else:
                    st.expired += n
                    st.expired_cells[cell] = st.expired_cells.get(cell, 0) + n
        return st

    # ---------------------------------------------------------------- 拒绝记录

    async def record_rejection(
        self,
        category: str,
        lang: Lang,
        model: str,
        reason: str,
        *,
        run_id: str | None = None,
    ) -> None:
        """只记录类别、语言、模型、原因与时间（不接受、不存储正文）。"""
        await self.db.execute(
            "INSERT INTO rejections (run_id, category, lang, model, reason, created_at) VALUES (?,?,?,?,?,?)",
            (run_id, category, lang, model, reason, self._now_iso()),
        )
        await self.db.commit()

    async def rejection_counts(self, since: float | datetime) -> dict[str, int]:
        """``since`` 之后各拒绝原因的次数（pool-stats 近 7 天占比使用）。"""
        async with self.db.execute(
            "SELECT reason, COUNT(*) AS n FROM rejections WHERE created_at >= ? GROUP BY reason",
            (to_iso(since),),
        ) as cur:
            return {r["reason"]: r["n"] async for r in cur}

    # ---------------------------------------------------------------- runs

    async def create_run(
        self,
        run_id: str,
        kind: RunKind,
        config_json: str,
        *,
        started_at: float | datetime | None = None,
    ) -> None:
        await self.db.execute(
            "INSERT INTO runs (run_id, kind, started_at, config_json) VALUES (?,?,?,?)",
            (run_id, kind, self._resolve_now(started_at), config_json),
        )
        await self.db.commit()

    async def update_run(self, run_id: str, *, throttle_stats: Any = None, **cols: Any) -> None:
        """更新 runs 列（计数、stop_reason 等）；``throttle_stats`` 为 ThrottleStats 快照。"""
        if throttle_stats is not None:
            cols["throttle_json"] = throttle_to_json(throttle_stats)
        bad = set(cols) - _RUN_MUTABLE_COLUMNS
        if bad:
            raise ValueError(f"unknown runs columns: {sorted(bad)}")
        if not cols:
            return
        names = sorted(cols)
        sets = ", ".join(f"{c} = ?" for c in names)
        await self.db.execute(
            f"UPDATE runs SET {sets} WHERE run_id = ?", [*(cols[c] for c in names), run_id]
        )
        await self.db.commit()

    async def finish_run(
        self,
        run_id: str,
        stop_reason: str,
        *,
        ended_at: float | datetime | None = None,
        throttle_stats: Any = None,
        **counters: Any,
    ) -> None:
        """写入 ended_at、stop_reason 与最终计数 / ThrottleStats 快照。"""
        await self.update_run(
            run_id,
            throttle_stats=throttle_stats,
            ended_at=self._resolve_now(ended_at),
            stop_reason=stop_reason,
            **counters,
        )

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)) as cur:
            row = await cur.fetchone()
        return RunRecord(**dict(row)) if row else None

    async def latest_run(self, kind: RunKind | None = None) -> RunRecord | None:
        sql = "SELECT * FROM runs"
        args: tuple[Any, ...] = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            args = (kind,)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT 1"
        async with self.db.execute(sql, args) as cur:
            row = await cur.fetchone()
        return RunRecord(**dict(row)) if row else None

    async def runs_cost_total(self) -> float:
        async with self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM runs") as cur:
            (v,) = await cur.fetchone()
        return float(v)

    # ---------------------------------------------------------------- ledger

    async def write_ledger(self, entries: Sequence[LedgerEntry]) -> None:
        """批量写入账本，并在同一事务中把花费累加到对应 ``runs.cost_usd``（保证 26.3 恒等式）。

        首次失败后最多重试 ``LEDGER_MAX_RETRIES``（3）次；仍失败抛出 :class:`LedgerWriteError`。
        """
        if not entries:
            return
        last_exc: BaseException | None = None
        for attempt in range(1 + LEDGER_MAX_RETRIES):
            if attempt:
                await self.clock.sleep(self.ledger_retry_delay_s)
            try:
                await self._write_ledger_once(entries)
                return
            except (sqlite3.Error, ValueError) as e:
                last_exc = e
                try:
                    await self.db.rollback()
                except sqlite3.Error:
                    pass
        raise LedgerWriteError(
            f"ledger write failed after {1 + LEDGER_MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def _write_ledger_once(self, entries: Sequence[LedgerEntry]) -> None:
        now = self._now_iso()
        await self.db.executemany(
            """INSERT INTO ledger
               (run_id, provider, model, input_tokens, output_tokens, units, cost_usd, flushed_at, account)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (e.run_id, e.provider, e.model, e.input_tokens, e.output_tokens, e.units, e.cost_usd, now,
                 e.account)
                for e in entries
            ],
        )
        per_run: dict[str, float] = {}
        for e in entries:
            per_run[e.run_id] = per_run.get(e.run_id, 0.0) + e.cost_usd
        await self.db.executemany(
            "UPDATE runs SET cost_usd = COALESCE(cost_usd, 0) + ? WHERE run_id = ?",
            [(c, r) for r, c in per_run.items()],
        )
        await self.db.commit()

    async def ledger_totals(
        self, providers: Iterable[str] | None = None, account: str | None = None
    ) -> list[LedgerTotal]:
        """按 (provider, model) 汇总花费；``account`` 非 None 时只统计该账号的账本行。"""
        where: list[str] = []
        args: list[str] = []
        if providers is not None:
            provs = list(providers)
            if not provs:
                return []
            where.append(f"provider IN ({','.join('?' * len(provs))})")
            args.extend(provs)
        if account is not None:
            where.append("account = ?")
            args.append(account)
        sql = (
            "SELECT provider, model, SUM(cost_usd) AS total_usd, SUM(input_tokens) AS in_tok, "
            "SUM(output_tokens) AS out_tok, SUM(units) AS units FROM ledger"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY provider, model ORDER BY provider, model"
        async with self.db.execute(sql, args) as cur:
            return [
                LedgerTotal(
                    provider=r["provider"],
                    model=r["model"],
                    total_usd=float(r["total_usd"] or 0.0),
                    in_tok=int(r["in_tok"] or 0),
                    out_tok=int(r["out_tok"] or 0),
                    units=int(r["units"] or 0),
                )
                async for r in cur
            ]

    async def spent_usd(self, providers: Iterable[str], account: str | None = None) -> float:
        """指定 provider（可选限定账号）的历史累计花费（CostGuard 初值）。"""
        return sum(t.total_usd for t in await self.ledger_totals(providers, account))

    async def ledger_accounts(self, providers: Iterable[str]) -> list[tuple[str, float]]:
        """各账号的累计花费 [(account, usd)]，按账号排序（budget 命令展示历史账号）。"""
        provs = list(providers)
        if not provs:
            return []
        sql = (
            f"SELECT account, SUM(cost_usd) AS usd FROM ledger WHERE provider IN ({','.join('?' * len(provs))}) "
            "GROUP BY account ORDER BY account"
        )
        async with self.db.execute(sql, provs) as cur:
            return [(r["account"], float(r["usd"] or 0.0)) async for r in cur]

    async def grounding_calls_today(self, now: float | None = None) -> int:
        """当前 UTC 日（00:00 起）已刷盘的 grounding 调用次数（provider='gemini_grounding' 的 units 之和）。"""
        ts = self.clock.wall() if now is None else now
        async with self.db.execute(
            """SELECT COALESCE(SUM(units), 0) FROM ledger
               WHERE provider = 'gemini_grounding' AND flushed_at >= ?""",
            (to_iso(utc_day_start(ts)),),
        ) as cur:
            (n,) = await cur.fetchone()
        return int(n)


__all__ = [
    "CorpusStore",
    "Deduplicator",
    "DupMatch",
    "GROUNDED_TTL",
    "InsertResult",
    "LEDGER_MAX_RETRIES",
    "LedgerEntry",
    "LedgerTotal",
    "LedgerWriteError",
    "NewPassage",
    "Passage",
    "PoolStats",
    "RunRecord",
    "from_iso",
    "simhash_to_signed",
    "throttle_to_json",
    "to_iso",
    "utc_day_start",
]
