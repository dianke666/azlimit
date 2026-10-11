"""20.4 供应场景集成测试（Requirements 4.2, 8.2, 8.3, 8.4）。

真实 CorpusStore：zh 不足 / en 不足 / 长度桶失衡 / 含过期段落的池 → refuse / warn / 过期过滤；
refuse 时 RunController 不创建 Azure 客户端、不发任何请求。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loadtest.clock import FakeClock
from loadtest.corpus.planner import check_for_run
from loadtest.corpus.store import CorpusStore, NewPassage
from loadtest.load.runner import EXIT_SUPPLY, RunController, RunOptions
from tests.unit.test_runner import make_settings

LEN = {"S": 650, "M": 1000, "L": 1350}


async def _fill(store: CorpusStore, lang: str, n: int, *, buckets: str = "SML", grounded: bool = False,
                start: int = 0) -> None:
    for i in range(n):
        b = buckets[i % len(buckets)]
        await store.insert(NewPassage(
            category="daily_news" if grounded else "cooking", subtopic="s", genre="说明文", lang=lang,  # type: ignore[arg-type]
            token_len=LEN[b], length_bucket=b, text=f"{lang}-{start + i}-{grounded}",  # type: ignore[arg-type]
            sha256=f"{lang}-{start + i}-{grounded}", simhash=start + i, gen_model="m",
            grounded=grounded, sources=("https://example.com",) if grounded else (),
        ))


def _check(tmp: Path, setup) -> tuple:
    async def go():
        settings = make_settings(tmp)
        clock = FakeClock()
        async with CorpusStore(tmp / "corpus.db", clock) as store:
            await setup(store, clock)
            stats = await store.stats()
            return check_for_run(stats, settings.load.duration_s, settings), stats

    return asyncio.run(go())


def test_zh_insufficient_refuses(tmp_path: Path) -> None:
    async def setup(s, c):
        await _fill(s, "zh", 50)
        await _fill(s, "en", 200)

    sc, _ = _check(tmp_path, setup)
    assert sc.verdict == "refuse"
    assert sc.available["zh"] < sc.required["zh"]
    assert any("zh" in m for m in sc.messages) and sc.suggested_command


def test_en_insufficient_refuses(tmp_path: Path) -> None:
    async def setup(s, c):
        await _fill(s, "zh", 300)
        await _fill(s, "en", 30)

    sc, _ = _check(tmp_path, setup)
    assert sc.verdict == "refuse" and sc.available["en"] < sc.required["en"]


def test_bucket_imbalance_warns(tmp_path: Path) -> None:
    async def setup(s, c):
        await _fill(s, "zh", 300, buckets="M")
        await _fill(s, "en", 200)

    sc, _ = _check(tmp_path, setup)
    assert sc.verdict == "warn"
    assert any("zh/S" in m for m in sc.messages) and any("zh/L" in m for m in sc.messages)


def test_balanced_pool_ok(tmp_path: Path) -> None:
    async def setup(s, c):
        await _fill(s, "zh", 300)
        await _fill(s, "en", 200)

    sc, _ = _check(tmp_path, setup)
    assert sc.verdict == "ok", sc.messages


def test_expired_passages_excluded(tmp_path: Path) -> None:
    async def setup(s, c):
        await _fill(s, "zh", 150)
        await _fill(s, "en", 200)
        await _fill(s, "zh", 200, grounded=True, start=10_000)
        c.advance(73 * 3600)  # grounded 段落全部过期（72h）

    sc, stats = _check(tmp_path, setup)
    assert stats.active == 350 and stats.expired == 200
    assert sc.available["zh"] == 150  # 过期的 200 段不计入库存
    assert sc.verdict == "warn" and any("zh 占比" in m for m in sc.messages)

    async def loaded():
        async with CorpusStore(tmp_path / "corpus.db", FakeClock(start=73 * 3600)) as store:
            return await store.load_active_passages()

    ps = asyncio.run(loaded())
    assert len(ps) == 350 and all(p.expires_at is None for p in ps)


def test_runner_refuse_sends_zero_requests(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    created: list[int] = []

    async def go():
        clock = FakeClock()
        async with CorpusStore(tmp_path / "corpus.db", clock) as store:
            await _fill(store, "zh", 20)
            await _fill(store, "en", 20)
            ctrl = RunController(
                settings, RunOptions(no_replenish=True, seed=1), clock=clock, store=store,
                client_factory=lambda s: created.append(1), install_signals=False, show_console=False,
                out=lambda m: None, run_id="run-refuse",
            )
            res = await ctrl.run()
            run = await store.get_run("run-refuse")
            return res, run

    res, run = asyncio.run(go())
    assert res.exit_code == EXIT_SUPPLY and res.stop_reason == "supply_refused"
    assert created == [] and res.azure_requests == 0
    assert run is not None and run.stop_reason == "supply_refused"
