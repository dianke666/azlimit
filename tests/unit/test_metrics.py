"""指标、控制台、JSONL/CSV 与报告单元测试（Req 1.7, 15.3, 16.5, 28.1–28.5）。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from rich.console import Console

from loadtest.clock import FakeClock
from loadtest.load.azure_client import AzureResult
from loadtest.load.models import PlannedRequest
from loadtest.metrics.collector import (
    Histogram,
    MetricsCollector,
    exact_percentile,
    max_run_uses,
    reuse_distribution,
)
from loadtest.metrics.console import ConsoleReporter, render
from loadtest.metrics.report import (
    build_report,
    probes_count,
    throttle_json,
    throttle_timeline,
    write_report,
)
from loadtest.metrics.writers import CsvWriter, JsonlWriter
from loadtest.outcome import Outcome
from loadtest.settings import ThrottleSettings, clear_registered_secrets, register_secrets
from loadtest.throttle import ThrottleController, ThrottleState

RAW_KEY = "sk-RAWKEY-0123456789abcdefXYZ"


@pytest.fixture(autouse=True)
def _secrets():
    clear_registered_secrets()
    register_secrets([RAW_KEY])
    yield
    clear_registered_secrets()


def planned(i: int, lang: str = "zh", inp: int = 1000, mx: int = 40, task_id: str = "summary") -> PlannedRequest:
    return PlannedRequest(
        request_id=f"r{i}",
        lang=lang,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": f"SECRET PASSAGE TEXT {i}"}],
        input_tokens=inp,
        max_tokens=mx,
        passage_ids=(i, i + 1),
        task_id=task_id,
        target_tokens=inp,
    )


def result(p: PlannedRequest, outcome: Outcome = Outcome.OK, lat: float = 100.0, comp: int = 30) -> AzureResult:
    return AzureResult(
        request_id=p.request_id,
        outcome=outcome,
        status=200 if outcome is Outcome.OK else 429,
        latency_ms=lat,
        prompt_tokens=p.input_tokens,
        completion_tokens=comp if outcome is Outcome.OK else None,
        text="MODEL OUTPUT TEXT",
    )


def drive_throttle(clock: FakeClock) -> tuple[ThrottleController, MetricsCollector]:
    """t=0 429 → HOLD；hold 到期 → PROBE；首探 429 → HOLD；再探 OK → RECOVER → 回满。"""
    th = ThrottleController(ThrottleSettings(), clock)
    m = MetricsCollector(clock, throttle=th)
    p0 = planned(0)
    tk = th.note_dispatch(clock(), False)
    m.record_dispatch(p0, clock())
    th.on_result(tk, result(p0, Outcome.THROTTLED), clock())
    th.note_done(tk)
    m.record_result(p0, result(p0, Outcome.THROTTLED), clock())
    probes = [Outcome.THROTTLED, Outcome.OK]
    i = 1
    for _ in range(400):
        clock.advance(1.0)
        th.tick(clock())
        n, _lvl = th.allowed_dispatch(clock())
        if th.state is ThrottleState.PROBE and n == 1 and probes:
            p = planned(100 + i)
            i += 1
            t = th.note_dispatch(clock(), True)
            m.record_dispatch(p, clock(), is_probe=True)
            r = result(p, probes.pop(0))
            th.on_result(t, r, clock())
            th.note_done(t)
            m.record_result(p, r, clock(), is_probe=True)
        if th.state is ThrottleState.NORMAL and not probes:
            break
    assert th.state is ThrottleState.NORMAL
    return th, m


# ---------------------------------------------------------------------------
# collector
# ---------------------------------------------------------------------------


def test_histogram_percentiles_close_to_exact():
    h = Histogram()
    vals = list(range(1, 10001))
    for v in vals:
        h.add(float(v))
    for p in (0.5, 0.95):
        exact = exact_percentile(vals, p)
        assert h.percentile(p) == pytest.approx(exact, rel=0.011)
    small = Histogram()
    for v in (10, 20, 30, 40):
        small.add(v)
    assert small.percentile(0.5) == 20
    assert small.percentile(0.95) == 40
    assert Histogram().percentile(0.5) is None


def test_exact_percentile_nearest_rank():
    assert exact_percentile([5, 1, 3, 2, 4], 0.5) == 3
    assert exact_percentile(range(1, 101), 0.95) == 95
    assert exact_percentile([], 0.5) is None


def test_window_stats_tpm_rpm_and_eviction():
    clock = FakeClock(0.0)
    m = MetricsCollector(clock, window_s=60)
    for i in range(10):  # t = 0..9，每个 1000 + 40
        p = planned(i, inp=1000 + i * 100)
        m.record_dispatch(p, clock())
        m.record_result(p, result(p, lat=100.0 * (i + 1), comp=20 + i), clock())
        clock.advance(1.0)
    w = m.window_stats(clock())
    assert w["rpm"] == 10
    assert w["tpm"] == sum(1000 + i * 100 + 40 for i in range(10))
    assert w["input_p50"] == 1400
    assert w["input_p95"] == 1900
    assert w["latency_p50_ms"] == 500
    assert w["latency_p95_ms"] == 1000
    assert w["avg_output_tokens"] == pytest.approx(sum(20 + i for i in range(10)) / 10)

    clock.set(65.5)  # t=0..5 的样本已出窗口
    w = m.window_stats(clock())
    assert w["rpm"] == 4
    assert w["tpm"] == sum(1000 + i * 100 + 40 for i in range(6, 10))
    # 累计值不受窗口影响
    s = m.snapshot(clock())
    assert s["dispatched"] == 10
    assert s["outcomes"]["ok"] == 10
    assert s["input_tokens_total"] == sum(1000 + i * 100 for i in range(10))


def test_outcome_counts_probe_separate_lang_and_dispatch_duration():
    clock = FakeClock(100.0)
    m = MetricsCollector(clock)
    m.mark_dispatch_start(100.0)
    langs = ["zh", "zh", "en", "zh", "en"]
    for i, lang in enumerate(langs):
        p = planned(i, lang=lang)
        m.record_dispatch(p, clock())
        m.record_result(p, result(p, Outcome.THROTTLED if i == 0 else Outcome.OK), clock())
    pp = planned(99)
    m.record_dispatch(pp, clock(), is_probe=True)
    m.record_result(pp, result(pp, Outcome.THROTTLED), clock(), is_probe=True)
    clock.advance(120.0)
    m.mark_dispatch_stop(clock())
    clock.advance(30.0)  # drain 时间不计入派发时长
    s = m.snapshot(clock(), extra={"target_tpm": 800000, "queue_len": 3})
    assert s["outcomes"]["throttled"] == 1 and s["outcomes"]["ok"] == 4
    assert s["probe_outcomes"]["throttled"] == 1 and s["probe_outcomes"]["ok"] == 0
    assert s["throttled_total"] == 2
    assert s["lang_counts"] == {"zh": 4, "en": 2}
    assert s["dispatch_duration_s"] == pytest.approx(120.0)
    assert s["target_tpm"] == 800000 and s["queue_len"] == 3
    assert s["avg_output_tokens_total"] == pytest.approx(30.0)


def test_reuse_distribution():
    ru = {1: 1, 2: 1, 3: 2, 4: 5, 5: 0}
    assert reuse_distribution(ru) == {1: 2, 2: 1, 5: 1}
    assert max_run_uses(ru) == 5
    assert max_run_uses({}) == 0


def test_snapshot_includes_throttle_state():
    clock = FakeClock(0.0)
    th, m = drive_throttle(clock)
    s = m.snapshot(clock())
    assert s["throttle"]["state"] == "NORMAL"
    assert s["throttle"]["level"] == 1.0
    st = s["throttle"]["stats"]
    assert st["throttle_events"] == 1
    assert st["probes_sent"] == 2 and st["probe_ok"] == 1 and st["probe_throttled"] == 1
    assert sum(st["time_in_state"].values()) == pytest.approx(clock())


# ---------------------------------------------------------------------------
# console / CSV / JSONL
# ---------------------------------------------------------------------------


def _console_text(snapshot) -> str:
    con = Console(record=True, width=220, color_system=None)
    ConsoleReporter(con).print(snapshot)
    return con.export_text()


def test_console_shows_throttle_state_and_probe_counts():
    clock = FakeClock(0.0)
    th = ThrottleController(ThrottleSettings(), clock)
    m = MetricsCollector(clock, throttle=th)
    th.on_throttle(5.0, clock())
    clock.advance(3.0)
    th.tick(clock())
    txt = _console_text(m.snapshot(clock(), extra={"ramp": "ramping 50%", "effective_level": 0.1, "cost": RAW_KEY}))
    assert "HOLD" in txt
    assert "probes sent/ok/429/err/cf" in txt
    assert "0/0/0/0/0" in txt
    assert "ramping 50%" in txt
    assert "HOLD=3.0" in txt
    assert RAW_KEY not in txt

    _th, m2 = drive_throttle(FakeClock(0.0))
    txt2 = _console_text(m2.snapshot())
    assert "NORMAL" in txt2 and "2/1/1/0/0" in txt2
    assert render(m2.snapshot()) is not None


def test_jsonl_fields_is_probe_and_no_text_or_key(tmp_path: Path):
    path = tmp_path / "requests.jsonl"
    w = JsonlWriter(path, flush_every=100)
    p1 = planned(1, task_id=f"task-{RAW_KEY}")
    p2 = planned(2, lang="en")
    w.write_result(p1, result(p1), 1_700_000_000.0, is_probe=False)
    w.write_result(p2, result(p2, Outcome.THROTTLED), 1_700_000_001.0, is_probe=True)
    assert w.pending == 2 and not path.exists()  # 批量缓冲
    w.flush()
    raw = path.read_text(encoding="utf-8")
    assert raw.count(RAW_KEY) == 0
    assert "SECRET PASSAGE TEXT" not in raw and "MODEL OUTPUT TEXT" not in raw
    lines = [json.loads(x) for x in raw.splitlines()]
    assert len(lines) == 2
    a, b = lines
    for k in (
        "timestamp",
        "request_id",
        "lang",
        "task_id",
        "passage_ids",
        "input_tokens",
        "output_tokens",
        "outcome",
        "status",
        "latency_ms",
        "is_probe",
    ):
        assert k in a
    assert a["is_probe"] is False and b["is_probe"] is True
    assert a["outcome"] == "ok" and b["outcome"] == "throttled"
    assert a["passage_ids"] == [1, 2] and a["output_tokens"] == 30
    assert "messages" not in a and "text" not in a


def test_jsonl_auto_flush(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    w = JsonlWriter(path, flush_every=2)
    for i in range(3):
        p = planned(i)
        w.write_result(p, result(p), None)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    w.close()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_csv_contains_throttle_state_probe_counts_and_no_key(tmp_path: Path):
    clock = FakeClock(0.0)
    th, m = drive_throttle(clock)
    path = tmp_path / "summary.csv"
    w = CsvWriter(path)
    extra = {"target_tpm": 800000.0, "effective_level": 1.0, "pool_active": 8000, "cost": 1.5, "note": RAW_KEY}
    w.write_row(m.snapshot(clock(), extra=extra))
    clock.advance(10)
    w.write_row(m.snapshot(clock(), extra={**extra, "ramp": RAW_KEY}))
    raw = path.read_text(encoding="utf-8")
    assert raw.count(RAW_KEY) == 0
    rows = list(csv.DictReader(raw.splitlines()))
    assert len(rows) == 2
    r = rows[0]
    assert r["throttle_state"] == "NORMAL"
    assert r["throttle.stats.probes_sent"] == "2"
    assert r["throttle.stats.probe_ok"] == "1"
    assert r["throttle.stats.probe_throttled"] == "1"
    assert r["throttle.stats.throttle_events"] == "1"
    for s in ("normal", "hold", "probe", "recover"):
        assert f"throttle.stats.time_in_state.{s}" in r
    for k in ("timestamp", "window.tpm", "window.rpm", "outcomes.ok", "lang_share.zh", "target_tpm", "in_flight"):
        assert k in r
    # 新 CsvWriter 追加时沿用表头
    CsvWriter(path).write_row(m.snapshot(clock(), extra=extra))
    assert len(list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))) == 3


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _summary(avg_comp: int = 30, **kw):
    clock = FakeClock(0.0)
    th, m = drive_throttle(clock)  # 含 1 个普通 + 2 个探测
    for i in range(10):
        p = planned(i, lang="zh" if i < 6 else "en", inp=2000)
        m.record_dispatch(p, clock())
        m.record_result(p, result(p, comp=avg_comp), clock())
    clock.advance(1.0)
    m.mark_dispatch_stop(clock())
    return m.summary(
        clock(),
        stop_reason="duration",
        run_uses={1: 1, 2: 2, 3: 2},
        passages_added=7,
        cost={"azure_usd": 0.25, "gemini_usd": 0.10},
        **kw,
    ), m, th


def test_report_fields():
    s, m, th = _summary()
    rj, md = build_report(s)
    assert rj["stop_reason"] == "duration"
    dur_min = m.dispatch_duration_s() / 60
    assert rj["avg_tpm"] == pytest.approx((m.input_tokens_total + m.max_tokens_total) / dur_min)
    assert rj["avg_rpm"] == pytest.approx(13 / dur_min)
    assert rj["requests"]["dispatched"] == 13
    assert rj["throttled"] == 2  # 普通 1 + 探测 1
    # zh: 1 + 2 探测 + 6 = 9 / 13；en 4 / 13
    assert rj["lang_share_pct"]["zh"] == round(9 * 100 / 13, 1)
    assert rj["lang_share_pct"]["en"] == round(4 * 100 / 13, 1)
    assert f"{rj['lang_share_pct']['zh']:.1f}%" in md
    assert rj["reuse_distribution"] == {"1": 1, "2": 2}
    assert rj["max_run_uses"] == 2
    assert rj["passages_added"] == 7
    assert rj["cost"]["azure_usd"] == 0.25
    assert rj["tokens"]["avg_output_tokens"] == pytest.approx(30.0)
    assert rj["warnings"] == []
    assert rj["probes"] == 2
    json.dumps(rj)  # JSON 兼容


@pytest.mark.parametrize("comp,warn", [(10, True), (19, True), (20, False), (40, False), (41, True)])
def test_output_token_deviation_warning(comp, warn):
    s, _m, _th = _summary(avg_comp=comp)
    # 探测 OK 也按 comp 记 30 —— 用 ok_* 字段覆盖为精确值
    s["ok_count"] = 10
    s["ok_completion_tokens_total"] = comp * 10
    rj, md = build_report(s)
    assert bool(rj["warnings"]) is warn
    assert ("偏离" in md) is warn


def test_report_md_contains_throttle_stats_and_timeline(tmp_path: Path):
    s, _m, th = _summary()
    rj, md = write_report(tmp_path, s)
    md_file = (tmp_path / "report.md").read_text(encoding="utf-8")
    js_file = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert md_file == md and js_file == rj
    for k in th.stats.to_dict():
        if k not in ("time_in_state", "timeline"):
            assert f"| {k} |" in md
    for st in ("normal", "hold", "probe", "recover"):
        assert f"| {st} |" in md
    assert "Throttle event timeline" in md
    tl = rj["throttle_timeline"]
    assert len(tl) == 1
    ev = tl[0]
    assert ev["hold_entered_at"] == 0.0
    assert ev["hold_cycles"] == 2
    assert ev["probe_count"] == 2
    assert [p["outcome"] for p in ev["probes"]] == ["throttled", "ok"]
    assert ev["full_at"] is not None and ev["time_to_full_s"] == ev["full_at"] - 0.0
    assert "throttled" in md and "ok" in md


def test_timeline_without_probe_log_is_inferred():
    _th, m = drive_throttle(FakeClock(0.0))
    tl = throttle_timeline(m.throttle.stats)
    assert [p["outcome"] for p in tl[0]["probes"]] == ["throttled", "ok"]


def test_report_has_no_raw_key(tmp_path: Path):
    s, _m, _th = _summary()
    s["stop_reason"] = f"error: {RAW_KEY}"
    s["run_id"] = RAW_KEY
    rj, md = write_report(tmp_path, s)
    for f in ("report.json", "report.md"):
        assert (tmp_path / f).read_text(encoding="utf-8").count(RAW_KEY) == 0
    assert json.dumps(rj).count(RAW_KEY) == 0 and md.count(RAW_KEY) == 0


def test_throttle_json_roundtrip():
    th, _m = drive_throttle(FakeClock(0.0))
    js = throttle_json(th.stats)
    d = json.loads(js)
    assert d == json.loads(json.dumps(th.stats.to_dict()))
    assert d["probes_sent"] == 2
    assert set(d["time_in_state"]) == {"normal", "hold", "probe", "recover"}
    assert probes_count(th.stats) == 2 == probes_count(d)
    assert throttle_json(d) == js


def test_report_without_throttle_or_ok():
    rj, md = build_report({"stop_reason": "supply_refused", "dispatched": 0})
    assert rj["avg_tpm"] is None and rj["throttle_timeline"] == []
    assert rj["lang_share_pct"] == {"zh": 0.0, "en": 0.0}
    assert rj["warnings"]  # 无 OK 请求
    assert "supply_refused" in md
