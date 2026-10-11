"""运行报告 report.json / report.md（Req 15.3, 16.5, 28.4, 28.5）。

``build_report(summary)`` 的 summary 一般来自 ``MetricsCollector.summary(stop_reason=..., run_uses=...,
passages_added=..., cost=...)``；识别的键：
    stop_reason, run_id, dispatch_duration_s, dispatched, probes_dispatched, completed,
    outcomes, probe_outcomes, lang_counts, input_tokens_total, max_tokens_total,
    prompt_tokens_total, cached_tokens_total, completion_tokens_total,
    ok_completion_tokens_total, ok_count, input_p50/p95, latency_p50_ms/p95_ms,
    run_uses 或 reuse_distribution, passages_added, cost, throttle_stats（ThrottleStats 或 dict）,
    probe_log（[[t, outcome], ...]）
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loadtest.metrics.collector import max_run_uses, reuse_distribution
from loadtest.settings import redact

OUTPUT_TOKENS_RANGE = (20.0, 40.0)


# ---------------------------------------------------------------------------
# ThrottleStats 序列化（runs.probes / runs.throttle_json）
# ---------------------------------------------------------------------------


def _stats_dict(stats: Any) -> dict[str, Any]:
    if stats is None:
        return {}
    if isinstance(stats, Mapping):
        return dict(stats)
    return stats.to_dict()


def throttle_json(stats: Any) -> str:
    """ThrottleStats（或其 to_dict 结果）→ JSON 字符串，可 ``json.loads`` 还原。"""
    return json.dumps(_stats_dict(stats), ensure_ascii=False, sort_keys=True)


def probes_count(stats: Any) -> int:
    """写入 runs.probes 的探测发送数。"""
    return int(_stats_dict(stats).get("probes_sent", 0) or 0)


# ---------------------------------------------------------------------------
# Throttle_Event 时间线
# ---------------------------------------------------------------------------


def throttle_timeline(stats: Any, probe_log: list | None = None) -> list[dict[str, Any]]:
    """按 Throttle_Event 汇总：进入 HOLD 时刻、HOLD 周期数、探测次数与结果、进入 RECOVER 时刻、回满时刻。

    状态转换来自 ``stats.timeline``；探测结果优先取 ``probe_log``（含 err / cf），否则由
    PROBE→RECOVER（ok）/ PROBE→HOLD（throttled）推断。
    """
    d = _stats_dict(stats)
    transitions = [(float(t), str(a), str(b)) for t, a, b in d.get("timeline", [])]
    events: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for t, a, b in transitions:
        if b == "hold" and a != "probe":
            cur = {
                "hold_entered_at": t,
                "hold_cycles": 1,
                "probe_started_at": None,
                "recover_at": None,
                "full_at": None,
                "time_to_full_s": None,
                "probes": [],
            }
            events.append(cur)
        elif cur is None:
            continue  # 节流事件之外的转换（如主动减速 NORMAL→RECOVER）
        elif b == "hold":  # PROBE → HOLD：探测被节流，新 HOLD 周期
            cur["hold_cycles"] += 1
            if probe_log is None:
                cur["probes"].append({"t": t, "outcome": "throttled"})
        elif b == "probe" and cur["probe_started_at"] is None:
            cur["probe_started_at"] = t
        elif b == "recover":
            if a == "probe" and probe_log is None:
                cur["probes"].append({"t": t, "outcome": "ok"})
            cur["recover_at"] = t
        elif b == "normal" and a == "recover":
            cur["full_at"] = t
            cur["time_to_full_s"] = t - cur["hold_entered_at"]
            cur = None
    if probe_log is not None:
        for t, o in probe_log:
            t = float(t)
            owner = None
            for ev in events:
                if ev["hold_entered_at"] <= t:
                    owner = ev
            if owner is not None:
                owner["probes"].append({"t": t, "outcome": str(o)})
    for ev in events:
        ev["probe_count"] = len(ev["probes"])
    return events


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _pct1(n: int, total: int) -> float:
    return round(n * 100.0 / total, 1) if total else 0.0


def build_report(summary: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    s = dict(summary)
    outcomes: dict[str, int] = dict(s.get("outcomes", {}) or {})
    probe_outcomes: dict[str, int] = dict(s.get("probe_outcomes", {}) or {})
    dispatched = int(s.get("dispatched", 0) or 0)
    inp = int(s.get("input_tokens_total", 0) or 0)
    mx = int(s.get("max_tokens_total", 0) or 0)
    dur_s = float(s.get("dispatch_duration_s", 0.0) or 0.0)
    minutes = dur_s / 60.0
    ok = int(s.get("ok_count", 0) or 0)
    ok_comp = int(s.get("ok_completion_tokens_total", s.get("completion_tokens_total", 0)) or 0)
    avg_out = ok_comp / ok if ok else None

    lang_counts: dict[str, int] = dict(s.get("lang_counts", {}) or {})
    for k in ("zh", "en"):
        lang_counts.setdefault(k, 0)
    lang_share = {k: _pct1(v, dispatched) for k, v in lang_counts.items()}

    if "reuse_distribution" in s:
        reuse = {int(k): int(v) for k, v in dict(s["reuse_distribution"]).items()}
        max_uses = int(s.get("max_run_uses", max(reuse, default=0)))
    else:
        run_uses = dict(s.get("run_uses", {}) or {})
        reuse = reuse_distribution(run_uses)
        max_uses = max_run_uses(run_uses)

    stats = _stats_dict(s.get("throttle_stats"))
    timeline = throttle_timeline(stats, s.get("probe_log")) if stats else []
    throttled = int(outcomes.get("throttled", 0)) + int(probe_outcomes.get("throttled", 0))

    warnings: list[str] = []
    lo, hi = OUTPUT_TOKENS_RANGE
    if avg_out is None:
        warnings.append("没有 OK 请求，无法计算平均输出 token")
    elif not (lo <= avg_out <= hi):
        warnings.append(
            f"平均输出 token {avg_out:.1f} 偏离预期范围 [{lo:.0f}, {hi:.0f}]，请检查提示词 / max_tokens"
        )

    rj: dict[str, Any] = {
        "run_id": s.get("run_id"),
        "stop_reason": s.get("stop_reason"),
        "dispatch_duration_s": dur_s,
        "requests": {
            "dispatched": dispatched,
            "probes_dispatched": int(s.get("probes_dispatched", 0) or 0),
            "completed": int(s.get("completed", 0) or 0),
            "ok": ok,
            "outcomes": outcomes,
            "probe_outcomes": probe_outcomes,
        },
        "tokens": {
            "input_tokens_total": inp,
            "max_tokens_total": mx,
            "prompt_tokens_total": int(s.get("prompt_tokens_total", 0) or 0),
            "cached_tokens_total": int(s.get("cached_tokens_total", 0) or 0),
            "completion_tokens_total": int(s.get("completion_tokens_total", 0) or 0),
            "avg_output_tokens": avg_out,
            "input_p50": s.get("input_p50"),
            "input_p95": s.get("input_p95"),
        },
        "latency": {"p50_ms": s.get("latency_p50_ms"), "p95_ms": s.get("latency_p95_ms")},
        "avg_tpm": (inp + mx) / minutes if minutes > 0 else None,
        "avg_rpm": dispatched / minutes if minutes > 0 else None,
        "throttled": throttled,
        "lang_counts": lang_counts,
        "lang_share_pct": lang_share,
        "reuse_distribution": {str(k): v for k, v in reuse.items()},
        "max_run_uses": max_uses,
        "passages_added": int(s.get("passages_added", 0) or 0),
        "cost": s.get("cost"),
        "throttle_stats": stats,
        "throttle_timeline": timeline,
        "probes": probes_count(stats),
        "warnings": warnings,
    }
    rj = redact(rj)
    return rj, redact(_markdown(rj))


def _f(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _markdown(r: Mapping[str, Any]) -> str:
    L: list[str] = []
    a = L.append
    a(f"# Load test report{(' ' + str(r['run_id'])) if r.get('run_id') else ''}")
    a("")
    a(f"- stop_reason: {r.get('stop_reason')}")
    a(f"- dispatch duration: {_f(r['dispatch_duration_s'])} s")
    a(f"- avg TPM: {_f(r['avg_tpm'], 0)}")
    a(f"- avg RPM: {_f(r['avg_rpm'], 1)}")
    a(f"- throttled: {r['throttled']}")
    a(f"- passages added: {r['passages_added']}")
    cost = r.get("cost")
    if isinstance(cost, Mapping):
        a("- cost: " + ", ".join(f"{k}={_f(v, 4)}" for k, v in cost.items()))
    else:
        a(f"- cost: {_f(cost, 4)}")
    a("")
    if r["warnings"]:
        a("## Warnings")
        a("")
        for w in r["warnings"]:
            a(f"- ⚠ {w}")
        a("")

    rq = r["requests"]
    a("## Requests")
    a("")
    a(f"- dispatched: {rq['dispatched']} (probes: {rq['probes_dispatched']})")
    a(f"- completed: {rq['completed']}, ok: {rq['ok']}")
    a("")
    a("| outcome | requests | probes |")
    a("|---|---|---|")
    for k in sorted(set(rq["outcomes"]) | set(rq["probe_outcomes"])):
        a(f"| {k} | {rq['outcomes'].get(k, 0)} | {rq['probe_outcomes'].get(k, 0)} |")
    a("")

    tk = r["tokens"]
    a("## Tokens")
    a("")
    for k in (
        "input_tokens_total",
        "max_tokens_total",
        "prompt_tokens_total",
        "cached_tokens_total",
        "completion_tokens_total",
        "avg_output_tokens",
        "input_p50",
        "input_p95",
    ):
        a(f"- {k}: {_f(tk.get(k))}")
    a(f"- latency P50 / P95 (ms): {_f(r['latency']['p50_ms'])} / {_f(r['latency']['p95_ms'])}")
    a("")

    a("## Language share")
    a("")
    a("| lang | requests | share |")
    a("|---|---|---|")
    for k, v in r["lang_counts"].items():
        a(f"| {k} | {v} | {r['lang_share_pct'][k]:.1f}% |")
    a("")

    a("## Passage reuse distribution")
    a("")
    a(f"- max run uses: {r['max_run_uses']}")
    a("")
    a("| uses | passages |")
    a("|---|---|")
    for k, v in r["reuse_distribution"].items():
        a(f"| {k} | {v} |")
    a("")

    st = r["throttle_stats"]
    a("## ThrottleStats")
    a("")
    if st:
        a("| field | value |")
        a("|---|---|")
        for k, v in st.items():
            if k in ("time_in_state", "timeline"):
                continue
            a(f"| {k} | {_f(v)} |")
        a("")
        a("| state | time_in_state (s) |")
        a("|---|---|")
        for k, v in (st.get("time_in_state") or {}).items():
            a(f"| {k} | {_f(float(v))} |")
        a("")
        a("### State transitions")
        a("")
        for t, x, y in st.get("timeline", []):
            a(f"- t={_f(float(t))}s {x} → {y}")
        a("")
    else:
        a("- (no throttle stats)")
        a("")

    a("## Throttle event timeline")
    a("")
    if r["throttle_timeline"]:
        a("| # | hold entered (s) | hold cycles | probe started (s) | probes | probe results | recover (s) | full (s) | time to full (s) |")
        a("|---|---|---|---|---|---|---|---|---|")
        for i, ev in enumerate(r["throttle_timeline"], 1):
            results = ", ".join(f"{_f(p['t'])}:{p['outcome']}" for p in ev["probes"]) or "-"
            a(
                f"| {i} | {_f(ev['hold_entered_at'])} | {ev['hold_cycles']} | {_f(ev['probe_started_at'])} "
                f"| {ev['probe_count']} | {results} | {_f(ev['recover_at'])} | {_f(ev['full_at'])} "
                f"| {_f(ev['time_to_full_s'])} |"
            )
    else:
        a("- (no throttle events)")
    a("")
    return "\n".join(L)


def write_report(out_dir: str | Path, summary: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """生成并写入 ``out_dir/report.json`` 与 ``out_dir/report.md``。"""
    rj, md = build_report(summary)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(rj, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "report.md").write_text(md, encoding="utf-8")
    return rj, md


__all__ = ["build_report", "probes_count", "throttle_json", "throttle_timeline", "write_report"]
