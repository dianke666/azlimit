"""rich 控制台输出（Req 28.1）。调用方每 10 秒调用一次 ``ConsoleReporter.print``。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rich.console import Console
from rich.table import Table

from loadtest.settings import redact


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.{digits}f}"
    if isinstance(v, Mapping):
        return " ".join(f"{k}={_fmt(x, digits)}" for k, x in v.items())
    return str(v)


def _pct(v: Any) -> str:
    return "-" if v is None else f"{float(v) * 100:.0f}%"


def _rows(snap: Mapping[str, Any]) -> list[tuple[str, str]]:
    w = snap.get("window", {}) or {}
    rows: list[tuple[str, str]] = [("elapsed", f"{_fmt(float(snap.get('elapsed_s', 0.0)))}s")]

    th = snap.get("throttle")
    if th:
        st = th.get("stats", {}) or {}
        rows += [
            ("throttle state", str(th.get("state", "-"))),
            ("throttle level", _pct(th.get("level"))),
            ("throttle events / stale", f"{st.get('throttle_events', 0)} / {st.get('stale_throttles', 0)}"),
            (
                "probes sent/ok/429/err/cf",
                f"{st.get('probes_sent', 0)}/{st.get('probe_ok', 0)}/{st.get('probe_throttled', 0)}"
                f"/{st.get('probe_error', 0)}/{st.get('probe_content_filter', 0)}",
            ),
            ("max consecutive probe fail", _fmt(st.get("max_consecutive_probe_fail", 0))),
            ("soft downs", _fmt(st.get("soft_downs", 0))),
            (
                "time in state (s)",
                " ".join(
                    f"{k.upper()}={float(v):.1f}" for k, v in (st.get("time_in_state") or {}).items()
                ),
            ),
        ]
    rows += [
        ("effective level", _pct(snap.get("effective_level"))),
        ("ramp", _fmt(snap.get("ramp"))),
        ("target TPM", _fmt(snap.get("target_tpm"), 0)),
        ("TPM (60s)", _fmt(w.get("tpm"), 0)),
        ("RPM (60s)", _fmt(w.get("rpm"), 1)),
        ("in flight", _fmt(snap.get("in_flight"))),
        ("outcomes", _fmt(snap.get("outcomes", {}))),
        ("probe outcomes", _fmt(snap.get("probe_outcomes", {}))),
        ("lang share %", _fmt(snap.get("lang_share", {}))),
        ("input P50/P95 (60s)", f"{_fmt(w.get('input_p50'), 0)} / {_fmt(w.get('input_p95'), 0)}"),
        ("avg output tokens (60s)", _fmt(w.get("avg_output_tokens"))),
        (
            "latency P50/P95 ms (60s)",
            f"{_fmt(w.get('latency_p50_ms'), 0)} / {_fmt(w.get('latency_p95_ms'), 0)}",
        ),
        ("pool active / added", f"{_fmt(snap.get('pool_active'))} / {_fmt(snap.get('pool_added'))}"),
        ("max run uses", _fmt(snap.get("max_run_uses"))),
        ("replenish queue", _fmt(snap.get("queue_len"))),
        ("cost", _fmt(snap.get("cost"), 4)),
    ]
    return rows


def render(snapshot: Mapping[str, Any]) -> Table:
    snap = redact(dict(snapshot))
    table = Table(title="loadtest", show_header=True, header_style="bold")
    table.add_column("metric", no_wrap=True)
    table.add_column("value")
    for k, v in _rows(snap):
        table.add_row(k, str(redact(v)))
    return table


class ConsoleReporter:
    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print(self, snapshot: Mapping[str, Any]) -> None:
        self.console.print(render(snapshot))


__all__ = ["ConsoleReporter", "render"]
