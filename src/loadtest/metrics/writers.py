"""JSONL（每请求一行）与 CSV（每 10 秒一行）输出（Req 1.7, 28.2, 28.3）。

- JSONL 只写元数据：不包含消息原文、模型输出与任何密钥；全部内容经 ``redact``。
- CSV 列 = timestamp + 快照中全部数值字段（嵌套字段以 ``.`` 拼接，如
  ``throttle.stats.time_in_state.hold``）+ throttle 状态；表头在首行确定。
"""

from __future__ import annotations

import csv
import io
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loadtest.settings import redact


def iso_utc(ts: float | None) -> str:
    ts = time.time() if ts is None else float(ts)
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="milliseconds")


def _val(o: Any) -> Any:
    return getattr(o, "value", o)


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def result_record(planned: Any, result: Any, now_wall: float | None, is_probe: bool) -> dict[str, Any]:
    """单请求 JSONL 记录（白名单字段，绝不含 messages / text / headers）。"""
    return {
        "timestamp": iso_utc(now_wall),
        "request_id": str(getattr(result, "request_id", None) or planned.request_id),
        "lang": str(planned.lang),
        "task_id": str(planned.task_id),
        "passage_ids": [int(p) for p in planned.passage_ids],
        "input_tokens": int(planned.input_tokens),
        "prompt_tokens": getattr(result, "prompt_tokens", None),
        "output_tokens": getattr(result, "completion_tokens", None),
        "outcome": str(_val(result.outcome)),
        "status": getattr(result, "status", None),
        "latency_ms": round(float(getattr(result, "latency_ms", 0.0) or 0.0), 3),
        "is_probe": bool(is_probe),
    }


class JsonlWriter:
    def __init__(self, path: str | Path, flush_every: int = 200) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = max(1, int(flush_every))
        self._buf: list[str] = []

    def write_result(
        self, planned: Any, result: Any, now_wall: float | None = None, is_probe: bool = False
    ) -> None:
        rec = redact(result_record(planned, result, now_wall, is_probe))
        self._buf.append(json.dumps(rec, ensure_ascii=False))
        if len(self._buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        data = "\n".join(self._buf) + "\n"
        self._buf.clear()
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(data)

    def close(self) -> None:
        self.flush()

    @property
    def pending(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def flatten_numeric(d: Mapping[str, Any], prefix: str = "") -> dict[str, float | int | None]:
    """展开嵌套 dict 中的数值字段（bool 视为 0/1；None 保留为空）；跳过字符串与列表。"""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            out.update(flatten_numeric(v, key + "."))
        elif isinstance(v, bool):
            out[key] = int(v)
        elif isinstance(v, (int, float)) or v is None:
            out[key] = v
    return out


def csv_row(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    snap = dict(snapshot)
    ts = snap.pop("timestamp", None)
    row: dict[str, Any] = {"timestamp": iso_utc(ts if isinstance(ts, (int, float)) else None)}
    th = snap.get("throttle")
    if isinstance(th, Mapping) and "state" in th:
        row["throttle_state"] = str(th["state"])
    ramp = snap.get("ramp")
    if isinstance(ramp, str):
        row["ramp"] = ramp
    row.update(flatten_numeric(snap))
    return row


class CsvWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.columns: list[str] | None = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), None)
            self.columns = list(header) if header else None

    def write_row(self, snapshot: Mapping[str, Any]) -> None:
        row = redact(csv_row(snapshot))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        if self.columns is None:
            self.columns = list(row.keys())
            w.writerow(self.columns)
        w.writerow(["" if row.get(c) is None else row.get(c) for c in self.columns])
        with self.path.open("a", encoding="utf-8", newline="") as f:
            f.write(buf.getvalue())

    def close(self) -> None:  # 每行即时写入
        pass


__all__ = ["CsvWriter", "JsonlWriter", "csv_row", "flatten_numeric", "iso_utc", "result_record"]
