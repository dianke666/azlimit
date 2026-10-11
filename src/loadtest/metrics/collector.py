"""MetricsCollector：运行期指标收集（Req 15.3, 16.5, 28.1）。

- 累计：各 Outcome 计数（普通请求与 Throttle_Probe 分开）、token 合计、语言计数、
  固定桶直方图（HDR 风格，不保存全部样本）的延迟 / 输入 token 百分位。
- 60 秒滑动窗口：派发 TPM（Σ input_tokens + max_tokens）/ RPM、延迟与输入 token P50/P95、
  平均输出 token（Σ completion / OK 数）；窗口内保留 (t, 值) 的 deque。
- 派发时长：``mark_dispatch_start`` → ``mark_dispatch_stop``（未调用 start 时取首次派发时刻）。
- 节流：可附加 ThrottleController，快照中包含 state / level / stats.to_dict()。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from loadtest.outcome import Outcome

# ---------------------------------------------------------------------------
# 固定桶直方图
# ---------------------------------------------------------------------------

_LINEAR_LIMIT = 2048  # [0, 2048) 每 1 一个桶（整数 token 精确）
_LOG_BASE = 1.01  # 之后按 1% 相对精度的对数桶


def _bucket(v: float) -> int:
    v = max(0.0, float(v))
    if v < _LINEAR_LIMIT:
        return int(v)
    return _LINEAR_LIMIT + int(math.log(v / _LINEAR_LIMIT) / math.log(_LOG_BASE))


def _bucket_value(b: int) -> float:
    if b < _LINEAR_LIMIT:
        return float(b)
    k = b - _LINEAR_LIMIT
    lo = _LINEAR_LIMIT * _LOG_BASE**k
    return lo * math.sqrt(_LOG_BASE)  # 桶内几何中点


class Histogram:
    """固定桶直方图；``percentile`` 相对误差 ≤ 1%（< 2048 时绝对误差 < 1）。"""

    def __init__(self) -> None:
        self.counts: dict[int, int] = {}
        self.n = 0
        self.total = 0.0
        self.min: float | None = None
        self.max: float | None = None

    def add(self, v: float) -> None:
        b = _bucket(v)
        self.counts[b] = self.counts.get(b, 0) + 1
        self.n += 1
        self.total += v
        self.min = v if self.min is None else min(self.min, v)
        self.max = v if self.max is None else max(self.max, v)

    def percentile(self, p: float) -> float | None:
        if self.n == 0:
            return None
        rank = max(1, math.ceil(p * self.n))
        acc = 0
        for b in sorted(self.counts):
            acc += self.counts[b]
            if acc >= rank:
                v = _bucket_value(b)
                # 夹在实际观测范围内
                return min(max(v, self.min or 0.0), self.max if self.max is not None else v)
        return self.max

    @property
    def mean(self) -> float | None:
        return self.total / self.n if self.n else None


def exact_percentile(values: Iterable[float], p: float) -> float | None:
    """最近秩法（nearest-rank）百分位。"""
    s = sorted(values)
    if not s:
        return None
    return float(s[max(1, math.ceil(p * len(s))) - 1])


# ---------------------------------------------------------------------------
# 复用分布
# ---------------------------------------------------------------------------


def reuse_distribution(run_uses: Mapping[Any, int]) -> dict[int, int]:
    """{使用次数: 段落数}（只统计使用次数 ≥ 1 的段落），按使用次数升序。"""
    dist: dict[int, int] = {}
    for n in run_uses.values():
        n = int(n)
        if n > 0:
            dist[n] = dist.get(n, 0) + 1
    return dict(sorted(dist.items()))


def max_run_uses(run_uses: Mapping[Any, int]) -> int:
    return max((int(n) for n in run_uses.values()), default=0)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def _outcome(o: Any) -> Outcome:
    return o if isinstance(o, Outcome) else Outcome(getattr(o, "value", o))


def _zero_outcomes() -> dict[str, int]:
    return {o.value: 0 for o in Outcome}


def _now_of(clock: Any) -> float:
    if clock is None:
        return 0.0
    if hasattr(clock, "now"):
        return float(clock.now())
    return float(clock())


class MetricsCollector:
    def __init__(self, clock: Any = None, window_s: float = 60.0, throttle: Any = None) -> None:
        self.clock = clock
        self.window_s = float(window_s)
        self.throttle = throttle
        self.created_at = _now_of(clock)

        # 累计
        self.dispatched = 0
        self.probes_dispatched = 0
        self.completed = 0
        self.outcomes = _zero_outcomes()  # 普通请求
        self.probe_outcomes = _zero_outcomes()  # Throttle_Probe
        self.lang_counts: dict[str, int] = {"zh": 0, "en": 0}
        self.input_tokens_total = 0  # 已派发请求 Σ input_tokens
        self.max_tokens_total = 0  # 已派发请求 Σ max_tokens
        self.prompt_tokens_total = 0  # 返回 usage 的 Σ prompt_tokens
        self.cached_tokens_total = 0
        self.completion_tokens_total = 0  # 全部结果 Σ completion_tokens
        self.ok_completion_tokens_total = 0  # OK 结果 Σ completion_tokens
        self.ok_count = 0  # OK（含探测）
        self.latency_hist = Histogram()
        self.input_hist = Histogram()
        self.probe_log: list[tuple[float, str]] = []  # (时刻, outcome)

        # 派发时长
        self.dispatch_start: float | None = None
        self.dispatch_stop: float | None = None

        # 窗口
        self._win_dispatch: deque[tuple[float, int, int]] = deque()  # (t, input, cost)
        self._win_result: deque[tuple[float, float, int, bool]] = deque()  # (t, latency, completion, ok)

    # ---- 附加 ----
    def attach_throttle(self, throttle: Any) -> None:
        self.throttle = throttle

    def mark_dispatch_start(self, now: float) -> None:
        if self.dispatch_start is None:
            self.dispatch_start = float(now)

    def mark_dispatch_stop(self, now: float) -> None:
        if self.dispatch_stop is None:
            self.dispatch_stop = float(now)

    def dispatch_duration_s(self, now: float | None = None) -> float:
        if self.dispatch_start is None:
            return 0.0
        end = self.dispatch_stop
        if end is None:
            end = _now_of(self.clock) if now is None else float(now)
        return max(0.0, end - self.dispatch_start)

    # ---- 记录 ----
    def record_dispatch(self, planned: Any, now: float, is_probe: bool = False) -> None:
        now = float(now)
        self.mark_dispatch_start(now)
        inp = int(planned.input_tokens)
        mx = int(planned.max_tokens)
        self.dispatched += 1
        if is_probe:
            self.probes_dispatched += 1
        lang = str(getattr(planned, "lang", "") or "")
        self.lang_counts[lang] = self.lang_counts.get(lang, 0) + 1
        self.input_tokens_total += inp
        self.max_tokens_total += mx
        self.input_hist.add(inp)
        self._win_dispatch.append((now, inp, inp + mx))
        self._evict(now)

    def record_result(self, planned: Any, result: Any, now: float, is_probe: bool = False) -> None:
        now = float(now)
        out = _outcome(result.outcome)
        self.completed += 1
        (self.probe_outcomes if is_probe else self.outcomes)[out.value] += 1
        if is_probe:
            self.probe_log.append((now, out.value))
        comp = int(getattr(result, "completion_tokens", None) or 0)
        self.completion_tokens_total += comp
        self.prompt_tokens_total += int(getattr(result, "prompt_tokens", None) or 0)
        self.cached_tokens_total += int(getattr(result, "cached_tokens", None) or 0)
        ok = out is Outcome.OK
        if ok:
            self.ok_count += 1
            self.ok_completion_tokens_total += comp
        lat = float(getattr(result, "latency_ms", 0.0) or 0.0)
        self.latency_hist.add(lat)
        self._win_result.append((now, lat, comp, ok))
        self._evict(now)

    def _evict(self, now: float) -> None:
        lo = now - self.window_s
        while self._win_dispatch and self._win_dispatch[0][0] <= lo:
            self._win_dispatch.popleft()
        while self._win_result and self._win_result[0][0] <= lo:
            self._win_result.popleft()

    # ---- 派生 ----
    @property
    def throttled_total(self) -> int:
        """节流计数（含探测，Req 20.5）。"""
        t = Outcome.THROTTLED.value
        return self.outcomes[t] + self.probe_outcomes[t]

    def avg_output_tokens(self) -> float | None:
        return self.ok_completion_tokens_total / self.ok_count if self.ok_count else None

    def window_stats(self, now: float) -> dict[str, Any]:
        now = float(now)
        self._evict(now)
        per_min = 60.0 / self.window_s
        disp = list(self._win_dispatch)
        res = list(self._win_result)
        ok = [r for r in res if r[3]]
        return {
            "tpm": sum(d[2] for d in disp) * per_min,
            "rpm": len(disp) * per_min,
            "input_p50": exact_percentile((d[1] for d in disp), 0.50),
            "input_p95": exact_percentile((d[1] for d in disp), 0.95),
            "latency_p50_ms": exact_percentile((r[1] for r in res), 0.50),
            "latency_p95_ms": exact_percentile((r[1] for r in res), 0.95),
            "avg_output_tokens": (sum(r[2] for r in ok) / len(ok)) if ok else None,
            "completed": len(res),
        }

    def lang_share(self) -> dict[str, float]:
        """各语言占已派发请求数的百分比（1 位小数）。"""
        if not self.dispatched:
            return {k: 0.0 for k in self.lang_counts}
        return {k: round(v * 100.0 / self.dispatched, 1) for k, v in self.lang_counts.items()}

    def throttle_snapshot(self) -> dict[str, Any] | None:
        t = self.throttle
        if t is None:
            return None
        state = getattr(t, "state", None)
        return {
            "state": getattr(state, "name", str(state)),
            "level": float(getattr(t, "level", 0.0)),
            "stats": t.stats.to_dict(),
        }

    def snapshot(self, now: float | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        now = _now_of(self.clock) if now is None else float(now)
        wall_fn: Callable[[], float] | None = getattr(self.clock, "wall", None)
        start = self.dispatch_start if self.dispatch_start is not None else self.created_at
        snap: dict[str, Any] = {
            "timestamp": float(wall_fn()) if wall_fn else None,
            "elapsed_s": max(0.0, now - start),
            "dispatch_duration_s": self.dispatch_duration_s(now),
            "window": self.window_stats(now),
            "dispatched": self.dispatched,
            "probes_dispatched": self.probes_dispatched,
            "completed": self.completed,
            "in_flight": max(0, self.dispatched - self.completed),
            "outcomes": dict(self.outcomes),
            "probe_outcomes": dict(self.probe_outcomes),
            "throttled_total": self.throttled_total,
            "lang_counts": dict(self.lang_counts),
            "lang_share": self.lang_share(),
            "input_tokens_total": self.input_tokens_total,
            "max_tokens_total": self.max_tokens_total,
            "prompt_tokens_total": self.prompt_tokens_total,
            "cached_tokens_total": self.cached_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "ok_count": self.ok_count,
            "avg_output_tokens_total": self.avg_output_tokens(),
            "input_p50_total": self.input_hist.percentile(0.50),
            "input_p95_total": self.input_hist.percentile(0.95),
            "latency_p50_ms_total": self.latency_hist.percentile(0.50),
            "latency_p95_ms_total": self.latency_hist.percentile(0.95),
        }
        th = self.throttle_snapshot()
        if th is not None:
            snap["throttle"] = th
        if extra:
            # 调用方提供：target_tpm、effective_level、ramp、in_flight、pool_active、pool_added、
            # max_run_uses、queue_len、cost 等
            snap.update(extra)
        return snap

    def summary(self, now: float | None = None, **extra: Any) -> dict[str, Any]:
        """供 report.build_report 使用的结束汇总。``extra`` 可含 stop_reason、run_uses、
        passages_added、cost、throttle_stats 等。"""
        now = _now_of(self.clock) if now is None else float(now)
        s: dict[str, Any] = {
            "dispatch_duration_s": self.dispatch_duration_s(now),
            "dispatched": self.dispatched,
            "probes_dispatched": self.probes_dispatched,
            "completed": self.completed,
            "outcomes": dict(self.outcomes),
            "probe_outcomes": dict(self.probe_outcomes),
            "lang_counts": dict(self.lang_counts),
            "input_tokens_total": self.input_tokens_total,
            "max_tokens_total": self.max_tokens_total,
            "prompt_tokens_total": self.prompt_tokens_total,
            "cached_tokens_total": self.cached_tokens_total,
            "completion_tokens_total": self.completion_tokens_total,
            "ok_completion_tokens_total": self.ok_completion_tokens_total,
            "ok_count": self.ok_count,
            "input_p50": self.input_hist.percentile(0.50),
            "input_p95": self.input_hist.percentile(0.95),
            "latency_p50_ms": self.latency_hist.percentile(0.50),
            "latency_p95_ms": self.latency_hist.percentile(0.95),
            "probe_log": [[t, o] for t, o in self.probe_log],
        }
        if self.throttle is not None:
            s["throttle_stats"] = self.throttle.stats.to_dict()
        s.update(extra)
        return s


__all__ = [
    "Histogram",
    "MetricsCollector",
    "exact_percentile",
    "max_run_uses",
    "reuse_distribution",
]
