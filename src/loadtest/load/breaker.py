"""CircuitBreaker 熔断器（需求 23）。

状态机::

    closed --(60s 窗口样本 ≥ 20 且错误率 > 50%，或连续失败 ≥ 10)--> open
    open   --(open_s = 60s 后)--> half_open
    half_open --(5 个探测全部成功)--> closed
    half_open --(任一探测失败)--> open

样本规则（23.1）：只有 OK 与 SERVER_ERROR / TIMEOUT / CLIENT_ERROR 计入；
THROTTLED 与 CONTENT_FILTER 既不计入分子分母，也不重置连续失败计数；仅 OK 重置。

调度器用法（23.3 / 23.6）::

    if not breaker.allow_dispatch(now):   # open，或 half_open 探测名额已用完
        await sleep_ticking(...)          # breaker.retry_at 为 open 结束时刻
    ...
    breaker.on_dispatch(now)              # 真正派发时占用 half_open 探测名额
    breaker.release_probe()               # 若占用后未能派发（预留失败等）则归还

open_count 由 RunController 读取：第 3 次 open → stop_reason=circuit_open（任务 16.1）。
时钟全部由调用方以 ``now`` 注入。
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from loadtest.outcome import Outcome

State = Literal["closed", "open", "half_open"]

_FAILURES = frozenset({Outcome.SERVER_ERROR, Outcome.TIMEOUT, Outcome.CLIENT_ERROR})


class CircuitBreaker:
    def __init__(
        self,
        *,
        window_s: float = 60.0,
        min_samples: int = 20,
        error_rate: float = 0.5,
        consecutive_failures: int = 10,
        open_s: float = 60.0,
        half_open_probes: int = 5,
    ) -> None:
        self.window_s = window_s
        self.min_samples = min_samples
        self.error_rate = error_rate
        self.consecutive_failures = consecutive_failures
        self.open_s = open_s
        self.half_open_probes = half_open_probes

        self._state: State = "closed"
        self._window: deque[tuple[float, bool]] = deque()  # (t, is_failure)
        self._window_failures = 0
        self._consecutive = 0
        self._open_until = 0.0
        self._probes_dispatched = 0
        self._probe_successes = 0
        self.open_count = 0

    # ------------------------------------------------------------------ 查询
    def state(self, now: float) -> State:
        self._advance(now)
        return self._state

    @property
    def retry_at(self) -> float:
        """open 状态结束（进入 half_open）的时刻；非 open 时为 0.0。"""
        return self._open_until if self._state == "open" else 0.0

    @property
    def consecutive(self) -> int:
        return self._consecutive

    def half_open_remaining(self, now: float) -> int | None:
        """half_open 下剩余可派发探测数；closed 为 None（不限）；open 为 0。"""
        st = self.state(now)
        if st == "closed":
            return None
        if st == "open":
            return 0
        return max(0, self.half_open_probes - self._probes_dispatched)

    def allow_dispatch(self, now: float) -> bool:
        rem = self.half_open_remaining(now)
        return rem is None or rem > 0

    # ------------------------------------------------------------------ 派发
    def on_dispatch(self, now: float) -> None:
        """实际派发一个请求时调用；half_open 下占用一个探测名额。"""
        if self.state(now) == "half_open":
            self._probes_dispatched += 1

    def release_probe(self) -> None:
        """占用名额后未能真正发送请求时归还。"""
        if self._state == "half_open" and self._probes_dispatched > 0:
            self._probes_dispatched -= 1

    # ------------------------------------------------------------------ 记录
    def record(self, outcome: Outcome | bool, now: float) -> None:
        """记录一个结果。``outcome`` 可为 Outcome，或 bool（True=OK，False=失败）。"""
        if isinstance(outcome, bool):
            failed = not outcome
        elif outcome is Outcome.OK:
            failed = False
        elif outcome in _FAILURES:
            failed = True
        else:  # THROTTLED / CONTENT_FILTER：完全忽略
            return

        st = self.state(now)
        if st == "open":
            # open 期间到达的是 open 前派发的在途请求，不影响状态
            return
        if st == "half_open":
            if failed:
                self._open(now)
                return
            self._probe_successes += 1
            if self._probe_successes >= self.half_open_probes:
                self._close()
            return

        # closed
        self._consecutive = self._consecutive + 1 if failed else 0
        self._window.append((now, failed))
        if failed:
            self._window_failures += 1
        self._prune(now)
        n = len(self._window)
        if self._consecutive >= self.consecutive_failures or (
            n >= self.min_samples and self._window_failures / n > self.error_rate
        ):
            self._open(now)

    # ------------------------------------------------------------------ 内部
    def _advance(self, now: float) -> None:
        if self._state == "open" and now >= self._open_until:
            self._state = "half_open"
            self._probes_dispatched = 0
            self._probe_successes = 0

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._window and self._window[0][0] <= cutoff:
            _, f = self._window.popleft()
            if f:
                self._window_failures -= 1

    def _reset_samples(self) -> None:
        self._window.clear()
        self._window_failures = 0
        self._consecutive = 0

    def _open(self, now: float) -> None:
        self._state = "open"
        self._open_until = now + self.open_s
        self.open_count += 1
        self._probes_dispatched = 0
        self._probe_successes = 0
        self._reset_samples()

    def _close(self) -> None:
        self._state = "closed"
        self._probes_dispatched = 0
        self._probe_successes = 0
        self._reset_samples()


__all__ = ["CircuitBreaker", "State"]
