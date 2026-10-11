"""ThrottleController：节流状态机（Azure / Gemini 共用实现）。

mode="probe"（Azure_Throttle）：NORMAL → HOLD → PROBE → RECOVER → NORMAL
    - Throttle_Event 立即 HOLD（allowed_dispatch == (0, 0)），hold_until = max(现值, now + cooldown)，
      cooldown = min(max(Retry-After, min_cooldown_s), max_retry_after_s)
    - hold 到期 → PROBE：恰好 1 个探测请求；只有 OK 探测进入 RECOVER
    - RECOVER：levels 逐级上升，每档 recover_step_s = recover_full_s / (len(levels) − 1)，到 100% → NORMAL
mode="level"（Gemini_Throttle）：NORMAL → HOLD → RECOVER → NORMAL（无探测，hold 期间档位 = levels[0]）

属性（设计 P-T1 … P-T5）：
    P-T1 hold_until 只增不减；HOLD 期间 allowed_dispatch == (0, 0)
    P-T2 PROBE 期间至多 1 个探测在途，结果返回前 max_new == 0
    P-T3 RECOVER 每次只升相邻一级，且 now − last_event ≥ recover_step_s
    P-T4 探测 OK → 100% ≥ recover_full_s；tick 间隔 ≤ 1s 时 ≤ recover_full_s + 1s
    P-T5 主动减速只降一级、不低于 soft_down_floor、不进入 HOLD、soft_down_cooldown_s 内最多一次
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import ThrottleSettings

_INF = math.inf
_EPS = 1e-9
_TIME_EPS = 1e-6  # 时间比较的浮点容差（秒）


class ThrottleState(Enum):
    NORMAL = "normal"
    HOLD = "hold"
    PROBE = "probe"
    RECOVER = "recover"


@dataclass(frozen=True)
class DispatchTicket:
    dispatched_at: float
    is_probe: bool


def _zero_times() -> dict[ThrottleState, float]:
    return {s: 0.0 for s in ThrottleState}


@dataclass
class ThrottleStats:
    throttle_events: int = 0  # 新的 Throttle_Event（导致进入 HOLD）
    stale_throttles: int = 0  # HOLD 前派发的在途请求返回的 429（只延长 hold 或被忽略）
    probes_sent: int = 0
    probe_ok: int = 0
    probe_throttled: int = 0
    probe_error: int = 0  # SERVER_ERROR / TIMEOUT / CLIENT_ERROR
    probe_content_filter: int = 0
    consecutive_probe_fail: int = 0  # 当前连续被节流的探测数
    max_consecutive_probe_fail: int = 0
    soft_downs: int = 0
    time_in_state: dict[ThrottleState, float] = field(default_factory=_zero_times)
    timeline: list[tuple[float, ThrottleState, ThrottleState]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON 兼容快照（供 runs.throttle_json / 报告使用）。"""
        return {
            "throttle_events": self.throttle_events,
            "stale_throttles": self.stale_throttles,
            "probes_sent": self.probes_sent,
            "probe_ok": self.probe_ok,
            "probe_throttled": self.probe_throttled,
            "probe_error": self.probe_error,
            "probe_content_filter": self.probe_content_filter,
            "consecutive_probe_fail": self.consecutive_probe_fail,
            "max_consecutive_probe_fail": self.max_consecutive_probe_fail,
            "soft_downs": self.soft_downs,
            "time_in_state": {s.value: v for s, v in self.time_in_state.items()},
            "timeline": [[t, a.value, b.value] for t, a, b in self.timeline],
        }


def _as_outcome(o: Any) -> Outcome:
    """接受 Outcome、同名枚举（按 value）或字符串。"""
    if isinstance(o, Outcome):
        return o
    return Outcome(getattr(o, "value", o))


def _ratio_low(remaining: int | None, limit: int | None, ratio: float) -> bool:
    # 缺失任一头或 limit ≤ 0 → 不判定该维度
    if remaining is None or limit is None or limit <= 0:
        return False
    return remaining / limit < ratio


class ThrottleController:
    def __init__(self, cfg: ThrottleSettings, clock: Callable[[], float]):
        self.cfg = cfg
        self.mode = cfg.mode
        self.levels: list[float] = list(cfg.levels)
        self.step_s: float = cfg.recover_step_s  # Azure 15s；Gemini 10s
        self.state = ThrottleState.NORMAL
        self._idx = len(self.levels) - 1  # 当前档位在 levels 中的下标
        self._floor_idx = min(
            i for i, x in enumerate(self.levels) if x >= cfg.soft_down_floor - _EPS
        )
        self.hold_until = -_INF
        self.hold_started_at = -_INF  # 当前 HOLD 周期的起点；早于它派发的请求为"在途旧请求"
        self.last_event = -_INF  # 最近一次 节流 / 主动减速 / 升档 / 探测成功
        self.last_soft = -_INF
        self.in_flight = 0  # 已派发、尚未完成的请求数
        self.probe_in_flight = False
        self.probe_phase_start = -_INF  # 进入 PROBE 的时刻
        self.probe_ready_at = -_INF  # 下一次允许发探测的时刻
        self.stats = ThrottleStats()
        self._last_acc = float(clock())

    # ---- 档位 ----
    @property
    def level(self) -> float:
        return self.levels[self._idx]

    @property
    def recover_step_s(self) -> float:
        return self.step_s

    # ---- 事件 ----
    def _cooldown(self, retry_after: float | None) -> float:
        ra = retry_after if retry_after is not None and retry_after > 0 else 0.0
        return min(max(ra, self.cfg.min_cooldown_s), self.cfg.max_retry_after_s)

    def on_throttle(
        self, retry_after: float | None, now: float, dispatched_at: float | None = None
    ) -> None:
        cooldown = self._cooldown(retry_after)
        if dispatched_at is not None and dispatched_at < self.hold_started_at:
            # HOLD 之前派发的在途请求：不改变状态与恢复进度
            self.stats.stale_throttles += 1
            if self.state is ThrottleState.HOLD:
                self.hold_until = max(self.hold_until, now + cooldown)
            return  # PROBE / RECOVER 中忽略
        self.stats.throttle_events += 1
        self._enter_hold(now, cooldown)

    def _enter_hold(self, now: float, cooldown: float) -> None:
        if self.state is not ThrottleState.HOLD:
            self.hold_started_at = now  # 新的 HOLD 周期
        self._set_state(ThrottleState.HOLD, now)
        self.hold_until = max(self.hold_until, now + cooldown)
        self._idx = 0  # level 模式下 hold 期间的档位
        self.last_event = now
        self.probe_in_flight = False

    def on_probe_result(self, outcome: Any, retry_after: float | None, now: float) -> None:
        outcome = _as_outcome(outcome)
        self.probe_in_flight = False
        st = self.stats
        if outcome is Outcome.OK:
            st.probe_ok += 1
            st.consecutive_probe_fail = 0
            if self.state is ThrottleState.PROBE:  # 只有 PROBE 中的 OK 探测进入 RECOVER
                self._set_state(ThrottleState.RECOVER, now)
                self._idx = 0
                self.last_event = now  # 从 levels[0] 开始计时
        elif outcome is Outcome.THROTTLED:
            st.probe_throttled += 1
            st.consecutive_probe_fail += 1
            st.max_consecutive_probe_fail = max(st.max_consecutive_probe_fail, st.consecutive_probe_fail)
            self._enter_hold(now, self._cooldown(retry_after))  # 新 HOLD 周期；探测次数不限
        elif outcome is Outcome.CONTENT_FILTER:
            st.probe_content_filter += 1
            self.probe_ready_at = now  # 不算成功，立即可再探
        else:  # SERVER_ERROR / TIMEOUT / CLIENT_ERROR
            st.probe_error += 1
            self.probe_ready_at = now + self.cfg.probe_error_retry_s

    def on_success(self, h: RateLimitHeaders | None, now: float) -> None:
        """主动减速（仅 NORMAL / RECOVER）。"""
        if h is None or self.state not in (ThrottleState.NORMAL, ThrottleState.RECOVER):
            return
        r = self.cfg.soft_down_ratio
        low = _ratio_low(h.remaining_tokens, h.limit_tokens, r) or _ratio_low(
            h.remaining_requests, h.limit_requests, r
        )
        if not low or now - self.last_soft < self.cfg.soft_down_cooldown_s:
            return
        if self._idx - 1 >= self._floor_idx:  # 1.0 → 0.75 → 0.50；下一级低于 floor 时保持
            self._idx -= 1
            self.stats.soft_downs += 1
            if self.state is ThrottleState.NORMAL:
                self._set_state(ThrottleState.RECOVER, now)
        self.last_soft = self.last_event = now  # 无论是否实际降档，都重置冷却与升档计时

    def on_result(self, ticket: DispatchTicket, res: Any, now: float) -> None:
        """worker 统一入口；res 需有 outcome、retry_after_s，可选 headers。"""
        outcome = _as_outcome(res.outcome)
        retry_after = getattr(res, "retry_after_s", None)
        if ticket.is_probe:
            self.on_probe_result(outcome, retry_after, now)
        elif outcome is Outcome.THROTTLED:
            self.on_throttle(retry_after, now, ticket.dispatched_at)
        elif outcome is Outcome.OK:
            self.on_success(getattr(res, "headers", None), now)

    # ---- 派发 ----
    def note_dispatch(self, now: float, is_probe: bool) -> DispatchTicket:
        self.in_flight += 1
        if is_probe:
            self.probe_in_flight = True
            self.stats.probes_sent += 1
        return DispatchTicket(dispatched_at=now, is_probe=is_probe)

    def note_done(self, ticket: DispatchTicket) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        if ticket.is_probe and self.probe_in_flight:
            # 探测未经 on_result 结束（如 worker 异常）：按探测错误处理，避免 PROBE 永久阻塞
            self.probe_in_flight = False
            self.probe_ready_at = self._last_acc + self.cfg.probe_error_retry_s

    def allowed_dispatch(self, now: float) -> tuple[int | None, float]:
        s = self.state
        if s is ThrottleState.HOLD:
            return 0, 0.0
        if s is ThrottleState.PROBE:
            if self.probe_in_flight or now < self.probe_ready_at:
                return 0, 0.0
            if self.in_flight > 0 and now - self.probe_phase_start < self.cfg.probe_inflight_wait_s:
                return 0, 0.0  # 等待 HOLD 前的在途请求返回（≤ probe_inflight_wait_s）
            return 1, self.levels[0]  # 恰好 1 个探测请求
        return None, self.level  # NORMAL / RECOVER

    # ---- 时钟 ----
    def tick(self, now: float) -> None:
        self._accumulate_time(now)
        if self.state is ThrottleState.HOLD:
            if now >= self.hold_until:
                if self.mode == "probe":
                    self._set_state(ThrottleState.PROBE, now)
                    self.probe_phase_start = self.probe_ready_at = now
                else:  # level 模式：直接恢复，计时从 hold 结束算起
                    self._set_state(ThrottleState.RECOVER, now)
                    self.last_event = self.hold_until
        elif self.state is ThrottleState.RECOVER and now - self.last_event >= self.step_s - _TIME_EPS:
            # _TIME_EPS 吸收浮点误差（如 25.4 − 10.4 = 14.999…），避免每档多等一个 tick
            if self._idx < len(self.levels) - 1:
                self._idx += 1
            self.last_event = now
            if self._idx == len(self.levels) - 1:
                self._set_state(ThrottleState.NORMAL, now)

    def is_holding(self, now: float) -> bool:
        return self.state is ThrottleState.HOLD and now < self.hold_until

    def _set_state(self, s: ThrottleState, now: float) -> None:
        self._accumulate_time(now)
        if s is not self.state:
            self.stats.timeline.append((now, self.state, s))
            self.state = s

    def _accumulate_time(self, now: float) -> None:
        dt = now - self._last_acc
        if dt > 0:
            self.stats.time_in_state[self.state] += dt
            self._last_acc = now


__all__ = ["ThrottleState", "DispatchTicket", "ThrottleStats", "ThrottleController"]
