"""StartupRamp 与 Effective_Level（Requirement 18）。

- StartupRamp：阶梯函数，默认 [(0, 0.5), (10, 1.0)]；`--no-ramp` 时恒为 1.0。
  已运行时间零点 = Scheduler 主循环开始时刻（供应检查与 Probe_Request 之后，18.5），
  调用方传入 elapsed = now − t_start。
- Effective_Level（18.4）：
    NORMAL / RECOVER → min(ramp, throttle.level)
    HOLD             → 0.0（不派发）
    PROBE            → levels[0]（Throttle_Probe 使用的档位）
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from loadtest.throttle import ThrottleState

DEFAULT_STEPS: tuple[tuple[float, float], ...] = ((0.0, 0.5), (10.0, 1.0))


def ramp_factor(steps: Sequence[tuple[float, float]], elapsed: float) -> float:
    """返回 max{f | (t, f) ∈ steps, t <= elapsed}；elapsed < 0 时取首步档位。

    前置条件（由 Settings 校验保证）：时间严格递增、首步时间 0、档位 ∈ (0, 1]、末步档位 1.0。
    """
    f = steps[0][1]
    for t, v in steps:
        if elapsed >= t:
            f = v
        else:
            break
    return f


class StartupRamp:
    def __init__(self, steps: Sequence[tuple[float, float]] | None = None, enabled: bool = True):
        steps = tuple((float(t), float(v)) for t, v in (steps if steps else DEFAULT_STEPS))
        if steps[0][0] != 0.0 or steps[-1][1] != 1.0:
            raise ValueError("startup_ramp must start at t=0 and end at level 1.0")
        if any(b[0] <= a[0] for a, b in zip(steps, steps[1:])):
            raise ValueError("startup_ramp times must be strictly increasing")
        if any(not (0.0 < v <= 1.0) for _, v in steps):
            raise ValueError("startup_ramp levels must be in (0, 1]")
        # --no-ramp：自启动起恒为 100%
        self.steps: tuple[tuple[float, float], ...] = steps if enabled else ((0.0, 1.0),)
        self.enabled = enabled

    @property
    def levels(self) -> frozenset[float]:
        """爬坡可能返回的档位集合。"""
        return frozenset(v for _, v in self.steps)

    def factor(self, elapsed_s: float) -> float:
        return ramp_factor(self.steps, elapsed_s)

    def done(self, elapsed_s: float) -> bool:
        return elapsed_s >= self.steps[-1][0]


class _ThrottleLike(Protocol):
    state: ThrottleState
    levels: list[float]

    @property
    def level(self) -> float: ...


def effective_level(ramp: float, throttle: _ThrottleLike) -> float:
    """按 throttle 状态给出有效档位（18.4）。HOLD 返回 0.0 表示不派发。"""
    s = throttle.state
    if s is ThrottleState.HOLD:
        return 0.0
    if s is ThrottleState.PROBE:
        return throttle.levels[0]
    return min(ramp, throttle.level)


__all__ = ["DEFAULT_STEPS", "ramp_factor", "StartupRamp", "effective_level"]
