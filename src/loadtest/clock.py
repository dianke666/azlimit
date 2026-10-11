"""时钟抽象：所有时间相关组件通过注入 Clock 获取时间与 sleep，测试中使用 FakeClock。

- ``Clock`` 协议：``now()`` 返回单调秒数；``sleep(s)`` 为异步等待；实例可直接调用（``clock()``
  等价于 ``clock.now()``），以匹配设计中 ``self.clock()`` 的写法。
- ``RealClock``：基于 ``time.monotonic`` 与 ``asyncio.sleep``；``wall()`` 返回 UTC 墙钟时间戳。
- ``FakeClock``：手动推进时间；``sleep`` 默认把时间推进到唤醒点后让出事件循环一次，
  不真正等待；也可注入自定义 sleep 实现（例如离散事件模拟器）。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

SleepFn = Callable[[float], Awaitable[None]]


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float:
        """单调时间（秒）。"""
        ...

    def wall(self) -> float:
        """墙钟时间（UTC epoch 秒），用于过期时间与 UTC 日边界。"""
        ...

    async def sleep(self, seconds: float) -> None: ...

    def __call__(self) -> float: ...


class RealClock:
    """生产时钟。"""

    def now(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))

    def __call__(self) -> float:
        return self.now()


class FakeClock:
    """可控时钟：时间只在 ``advance`` / ``set`` / ``sleep`` 时前进。

    参数:
        start: 初始单调时间。
        wall_offset: ``wall() = now() + wall_offset``。
        sleep_fn: 可选的注入 sleep；提供时 ``sleep`` 直接委托给它（不自动推进时间）。
    """

    def __init__(
        self,
        start: float = 0.0,
        wall_offset: float = 1_700_000_000.0,
        sleep_fn: SleepFn | None = None,
    ) -> None:
        self._t = float(start)
        self._wall_offset = float(wall_offset)
        self._sleep_fn = sleep_fn
        self.sleeps: list[float] = []  # 记录每次 sleep 请求的时长，便于断言

    def now(self) -> float:
        return self._t

    def wall(self) -> float:
        return self._t + self._wall_offset

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self._t += seconds
        return self._t

    def set(self, t: float) -> float:
        if t < self._t:
            raise ValueError("FakeClock cannot move backwards")
        self._t = float(t)
        return self._t

    async def sleep(self, seconds: float) -> None:
        seconds = max(0.0, float(seconds))
        self.sleeps.append(seconds)
        if self._sleep_fn is not None:
            await self._sleep_fn(seconds)
            return
        self._t += seconds
        await asyncio.sleep(0)  # 让出事件循环，但不真正等待


__all__ = ["Clock", "RealClock", "FakeClock", "SleepFn"]
