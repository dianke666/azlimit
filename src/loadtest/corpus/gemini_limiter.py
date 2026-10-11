"""GeminiRateLimiter：Gemini 侧双桶限流 + 独立的 level 模式节流器（组件 4）。

速率（Requirement 11.1）:
    RPM = quota_rpm × usage_ratio × throttle.level（可选 rpm_cap 取 min）
    TPM = quota_tpm × usage_ratio × throttle.level（可选 tpm_cap 取 min）
节流（Requirement 11.2 / 11.7）:
    Gemini_Throttle 为 ``ThrottleController(mode="level")``；HOLD 期间 ``acquire`` 以 ≤ 1s 步长
    等待并调用 ``throttle.tick``，不放行任何调用。
等待:
    每次尝试前先 ``tick`` + 同步档位到桶速率，再 ``try_acquire``；不足时按估计等待时长
    （截断到 ≤ 0.25s）睡眠后重试，因此档位变化能在下一步立即生效。
    自身的 asyncio.Lock 保证等待者 FIFO。
预热与后台补充各自创建独立实例（Requirement 10.1：补充侧 rpm_cap = replenish_rpm ≤ 30）。
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from loadtest.bucket import DualTokenBucket
from loadtest.clock import Clock
from loadtest.settings import ThrottleSettings
from loadtest.throttle import ThrottleController

if TYPE_CHECKING:
    from loadtest.settings import GeminiSettings

# 单个 Gemini 调用（提示词 + 输出）约 3000 token，留足余量
GEMINI_MIN_TOK_CAP = 20_000
HOLD_STEP_S = 1.0
WAIT_STEP_S = 0.25


def gemini_throttle_settings() -> ThrottleSettings:
    """Gemini_Throttle 默认配置：level 模式，10% → 50% → 100% 每档 10s。"""
    return ThrottleSettings(mode="level", levels=[0.10, 0.50, 1.00], recover_full_s=20.0)


class GeminiRateLimiter:
    def __init__(
        self,
        quota_rpm: float = 600,
        quota_tpm: float = 800_000,
        usage_ratio: float = 0.5,
        *,
        clock: Clock,
        throttle: ThrottleController | None = None,
        throttle_cfg: ThrottleSettings | None = None,
        rpm_cap: float | None = None,
        tpm_cap: float | None = None,
        burst_s: float = 5.0,
        initial_fill_s: float = 2.0,
        min_tok_cap: float = GEMINI_MIN_TOK_CAP,
        hold_step_s: float = HOLD_STEP_S,
        wait_step_s: float = WAIT_STEP_S,
    ) -> None:
        if quota_rpm <= 0 or quota_tpm <= 0:
            raise ValueError("quota_rpm / quota_tpm must be > 0")
        if not 0 < usage_ratio <= 1:
            raise ValueError("usage_ratio must be in (0, 1]")
        if rpm_cap is not None and rpm_cap < 0:
            raise ValueError("rpm_cap must be >= 0")
        if tpm_cap is not None and tpm_cap < 0:
            raise ValueError("tpm_cap must be >= 0")
        if not 0 < hold_step_s <= HOLD_STEP_S:
            raise ValueError(f"hold_step_s must be in (0, {HOLD_STEP_S}]")
        if not 0 < wait_step_s <= WAIT_STEP_S:
            raise ValueError(f"wait_step_s must be in (0, {WAIT_STEP_S}]")
        self.clock = clock
        self.quota_rpm = float(quota_rpm)
        self.quota_tpm = float(quota_tpm)
        self.usage_ratio = float(usage_ratio)
        self.rpm_cap = rpm_cap
        self.tpm_cap = tpm_cap
        self.hold_step_s = float(hold_step_s)
        self.wait_step_s = float(wait_step_s)
        if throttle is None:
            cfg = throttle_cfg or gemini_throttle_settings()
            if cfg.mode != "level":
                raise ValueError("Gemini_Throttle must use mode='level'")
            throttle = ThrottleController(cfg, clock)
        self.throttle = throttle
        self._level = throttle.level
        rpm, tpm = self.target_rates(self._level)
        self.bucket = DualTokenBucket(
            tpm,
            rpm,
            burst_s=burst_s,
            initial_fill_s=initial_fill_s,
            clock=clock,
            min_tok_cap=min_tok_cap,
            max_wait_step_s=wait_step_s,
        )
        self._lock = asyncio.Lock()

    @classmethod
    def from_settings(
        cls,
        gs: GeminiSettings,
        clock: Clock,
        *,
        replenisher: bool = False,
        throttle: ThrottleController | None = None,
    ) -> GeminiRateLimiter:
        """预热实例（replenisher=False）或后台补充实例（rpm_cap = replenish_rpm）。"""
        return cls(
            gs.quota_rpm,
            gs.quota_tpm,
            gs.usage_ratio,
            clock=clock,
            throttle=throttle,
            throttle_cfg=gs.throttle,
            rpm_cap=gs.replenish_rpm if replenisher else None,
        )

    # ---- 速率 ----
    def target_rates(self, level: float | None = None) -> tuple[float, float]:
        """返回 (rpm, tpm)。"""
        lv = self.throttle.level if level is None else level
        rpm = self.quota_rpm * self.usage_ratio * lv
        tpm = self.quota_tpm * self.usage_ratio * lv
        if self.rpm_cap is not None:
            rpm = min(rpm, float(self.rpm_cap))
        if self.tpm_cap is not None:
            tpm = min(tpm, float(self.tpm_cap))
        return rpm, tpm

    def _sync(self) -> None:
        """tick 节流器，并在档位变化时更新桶速率。"""
        self.throttle.tick(self.clock.now())
        lv = self.throttle.level
        if lv != self._level:
            self._level = lv
            rpm, tpm = self.target_rates(lv)
            self.bucket.set_rates(tpm, rpm)

    @property
    def rpm(self) -> float:
        return self.bucket.req_rate * 60.0

    @property
    def tpm(self) -> float:
        return self.bucket.tok_rate * 60.0

    # ---- 节流事件转发 ----
    def on_throttle(self, retry_after: float | None, now: float | None = None) -> None:
        """Gemini 返回 429 / RESOURCE_EXHAUSTED：立即 10% + HOLD。"""
        self.throttle.on_throttle(retry_after, self.clock.now() if now is None else now)
        self._sync()

    def is_holding(self) -> bool:
        now = self.clock.now()
        self.throttle.tick(now)
        return self.throttle.is_holding(now)

    async def wait_if_holding(self) -> None:
        """HOLD 期间以 ≤ hold_step_s 步长等待，期间调用 tick。"""
        while True:
            now = self.clock.now()
            self.throttle.tick(now)
            if not self.throttle.is_holding(now):
                return
            remaining = self.throttle.hold_until - now
            await self.clock.sleep(min(max(remaining, 1e-3), self.hold_step_s))

    def clamp_cost(self, est_tokens: int) -> int:
        cost = max(0, int(est_tokens))
        return min(cost, int(math.floor(self.bucket.tok_cap)))

    # ---- 放行 ----
    async def acquire(self, est_tokens: int) -> int:
        """阻塞直至放行 1 次调用；返回实际扣除的 token 数（已截断到 tok_cap）。"""
        async with self._lock:
            while True:
                await self.wait_if_holding()
                self._sync()
                cost = self.clamp_cost(est_tokens)
                if self.bucket.try_acquire(cost):
                    return cost
                wait = self.bucket._wait_s(cost)
                await self.clock.sleep(min(max(wait, 1e-6), self.wait_step_s))

    def try_acquire(self, est_tokens: int) -> bool:
        if self._lock.locked() or self.is_holding():
            return False
        self._sync()
        return self.bucket.try_acquire(self.clamp_cost(est_tokens))


__all__ = ["GeminiRateLimiter", "gemini_throttle_settings", "GEMINI_MIN_TOK_CAP"]
