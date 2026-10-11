"""双令牌桶限流器（Dual_Token_Bucket）。

单一实例同时约束 token 速率与请求速率：每次放行扣除 ``cost`` 个 token 与 1 个请求额度。

不变式:
    0 <= tok <= tok_cap；0 <= req <= req_cap
性质:
    任意区间 [t0, t1] 内放行的 token 总量 <= tok_cap + tok_rate × (t1 − t0)，
    放行请求数 <= req_cap + req_rate × (t1 − t0)。
容量:
    tok_cap = max(TPM × burst_s / 60, min_tok_cap)（默认 burst_s = 5、min_tok_cap = 6040）
    req_cap = max(RPM × burst_s / 60, 1)
初始存量:
    tok = initial_fill_s × tok_rate；req = max(1, initial_fill_s × req_rate)（均截断到容量）
降速:
    ``set_rates`` 先按旧速率补充到当前时刻，再更新速率与容量并立即截断存量。
等待:
    ``acquire`` 持全局 asyncio.Lock（FIFO），存量不足时以 ≤ ``max_wait_step_s``（默认 0.25s）
    的步长经注入的 Clock 睡眠并重查；速率为 0 时同样按步长等待，不会除零。

Gemini 侧（GeminiRateLimiter）以不同的 burst / 初始存量 / 最小 token 容量复用本类。
"""

from __future__ import annotations

import asyncio
import math

from loadtest.clock import Clock

# 单个 Azure 请求的最大成本：输入上限 6000 + max_tokens 40
MAX_SINGLE_REQUEST = 6040
DEFAULT_BURST_S = 5.0
DEFAULT_INITIAL_FILL_S = 2.0
MAX_WAIT_STEP_S = 0.25


class DualTokenBucket:
    def __init__(
        self,
        tokens_per_min: float,
        requests_per_min: float,
        burst_s: float = DEFAULT_BURST_S,
        initial_fill_s: float = DEFAULT_INITIAL_FILL_S,
        clock: Clock | None = None,
        *,
        min_tok_cap: float = MAX_SINGLE_REQUEST,
        max_wait_step_s: float = MAX_WAIT_STEP_S,
    ) -> None:
        if clock is None:
            raise ValueError("clock is required")
        if burst_s <= 0:
            raise ValueError("burst_s must be > 0")
        if initial_fill_s < 0:
            raise ValueError("initial_fill_s must be >= 0")
        if min_tok_cap < 0:
            raise ValueError("min_tok_cap must be >= 0")
        if not 0 < max_wait_step_s <= MAX_WAIT_STEP_S:
            raise ValueError(f"max_wait_step_s must be in (0, {MAX_WAIT_STEP_S}]")
        self.clock = clock
        self.burst_s = float(burst_s)
        self.initial_fill_s = float(initial_fill_s)
        self.min_tok_cap = float(min_tok_cap)
        self.max_wait_step_s = float(max_wait_step_s)
        self.lock = asyncio.Lock()

        self.tok_rate = 0.0
        self.req_rate = 0.0
        self.tok_cap = 0.0
        self.req_cap = 0.0
        self._apply_rates(tokens_per_min, requests_per_min)

        self.tok = min(self.tok_cap, self.initial_fill_s * self.tok_rate)
        self.req = min(self.req_cap, max(1.0, self.initial_fill_s * self.req_rate))
        self.last = self.clock.now()

        # 统计：累计放行量（便于测试与指标）
        self.granted_tokens = 0
        self.granted_requests = 0

    # ---- 内部 ----
    def _apply_rates(self, tpm: float, rpm: float) -> None:
        if tpm < 0 or rpm < 0 or math.isnan(tpm) or math.isnan(rpm):
            raise ValueError("rates must be non-negative numbers")
        self.tok_rate = tpm / 60.0
        self.req_rate = rpm / 60.0
        self.tok_cap = max(tpm * self.burst_s / 60.0, self.min_tok_cap)
        self.req_cap = max(rpm * self.burst_s / 60.0, 1.0)

    def _refill(self, now: float) -> None:
        dt = max(0.0, now - self.last)
        self.tok = min(self.tok_cap, self.tok + dt * self.tok_rate)
        self.req = min(self.req_cap, self.req + dt * self.req_rate)
        self.last = max(self.last, now)

    def _check_cost(self, cost: int) -> None:
        if cost < 0:
            raise ValueError("cost must be >= 0")
        if cost > self.tok_cap:
            raise ValueError(f"cost {cost} exceeds token capacity {self.tok_cap:g}")

    def _take(self, cost: int) -> bool:
        if self.tok >= cost and self.req >= 1.0:
            self.tok -= cost
            self.req -= 1.0
            self.granted_tokens += cost
            self.granted_requests += 1
            return True
        return False

    def _wait_s(self, cost: int) -> float:
        """距离可放行的估计时长（速率为 0 时为 inf），调用方再截断到步长。"""
        tok_need = cost - self.tok
        req_need = 1.0 - self.req
        tok_wait = 0.0 if tok_need <= 0 else (tok_need / self.tok_rate if self.tok_rate > 0 else math.inf)
        req_wait = 0.0 if req_need <= 0 else (req_need / self.req_rate if self.req_rate > 0 else math.inf)
        return max(tok_wait, req_wait)

    # ---- 公共接口 ----
    def set_rates(self, tokens_per_min: float, requests_per_min: float) -> None:
        """更新速率；按旧速率补充到当前时刻后切换，降速时立即截断存量到新容量。"""
        self._refill(self.clock.now())
        self._apply_rates(tokens_per_min, requests_per_min)
        self.tok = min(self.tok, self.tok_cap)
        self.req = min(self.req, self.req_cap)

    def try_acquire(self, tokens: int) -> bool:
        """同步尝试放行；有异步等待者持锁时返回 False 以保持 FIFO。"""
        self._check_cost(tokens)
        if self.lock.locked():
            return False
        self._refill(self.clock.now())
        return self._take(tokens)

    async def acquire(self, tokens: int) -> None:
        """阻塞直至同时扣除 ``tokens`` 个 token 与 1 个请求额度（FIFO）。"""
        async with self.lock:
            while True:
                self._refill(self.clock.now())
                self._check_cost(tokens)  # 容量可能随 set_rates 变化
                if self._take(tokens):
                    return
                wait = self._wait_s(tokens)
                # 下限 1µs 防止浮点误差导致零时长空转
                await self.clock.sleep(min(max(wait, 1e-6), self.max_wait_step_s))

    def snapshot(self) -> dict[str, float]:
        self._refill(self.clock.now())
        return {
            "tok": self.tok,
            "req": self.req,
            "tok_cap": self.tok_cap,
            "req_cap": self.req_cap,
            "tok_rate": self.tok_rate,
            "req_rate": self.req_rate,
        }


__all__ = ["DualTokenBucket", "MAX_SINGLE_REQUEST", "DEFAULT_BURST_S", "DEFAULT_INITIAL_FILL_S"]
