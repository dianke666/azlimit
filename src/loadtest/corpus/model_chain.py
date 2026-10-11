"""ModelChain：Gemini 主模型 + 备用模型链（组件 4，Requirement 12.1–12.5）。

规则:
    - ``current(now)`` 始终 ∈ models。
    - 当前（活动）模型在重试耗尽后连续失败达到 ``failover_after`` → 切到下一个模型并告警；
      已在末位或下一个是昂贵模型且预算剩余比例 < ``low_budget_ratio`` → 保持并告警。
      动作后该模型的连续失败计数清零（每满 failover_after 次才再次动作）。
    - 任一模型调用成功 → 该模型连续失败计数清零。
    - 切到备用模型（或主模型探测失败）后经过 ``failback_after_s``，下一次尝试发往主模型（探测）；
      探测期间主模型连续 ``restore_successes`` 次成功 → 恢复主模型为活动模型；
      探测失败 → 回到备用模型并重新计时。
    - 预算剩余比例 < ``low_budget_ratio`` 时 ``current()`` 不返回昂贵模型（若活动模型是昂贵模型，
      回退到链中最近的非昂贵模型）。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

log = logging.getLogger(__name__)

DEFAULT_MODEL_CHAIN = ("gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash")
DEFAULT_EXPENSIVE = ("gemini-3.8-flash",)


class ModelChain:
    def __init__(
        self,
        models: Sequence[str] = DEFAULT_MODEL_CHAIN,
        failover_after: int = 5,
        failback_after_s: float = 600.0,
        budget_ratio_fn: Callable[[], float] | None = None,
        expensive: Sequence[str] = DEFAULT_EXPENSIVE,
        restore_successes: int = 3,
        low_budget_ratio: float = 0.2,
    ) -> None:
        models = list(models)
        if not models:
            raise ValueError("models must not be empty")
        if len(set(models)) != len(models):
            raise ValueError("models must be unique")
        if failover_after < 1:
            raise ValueError("failover_after must be >= 1")
        if failback_after_s < 0:
            raise ValueError("failback_after_s must be >= 0")
        if restore_successes < 1:
            raise ValueError("restore_successes must be >= 1")
        self.models: list[str] = models
        self.failover_after = int(failover_after)
        self.failback_after_s = float(failback_after_s)
        self.budget_ratio_fn = budget_ratio_fn
        self.expensive = frozenset(expensive)
        self.restore_successes = int(restore_successes)
        self.low_budget_ratio = float(low_budget_ratio)

        self._idx = 0  # 活动模型下标（不含探测）
        self.failures: dict[str, int] = {m: 0 for m in models}  # 连续失败计数
        self.fallback_since: float | None = None  # 进入备用 / 探测失败的时刻
        self.probing = False
        self.probe_successes = 0
        self.warnings: list[str] = []

    # ---- 查询 ----
    @property
    def primary(self) -> str:
        return self.models[0]

    @property
    def active(self) -> str:
        """活动模型（不考虑探测）。"""
        return self.models[self._idx]

    def budget_low(self) -> bool:
        if self.budget_ratio_fn is None:
            return False
        return self.budget_ratio_fn() < self.low_budget_ratio

    def _blocked(self, model: str, low: bool) -> bool:
        return low and model in self.expensive

    def current(self, now: float) -> str:
        low = self.budget_low()
        if self._blocked(self.active, low):
            self._retreat_from_expensive()
        if self._idx != 0 and not self._blocked(self.primary, low):
            if not self.probing and self.fallback_since is not None and (
                now - self.fallback_since >= self.failback_after_s
            ):
                self.probing = True
                self.probe_successes = 0
                log.info("model chain: probing primary %s", self.primary)
            if self.probing:
                return self.primary
        return self.active

    def _retreat_from_expensive(self) -> None:
        """预算不足时把活动模型移到最近的非昂贵模型（先向前找，再向后找）。"""
        order = list(range(self._idx - 1, -1, -1)) + list(range(self._idx + 1, len(self.models)))
        for j in order:
            if self.models[j] not in self.expensive:
                self._warn(
                    f"budget low: switching from expensive {self.active} to {self.models[j]}"
                )
                self._idx = j
                if j == 0:
                    self.probing = False
                    self.fallback_since = None
                return
        # 全部为昂贵模型：无法回避，保持

    # ---- 报告 ----
    def report_success(self, model: str, now: float) -> None:
        if model not in self.failures:
            return
        self.failures[model] = 0
        if self.probing and model == self.primary and self._idx != 0:
            self.probe_successes += 1
            if self.probe_successes >= self.restore_successes:
                self._warn(f"primary {self.primary} restored after {self.probe_successes} successes")
                self._idx = 0
                self.probing = False
                self.probe_successes = 0
                self.fallback_since = None

    def report_failure(self, model: str, now: float) -> None:
        """重试耗尽后计一次失败。"""
        if model not in self.failures:
            return
        self.failures[model] += 1
        if self.probing and model == self.primary and self._idx != 0:
            self.probing = False
            self.probe_successes = 0
            self.fallback_since = now
            self.failures[model] = 0
            self._warn(f"primary {self.primary} probe failed; back to {self.active}")
            return
        if model != self.active or self.failures[model] < self.failover_after:
            return
        self.failures[model] = 0
        j = self._idx + 1
        if j >= len(self.models):
            self._warn(f"{model} failed {self.failover_after} times; already last model, staying")
            return
        nxt = self.models[j]
        if self._blocked(nxt, self.budget_low()):
            self._warn(f"{model} failed {self.failover_after} times; next {nxt} is expensive under low budget, staying")
            return
        self._warn(f"{model} failed {self.failover_after} times; switching to {nxt}")
        self._idx = j
        self.probing = False
        self.probe_successes = 0
        self.fallback_since = now

    def _warn(self, msg: str) -> None:
        self.warnings.append(msg)
        log.warning("model chain: %s", msg)


__all__ = ["ModelChain", "DEFAULT_MODEL_CHAIN", "DEFAULT_EXPENSIVE"]
