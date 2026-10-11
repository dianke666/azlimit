"""目标输入长度分布（Requirement 13）。

混合分布：
- ``core_ratio``（默认 80%）：截断到 [core_low, core_high] 的正态
  N(μ = (core_low + core_high) / 2, σ = (core_high − core_low) / 3.33)，拒绝采样后四舍五入；
- 其余对半分到两个尾部（各 10%）：均匀 [input_min, core_low) 与均匀 (core_high, input_max]。

返回值恒为 [input_min, input_max] 内的整数。随机源由调用方注入（``random.Random``），
便于复现与测试。
"""

from __future__ import annotations

from random import Random

from loadtest.settings import LoadSettings

# σ = (core_high − core_low) / SIGMA_DIVISOR（默认约 300.3）
SIGMA_DIVISOR = 3.33


def sample_target_length(rng: Random, cfg: LoadSettings) -> int:
    """采样一个目标输入长度（token）。

    前置条件: input_min < core_low < core_high < input_max；0 < core_ratio < 1（由 LoadSettings 校验）。
    后置条件: input_min <= 返回值 <= input_max。
    """
    u = rng.random()
    if u < cfg.core_ratio:
        mu = (cfg.core_low + cfg.core_high) / 2
        sigma = (cfg.core_high - cfg.core_low) / SIGMA_DIVISOR
        while True:  # 截断正态：±1.665σ 内接受率约 90%，期望 < 1.1 次
            x = rng.gauss(mu, sigma)
            if cfg.core_low <= x <= cfg.core_high:
                # 边界为整数，round 后仍落在 [core_low, core_high]
                return round(x)
    tail = (1.0 - cfg.core_ratio) / 2
    if u < cfg.core_ratio + tail:
        return rng.randint(cfg.input_min, cfg.core_low - 1)  # [input_min, core_low)
    return rng.randint(cfg.core_high + 1, cfg.input_max)  # (core_high, input_max]


class LengthDistribution:
    """:func:`sample_target_length` 的面向对象封装，绑定一份 LoadSettings。"""

    def __init__(self, cfg: LoadSettings | None = None) -> None:
        self.cfg = cfg if cfg is not None else LoadSettings()

    def sample(self, rng: Random) -> int:
        return sample_target_length(rng, self.cfg)


__all__ = ["SIGMA_DIVISOR", "LengthDistribution", "sample_target_length"]
