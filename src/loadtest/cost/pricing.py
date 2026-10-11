"""计价函数（USD）。单价单位：USD / 1M token；grounding 按次 ``per_unit``（USD / 次）。"""

from __future__ import annotations

from collections.abc import Mapping

from loadtest.settings import ModelPrice

#: 预留时输入 token 的计数容差（本地 tiktoken 与服务端计数差异）
INPUT_TOLERANCE = 1.02

#: grounding 价格表键名（provider = gemini_grounding）
GROUNDING_PRICE_KEY = "google-search-grounding"


class UnknownModelPrice(KeyError):
    """价格表中没有该模型。"""


def get_price(prices: Mapping[str, ModelPrice], model: str) -> ModelPrice:
    try:
        return prices[model]
    except KeyError:
        raise UnknownModelPrice(f"no price configured for model {model!r}") from None


def token_cost(
    price: ModelPrice,
    input_tokens: float,
    output_tokens: float,
    cached_tokens: float = 0,
) -> float:
    """token 部分花费。``cached_tokens`` 是 ``input_tokens`` 的子集（与 OpenAI usage 语义一致），
    按 ``cached_input_per_m`` 计价，未配置时按 ``input_per_m``。"""
    input_tokens = max(0.0, float(input_tokens))
    cached = min(max(0.0, float(cached_tokens)), input_tokens)
    cached_rate = price.cached_input_per_m if price.cached_input_per_m is not None else price.input_per_m
    return (
        (input_tokens - cached) * price.input_per_m
        + cached * cached_rate
        + max(0.0, float(output_tokens)) * price.output_per_m
    ) / 1_000_000


def unit_cost(price: ModelPrice, units: int) -> float:
    """按次计费部分（grounding）。未配置 ``per_unit`` 时为 0。"""
    if units <= 0 or price.per_unit is None:
        return 0.0
    return units * price.per_unit


def cost_of(
    price: ModelPrice,
    input_tokens: float,
    output_tokens: float,
    cached_tokens: float = 0,
    units: int = 0,
) -> float:
    """单个价格条目的总花费 = token 花费 + 按次花费。"""
    return token_cost(price, input_tokens, output_tokens, cached_tokens) + unit_cost(price, units)


def worst_case_token_cost(price: ModelPrice, input_tokens: int, max_tokens: int) -> float:
    """预留用最坏情况：输入 × 1.02 全价（不按缓存折扣）+ max_tokens 输出。"""
    return token_cost(price, input_tokens * INPUT_TOLERANCE, max_tokens, 0)
