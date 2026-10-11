"""CostGuard：预算预留 / 结算 + 待刷写账本（Requirements 25、26）。

不变式（只要每次结算的实际 usage ≤ 预留量）::

    spent + reserved <= cap - safety_margin          （历史花费本身已超限时除外）

内存中的 ``spent`` 是权威值，不依赖刷盘；``flush`` 只负责把待写账本持久化。
所有修改状态的方法内部没有 ``await``，在单个 asyncio 事件循环内天然原子。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from loadtest.corpus.store import LedgerEntry
from loadtest.cost.pricing import (
    GROUNDING_PRICE_KEY,
    get_price,
    token_cost,
    unit_cost,
    worst_case_token_cost,
)
from loadtest.settings import CostSettings, ModelPrice

AZURE_PROVIDERS: tuple[str, ...] = ("azure",)
GEMINI_PROVIDERS: tuple[str, ...] = ("gemini", "gemini_grounding")
GROUNDING_PROVIDER = "gemini_grounding"

_EPS = 1e-12


@dataclass(frozen=True)
class Usage:
    """归一化后的 usage。``cached_tokens`` 是 ``prompt_tokens`` 的子集。"""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0


@dataclass(frozen=True)
class Reservation:
    id: int
    model: str
    in_tok: int
    max_out: int
    units: int
    token_amount: float  # 最坏 token 花费
    unit_amount: float  # 按次花费（grounding）

    @property
    def amount(self) -> float:
        return self.token_amount + self.unit_amount


def _field(usage: Any, name: str) -> Any:
    if isinstance(usage, Mapping):
        return usage.get(name)
    return getattr(usage, name, None)


def normalize_usage(usage: Any) -> Usage:
    """接受 :class:`Usage`、OpenAI ``CompletionUsage``（``prompt_tokens_details.cached_tokens``）或 dict。"""
    if isinstance(usage, Usage):
        return usage
    prompt = int(_field(usage, "prompt_tokens") or 0)
    completion = int(_field(usage, "completion_tokens") or 0)
    cached = _field(usage, "cached_tokens")
    if cached is None:
        details = _field(usage, "prompt_tokens_details")
        cached = _field(details, "cached_tokens") if details is not None else None
    return Usage(prompt, completion, int(cached or 0))


class CostGuard:
    def __init__(
        self,
        cap_usd: float,
        margin: float,
        prices: Mapping[str, ModelPrice],
        providers: Iterable[str] = AZURE_PROVIDERS,
        per_run_cap: float | None = None,
        *,
        spent: float = 0.0,
        spent_by_model: Mapping[tuple[str, str], float] | None = None,
        grounding_key: str = GROUNDING_PRICE_KEY,
        account: str = "",
    ) -> None:
        self.account = account  # 写入账本的预算归属账号
        self.providers: tuple[str, ...] = tuple(providers)
        if not self.providers:
            raise ValueError("providers must not be empty")
        self.cap = float(cap_usd)
        self.margin = float(margin)
        self.prices = dict(prices)
        self.per_run_cap = per_run_cap
        self.grounding_key = grounding_key
        self.token_provider = self.providers[0]
        self.unit_provider = GROUNDING_PROVIDER if GROUNDING_PROVIDER in self.providers else self.providers[0]

        self.spent = float(spent)  # 历史 + 本次运行（权威值）
        self.run_spent = 0.0
        self.reserved = 0.0
        self.spent_by_model: dict[tuple[str, str], float] = dict(spent_by_model or {})
        self._outstanding: dict[int, Reservation] = {}
        self._ids = itertools.count(1)
        # (provider, model) -> [input_tokens, output_tokens, units, cost]
        self._pending: dict[tuple[str, str], list[float]] = {}

    # ------------------------------------------------------------ 构造

    @classmethod
    async def from_store(
        cls,
        store: Any,
        cap_usd: float,
        margin: float,
        prices: Mapping[str, ModelPrice],
        providers: Iterable[str] = AZURE_PROVIDERS,
        per_run_cap: float | None = None,
        *,
        account: str | None = None,
        **kw: Any,
    ) -> "CostGuard":
        """以 ledger_totals 中自身 provider 的历史花费作为初值（26.2）。

        account 非 None 时只累计该账号的历史花费（换 GCP 赠金账号后预算从 0 重新计），
        并把新账本行记到该账号下；None 时累计全部历史、账本行记为 ''。
        """
        providers = tuple(providers)
        totals = await (store.ledger_totals(providers, account) if account is not None
                        else store.ledger_totals(providers))
        by_model: dict[tuple[str, str], float] = {}
        for t in totals:
            by_model[(t.provider, t.model)] = by_model.get((t.provider, t.model), 0.0) + t.total_usd
        return cls(
            cap_usd,
            margin,
            prices,
            providers,
            per_run_cap,
            spent=sum(by_model.values()),
            spent_by_model=by_model,
            account=account or "",
            **kw,
        )

    @classmethod
    async def azure_from_settings(cls, store: Any, cs: CostSettings) -> "CostGuard":
        return await cls.from_store(
            store, cs.azure_budget_cap_usd, cs.safety_margin_usd, cs.prices, AZURE_PROVIDERS, cs.per_run_cap_usd
        )

    @classmethod
    async def gemini_from_settings(
        cls, store: Any, cs: CostSettings, project: str | None = None
    ) -> "CostGuard":
        """Gemini 预算按账号独立计算：账号 = cost.gemini_budget_account，未设置时用 gemini.project。"""
        account = cs.gemini_budget_account or project
        return await cls.from_store(
            store, cs.gemini_budget_cap_usd, cs.safety_margin_usd, cs.prices, GEMINI_PROVIDERS, cs.per_run_cap_usd,
            account=account,
        )

    # ------------------------------------------------------------ 计价辅助

    def _unit_cost(self, model: str, units: int) -> float:
        if units <= 0:
            return 0.0
        own = self.prices.get(model)
        price = own if own is not None and own.per_unit is not None else get_price(self.prices, self.grounding_key)
        return unit_cost(price, units)

    def worst_case(self, model: str, in_tok: int, max_out: int, units: int = 0) -> float:
        return worst_case_token_cost(get_price(self.prices, model), in_tok, max_out) + self._unit_cost(model, units)

    # ------------------------------------------------------------ 预留 / 结算

    def can_afford(self, amount: float) -> bool:
        if self.spent + self.reserved + amount > self.cap - self.margin + _EPS:
            return False
        if self.per_run_cap is not None and self.run_spent + self.reserved + amount > self.per_run_cap + _EPS:
            return False
        return True

    async def reserve(self, model: str, in_tok: int, max_out: int, units: int = 0) -> Reservation | None:
        """按最坏情况预留；超出 cap − margin 或 per_run_cap 时返回 None（25.2）。"""
        tok = worst_case_token_cost(get_price(self.prices, model), in_tok, max_out)
        uc = self._unit_cost(model, units)
        if not self.can_afford(tok + uc):
            return None
        r = Reservation(next(self._ids), model, int(in_tok), int(max_out), int(units), tok, uc)
        self._outstanding[r.id] = r
        self.reserved += r.amount
        return r

    def _take(self, r: Reservation) -> bool:
        """移除在途预留；重复结算 / 释放时返回 False（幂等）。"""
        if self._outstanding.pop(r.id, None) is None:
            return False
        if self._outstanding:
            self.reserved = max(0.0, self.reserved - r.amount)
        else:
            self.reserved = 0.0  # 消除浮点累积误差
        return True

    def _charge(self, provider: str, model: str, in_tok: int, out_tok: int, units: int, cost: float) -> None:
        if cost == 0 and in_tok == 0 and out_tok == 0 and units == 0:
            return
        self.spent += cost
        self.run_spent += cost
        key = (provider, model)
        self.spent_by_model[key] = self.spent_by_model.get(key, 0.0) + cost
        row = self._pending.setdefault(key, [0, 0, 0, 0.0])
        row[0] += in_tok
        row[1] += out_tok
        row[2] += units
        row[3] += cost

    async def settle(
        self, r: Reservation, usage: Any | None, model: str | None = None, units: int | None = None
    ) -> float:
        """按 usage 结算（缓存输入按 cached_input_per_m）；usage 为 None（超时等）按预留金额结算（25.3）。

        ``model`` 可覆盖实际计价模型（默认为预留时的模型）。
        ``units`` 可覆盖实际计次数（Gemini 3 grounding 按实际去重搜索查询数计费；默认为预留的次数）。
        返回本次计入的花费。
        """
        if not self._take(r):
            return 0.0
        m = model or r.model
        if units is not None and units != r.units:
            n = max(0, int(units))
            r = replace(r, units=n, unit_amount=self._unit_cost(r.model, n))
        if usage is None:
            tok_cost, in_tok, out_tok = r.token_amount, r.in_tok, r.max_out
        else:
            u = normalize_usage(usage)
            tok_cost = token_cost(get_price(self.prices, m), u.prompt_tokens, u.completion_tokens, u.cached_tokens)
            in_tok, out_tok = u.prompt_tokens, u.completion_tokens
        units_provider_model = (
            (self.token_provider, m) if self.unit_provider == self.token_provider else (self.unit_provider, self.grounding_key)
        )
        if m == self.grounding_key:
            # 直接以 grounding 键预留：全部计入 unit provider
            self._charge(self.unit_provider, m, in_tok, out_tok, r.units, tok_cost + r.unit_amount)
        else:
            self._charge(self.token_provider, m, in_tok, out_tok, 0, tok_cost)
            if r.units:
                self._charge(*units_provider_model, 0, 0, r.units, r.unit_amount)
        return tok_cost + r.unit_amount

    async def settle_zero(self, r: Reservation) -> None:
        """THROTTLED 等未被处理的请求：不计费（花费 0）。"""
        self._take(r)

    async def release(self, r: Reservation) -> None:
        """未派发的预留原样退回（不计费）。"""
        self._take(r)

    # ------------------------------------------------------------ 查询

    @property
    def outstanding(self) -> int:
        return len(self._outstanding)

    def remaining_ratio(self) -> float:
        """(cap − spent − reserved) / cap，下限 0。"""
        if self.cap <= 0:
            return 0.0
        return max(0.0, (self.cap - self.spent - self.reserved) / self.cap)

    def remaining_usd(self) -> float:
        return max(0.0, self.cap - self.margin - self.spent - self.reserved)

    def estimate_run_cost(
        self, tpm: float, rpm: float, minutes: float, model: str, avg_output_tokens: float = 30
    ) -> float:
        """预估：每分钟 ``tpm`` 个输入 token 全价 + ``rpm`` × avg_output_tokens 个输出 token。"""
        price = get_price(self.prices, model)
        return minutes * token_cost(price, tpm, rpm * avg_output_tokens)

    def remaining_runs(self, run_estimate: float) -> int:
        """floor((cap − margin − spent) / 单次预估)（25.8）。"""
        if run_estimate <= 0:
            return 0
        return max(0, math.floor((self.cap - self.margin - self.spent) / run_estimate))

    # ------------------------------------------------------------ 账本

    def pending_entries(self, run_id: str) -> list[LedgerEntry]:
        return [
            LedgerEntry(
                run_id=run_id,
                provider=prov,  # type: ignore[arg-type]
                model=model,
                input_tokens=int(v[0]),
                output_tokens=int(v[1]),
                cost_usd=float(v[3]),
                units=int(v[2]),
                account=self.account,
            )
            for (prov, model), v in sorted(self._pending.items())
        ]

    async def flush(self, store: Any, run_id: str) -> int:
        """把待写账本写入 store；仅在成功后清除（失败时并回，异常向上抛出 LedgerWriteError）。

        返回写入的行数。
        """
        if not self._pending:
            return 0
        snapshot, self._pending = self._pending, {}
        entries = [
            LedgerEntry(run_id, prov, model, int(v[0]), int(v[1]), float(v[3]), int(v[2]),  # type: ignore[arg-type]
                        account=self.account)
            for (prov, model), v in sorted(snapshot.items())
        ]
        try:
            await store.write_ledger(entries)
        except BaseException:
            # 写入期间可能有新的结算进入 self._pending：合并回去
            for key, v in snapshot.items():
                row = self._pending.setdefault(key, [0, 0, 0, 0.0])
                for i in range(4):
                    row[i] += v[i]
            raise
        return len(entries)
