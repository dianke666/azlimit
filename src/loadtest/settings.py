"""配置加载与校验（Requirement 1）。

- 非密钥参数来自 ``config.yaml``；密钥 / 连接参数只来自环境变量或 ``.env``：
  ``AZURE_OPENAI_API_KEY``、``AZURE_OPENAI_ENDPOINT``、``GOOGLE_APPLICATION_CREDENTIALS``。
- 校验失败时 :func:`load_settings` 抛出 :class:`SettingsError`（只含参数名与原因，
  从不包含输入值）；:func:`load_settings_or_exit` 打印到 stderr 并以非零退出码终止。
  加载过程本身不发起任何 Azure / Gemini 调用。
- :func:`redact` / :func:`config_snapshot` 用掩码替换所有密钥值，供日志、
  ``runs.config_json``、JSONL、CSV 与报告使用。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

Lang = Literal["zh", "en"]

# 环境变量名 → 配置中的位置
ENV_AZURE_API_KEY = "AZURE_OPENAI_API_KEY"
ENV_AZURE_ENDPOINT = "AZURE_OPENAI_ENDPOINT"
ENV_GOOGLE_CREDENTIALS = "GOOGLE_APPLICATION_CREDENTIALS"

_ENV_FIELDS: dict[str, tuple[str, str]] = {
    ENV_AZURE_API_KEY: ("azure", "api_key"),
    ENV_AZURE_ENDPOINT: ("azure", "endpoint"),
    ENV_GOOGLE_CREDENTIALS: ("gemini", "credentials_file"),
}
_LOC_TO_ENV: dict[tuple[str, str], str] = {loc: env for env, loc in _ENV_FIELDS.items()}

MASK = "***REDACTED***"
EXIT_CONFIG_ERROR = 2

_FLOAT_EPS = 1e-9

AZURE_THROTTLE_DEFAULT_LEVELS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)
GEMINI_THROTTLE_DEFAULT_LEVELS: tuple[float, ...] = (0.10, 0.50, 1.00)


class _Model(BaseModel):
    """所有子配置：拒绝未知字段（防止 YAML 拼写错误被静默忽略）。"""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 节流
# ---------------------------------------------------------------------------


class ThrottleSettings(_Model):
    mode: Literal["probe", "level"] = "probe"  # Azure = probe；Gemini = level
    levels: list[float] = Field(default_factory=lambda: list(AZURE_THROTTLE_DEFAULT_LEVELS))
    recover_full_s: float = 60.0  # RECOVER 从首档到 100% 的总时长
    min_cooldown_s: float = 10.0  # hold 下限
    max_retry_after_s: float = 120.0  # hold 上限（Retry-After 截断）
    probe_inflight_wait_s: float = 30.0  # PROBE 前等待在途请求清空的上限
    probe_error_retry_s: float = 5.0  # 探测 SERVER_ERROR / TIMEOUT / CLIENT_ERROR 后的重试间隔
    soft_down_ratio: float = Field(default=0.10, ge=0.0, le=1.0)  # remaining < 10% 触发主动减速
    soft_down_floor: float = 0.50  # 主动减速下限（必须 ∈ levels）
    soft_down_cooldown_s: float = Field(default=5.0, ge=0.0)

    @property
    def recover_step_s(self) -> float:
        """每档停留时长 = recover_full_s / (len(levels) - 1)。"""
        return self.recover_full_s / (len(self.levels) - 1)

    @model_validator(mode="after")
    def _check(self) -> ThrottleSettings:
        lv = self.levels
        if len(lv) < 2:
            raise ValueError("levels must contain at least 2 items (first 0.10, last 1.00)")
        if any(b <= a for a, b in zip(lv, lv[1:])):
            raise ValueError("levels must be strictly increasing")
        if not math.isclose(lv[0], 0.10, abs_tol=_FLOAT_EPS):
            raise ValueError("levels[0] must equal 0.10")
        if not math.isclose(lv[-1], 1.00, abs_tol=_FLOAT_EPS):
            raise ValueError("levels[-1] must equal 1.00")
        if not self.recover_full_s > 0:
            raise ValueError("recover_full_s must be > 0")
        if not any(math.isclose(self.soft_down_floor, x, abs_tol=_FLOAT_EPS) for x in lv):
            raise ValueError("soft_down_floor must be one of levels")
        if self.min_cooldown_s > self.max_retry_after_s:
            raise ValueError("min_cooldown_s must be <= max_retry_after_s")
        if self.min_cooldown_s < 0:
            raise ValueError("min_cooldown_s must be >= 0")
        if not self.probe_inflight_wait_s > 0:
            raise ValueError("probe_inflight_wait_s must be > 0")
        if not self.probe_error_retry_s > 0:
            raise ValueError("probe_error_retry_s must be > 0")
        return self


def _gemini_throttle_defaults() -> dict[str, Any]:
    return {"mode": "level", "levels": list(GEMINI_THROTTLE_DEFAULT_LEVELS), "recover_full_s": 20.0}


# ---------------------------------------------------------------------------
# Azure / Gemini
# ---------------------------------------------------------------------------


class AzureSettings(_Model):
    endpoint: HttpUrl  # 来自 .env: AZURE_OPENAI_ENDPOINT
    api_key: SecretStr  # 来自 .env: AZURE_OPENAI_API_KEY
    api_version: str = "2024-10-21"
    deployment: str = "gpt-4.1-nano"
    model_family: Literal["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini"] = "gpt-4.1-nano"
    quota_tpm: int = Field(default=1_000_000, gt=0)
    quota_rpm: int = Field(default=1_000, gt=0)

    @field_validator("api_key")
    @classmethod
    def _non_empty_key(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value().strip():
            raise ValueError("must not be empty")
        return v


class GeminiSettings(_Model):
    credentials_file: Path  # 服务账号 JSON，来自 .env: GOOGLE_APPLICATION_CREDENTIALS
    project: str
    location: str = "global"
    model_chain: list[str] = Field(
        default_factory=lambda: ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"],
        min_length=1,
    )
    # 预算剩余 < 20% 时跳过的高价模型（Requirement 12.5）
    expensive_models: list[str] = Field(default_factory=lambda: ["gemini-3.8-flash"])
    # Gemini 3 官方建议保持默认 temperature 1.0（低温可能导致重复/循环输出）
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    # 每个模型的 thinking_level（Gemini 3+）；未列出的 gemini-3* 模型用 "low"，gemini-2.5* 用 thinking_budget=0
    thinking_levels: dict[str, Literal["minimal", "low", "medium", "high"]] = Field(
        default_factory=lambda: {
            "gemini-3.1-flash-lite": "minimal",
            "gemini-3.5-flash-lite": "minimal",
            "gemini-3.8-flash": "low",  # 3.8 Flash 不支持 minimal
        }
    )
    quota_rpm: int = Field(default=600, gt=0)
    quota_tpm: int = Field(default=800_000, gt=0)
    usage_ratio: float = Field(default=0.5, gt=0.0, le=1.0)  # 最多使用配额的 50%
    warmup_concurrency: int = Field(default=32, gt=0)
    replenish_rpm: int = Field(default=30, gt=0)
    replenish_concurrency: int = Field(default=4, gt=0)
    replenish_queue_size: int = Field(default=1024, gt=0)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    backoff_base_s: float = Field(default=2.0, ge=0)
    backoff_cap_s: float = Field(default=30.0, ge=0)
    failover_after_failures: int = Field(default=5, gt=0)
    failback_after_s: int = Field(default=600, ge=0)
    grounding_categories: list[str] = Field(
        default_factory=lambda: ["daily_news", "daily_economy", "sports_events"]
    )
    # 每 UTC 日 grounding 计费搜索查询数上限（Gemini 3 按去重查询计费，一次生成通常 1–3 次查询）
    grounding_daily_limit: int = Field(default=600, ge=0)
    # 每次 grounding 尝试预留的查询数；成功后按实际查询数结算，超时/5xx 按预留数计
    grounding_reserve_queries: int = Field(default=3, ge=1)
    # Gemini_Throttle：level 模式，10% → 50% → 100% 每档 10s
    throttle: ThrottleSettings = Field(
        default_factory=lambda: ThrottleSettings(**_gemini_throttle_defaults())
    )

    @field_validator("project")
    @classmethod
    def _non_empty_project(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @field_validator("credentials_file")
    @classmethod
    def _credentials_readable(cls, v: Path) -> Path:
        # 只检查存在性与可读性：打开后立即关闭，不读取、不输出任何内容
        if not v.is_file():
            raise ValueError("credentials file does not exist or is not a regular file")
        try:
            with open(v, "rb"):
                pass
        except OSError:
            raise ValueError("credentials file is not readable") from None
        return v

    @field_validator("throttle", mode="before")
    @classmethod
    def _merge_gemini_throttle_defaults(cls, v: Any) -> Any:
        # YAML 只给出部分字段时，以 Gemini 默认值（level 模式）为基底，而非 Azure 默认值
        if isinstance(v, Mapping):
            return {**_gemini_throttle_defaults(), **v}
        return v

    @field_validator("throttle")
    @classmethod
    def _gemini_throttle_mode(cls, v: ThrottleSettings) -> ThrottleSettings:
        if v.mode != "level":
            raise ValueError("mode must be 'level' (Gemini_Throttle)")
        return v


# ---------------------------------------------------------------------------
# 负载 / 语料 / 成本
# ---------------------------------------------------------------------------


class LoadSettings(_Model):
    target_tpm: int = Field(default=800_000, gt=0)
    tpm_walk_low: int = Field(default=700_000, gt=0)
    tpm_walk_high: int = Field(default=850_000, gt=0)
    rpm_min: int = Field(default=200, gt=0)
    rpm_max: int = Field(default=500, gt=0)
    input_min: int = Field(default=1500, gt=0)
    input_max: int = Field(default=6000, gt=0)
    core_low: int = Field(default=2500, gt=0)
    core_high: int = Field(default=3500, gt=0)
    core_ratio: float = Field(default=0.80, gt=0.0, lt=1.0)
    max_tokens: int = Field(default=40, gt=0)
    max_concurrency: int = Field(default=64, gt=0)
    duration_s: int = Field(default=1200, gt=0)  # 20 分钟
    lang_ratio: dict[Lang, float] = Field(default_factory=lambda: {"zh": 0.60, "en": 0.40})
    startup_ramp: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.5), (10.0, 1.0)]
    )  # (秒, 档位)，阶梯式
    bucket_burst_s: float = Field(default=5.0, gt=0)
    bucket_initial_fill_s: float = Field(default=2.0, ge=0)
    ewma_alpha: float = Field(default=0.05, gt=0.0, le=1.0)
    tick_interval_s: float = Field(default=1.0, gt=0, le=1.0)
    drain_timeout_s: float = Field(default=30.0, ge=0)
    walk_low_ratio: float = Field(default=0.875, gt=0)
    walk_high_ratio: float = Field(default=1.0625, gt=0)
    probe_max_retries: int = Field(default=3, ge=0)
    probe_retry_interval_s: float = Field(default=5.0, ge=0)
    pid_file: Path = Path("runs/loadtest.pid")

    @field_validator("lang_ratio")
    @classmethod
    def _check_lang_ratio(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v.keys()) != {"zh", "en"}:
            raise ValueError("keys must be exactly {zh, en}")
        for k, x in v.items():
            if not (0.0 <= x <= 1.0):
                raise ValueError(f"value for '{k}' must be in [0, 1]")
        total = sum(v.values())
        if abs(total - 1.0) > _FLOAT_EPS:
            raise ValueError("values must sum to 1 (|sum - 1| <= 1e-9)")
        return v

    @field_validator("startup_ramp")
    @classmethod
    def _check_startup_ramp(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not v:
            raise ValueError("must contain at least one step")
        times = [t for t, _ in v]
        if times[0] != 0.0:
            raise ValueError("first step time must be 0")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("step times must be strictly increasing")
        for _, lvl in v:
            if not (0.0 < lvl <= 1.0):
                raise ValueError("step levels must be in (0, 1]")
        if v[-1][1] != 1.0:
            raise ValueError("last step level must equal 1.0")
        return v

    @model_validator(mode="after")
    def _check_ranges(self) -> LoadSettings:
        if not (self.input_min < self.core_low < self.core_high < self.input_max):
            raise ValueError("require input_min < core_low < core_high < input_max")
        if self.tpm_walk_low > self.tpm_walk_high:
            raise ValueError("require tpm_walk_low <= tpm_walk_high")
        if self.rpm_min > self.rpm_max:
            raise ValueError("require rpm_min <= rpm_max")
        if self.walk_low_ratio > self.walk_high_ratio:
            raise ValueError("require walk_low_ratio <= walk_high_ratio")
        return self


class CorpusSettings(_Model):
    min_runnable: int = Field(default=2_000, ge=0)
    healthy_size: int = Field(default=8_000, ge=0)
    target_size: int = Field(default=15_000, ge=0)
    per_run_max_uses: int = Field(default=4, gt=0)
    use_weight_alpha: float = Field(default=1.0, ge=0)  # 采样权重 = 1 / (1 + use_count)^alpha
    lang_share_tolerance: float = Field(default=0.05, ge=0, le=1)
    supply_margin: float = Field(default=0.20, ge=0)
    passage_min_tokens: int = Field(default=500, gt=0)
    passage_max_tokens: int = Field(default=1500, gt=0)
    length_buckets: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {"S": (500, 799), "M": (800, 1199), "L": (1200, 1500)}
    )
    min_bucket_share: float = Field(default=0.15, ge=0, le=1)
    grounded_ttl_h: int = Field(default=72, gt=0)

    @model_validator(mode="after")
    def _check(self) -> CorpusSettings:
        if self.passage_min_tokens > self.passage_max_tokens:
            raise ValueError("require passage_min_tokens <= passage_max_tokens")
        for name, (lo, hi) in self.length_buckets.items():
            if lo > hi:
                raise ValueError(f"length_buckets['{name}'] lower bound must be <= upper bound")
        return self


class ModelPrice(_Model):
    """单价（USD / 1M token）；grounding 按次计费使用 per_unit（USD / 次）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_m: float = Field(ge=0)
    output_per_m: float = Field(ge=0)
    cached_input_per_m: float | None = Field(default=None, ge=0)
    per_unit: float | None = Field(default=None, ge=0)


def _default_prices() -> dict[str, ModelPrice]:
    return {
        "gpt-4.1-nano": ModelPrice(input_per_m=0.10, output_per_m=0.40, cached_input_per_m=0.025),
        "gpt-4o-mini": ModelPrice(input_per_m=0.15, output_per_m=0.60, cached_input_per_m=0.075),
        "gpt-4.1-mini": ModelPrice(input_per_m=0.40, output_per_m=1.60, cached_input_per_m=0.10),
        # Vertex AI global 端点标准价（2026-10 核对）；思考 token 按输出价计费
        "gemini-3.1-flash-lite": ModelPrice(input_per_m=0.25, output_per_m=1.50, cached_input_per_m=0.025),
        "gemini-3.5-flash-lite": ModelPrice(input_per_m=0.30, output_per_m=2.50),
        # 3.8-flash 2026-12-31 前促销价 0.75/3.75，此处按促销结束后的 1.50/7.50 保守计
        "gemini-3.8-flash": ModelPrice(input_per_m=1.50, output_per_m=7.50),
        # Gemini 3 grounding：每月 5,000 次免费，之后 $14 / 1000 次（按次计费，此处不扣免费额度，偏保守）
        "google-search-grounding": ModelPrice(input_per_m=0.0, output_per_m=0.0, per_unit=0.014),
    }


class CostSettings(_Model):
    prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)
    azure_budget_cap_usd: float = Field(default=180.0, gt=0)
    gemini_budget_cap_usd: float = Field(default=250.0, gt=0)  # 每个 GCP 赠金账号的上限
    # Gemini 预算归属账号；None 时使用 gemini.project。换赠金账号（新项目）后预算自动从 0 重新计；
    # 若同一项目换绑了新的结算账号，把这里改成新的标识即可重新计。
    gemini_budget_account: str | None = None
    per_run_cap_usd: float | None = Field(default=None, gt=0)
    safety_margin_usd: float = Field(default=0.5, ge=0)


# ---------------------------------------------------------------------------
# 顶层
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", env_file=None)

    azure: AzureSettings
    gemini: GeminiSettings
    load: LoadSettings = Field(default_factory=LoadSettings)
    throttle: ThrottleSettings = Field(default_factory=ThrottleSettings)  # Azure_Throttle，probe
    corpus: CorpusSettings = Field(default_factory=CorpusSettings)
    cost: CostSettings = Field(default_factory=CostSettings)
    db_path: Path = Path("data/corpus.db")
    log_dir: Path = Path("runs")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 只接受显式传入的数据（由 load_settings 合并 YAML 与三个密钥环境变量），
        # 避免任意环境变量（如 LOAD、DB_PATH）意外覆盖配置。
        return (init_settings,)

    @field_validator("throttle")
    @classmethod
    def _azure_throttle_mode(cls, v: ThrottleSettings) -> ThrottleSettings:
        if v.mode != "probe":
            raise ValueError("mode must be 'probe' (Azure_Throttle)")
        return v

    @model_validator(mode="after")
    def _check_cross(self) -> Settings:
        missing = [
            m for m in [self.azure.model_family, *self.gemini.model_chain] if m not in self.cost.prices
        ]
        if missing:
            raise ValueError(f"cost.prices missing entries for models: {', '.join(missing)}")
        return self

    # -- 密钥 ---------------------------------------------------------------

    def secret_values(self) -> list[str]:
        """需要在所有输出中掩码的原始密钥值。"""
        key = self.azure.api_key.get_secret_value()
        return [key] if key else []


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


class SettingsError(Exception):
    """配置不合法。``errors`` 为 (参数名, 原因)，从不包含输入值。"""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        super().__init__(self.format())

    @property
    def params(self) -> list[str]:
        return [p for p, _ in self.errors]

    def format(self) -> str:
        lines = ["Invalid configuration:"]
        lines += [f"  - {p}: {msg}" for p, msg in self.errors]
        return "\n".join(lines)


def _param_name(loc: tuple[Any, ...]) -> str:
    parts = [str(p) for p in loc]
    if len(parts) >= 2 and (parts[0], parts[1]) in _LOC_TO_ENV:
        return _LOC_TO_ENV[(parts[0], parts[1])]
    return ".".join(parts) if parts else "settings"


def _strip_value_error_prefix(msg: str) -> str:
    return msg.removeprefix("Value error, ")


def _validation_errors(exc: ValidationError) -> list[tuple[str, str]]:
    # 只使用 loc 与 msg；不使用 input / ctx，以免泄露密钥或文件内容
    out: list[tuple[str, str]] = []
    for err in exc.errors(include_url=False, include_input=False, include_context=False):
        loc = tuple(err.get("loc", ()))
        msg = "missing" if err.get("type") == "missing" else _strip_value_error_prefix(err.get("msg", "invalid"))
        out.append((_param_name(loc), msg))
    return out


def _read_env(env_file: str | Path | None, environ: Mapping[str, str] | None) -> dict[str, str]:
    """读取三个密钥变量；进程环境优先于 .env 文件。空值视为缺失。"""
    values: dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        for k, v in dotenv_values(env_file).items():
            if k in _ENV_FIELDS and v:
                values[k] = v
    env = os.environ if environ is None else environ
    for k in _ENV_FIELDS:
        v = env.get(k)
        if v:
            values[k] = v
    return values


def load_settings(
    config_path: str | Path = "config.yaml",
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """从 YAML + 环境变量 / ``.env`` 加载并校验配置；失败抛出 :class:`SettingsError`。

    不发起任何网络调用。成功后把密钥登记到全局掩码表，供 :func:`redact` 使用。
    """
    path = Path(config_path)
    if not path.is_file():
        raise SettingsError([("config", f"config file not found: {path}")])
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        raise SettingsError([("config", f"cannot parse YAML ({type(e).__name__})")]) from None
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise SettingsError([("config", "top level of YAML must be a mapping")])

    errors: list[tuple[str, str]] = []
    # 密钥 / 连接参数不得写在 YAML 中
    for env_name, (section, field) in _ENV_FIELDS.items():
        sec = raw.get(section)
        if isinstance(sec, dict) and field in sec:
            errors.append((env_name, f"must be set via environment / .env, not in YAML ({section}.{field})"))
    if errors:
        raise SettingsError(errors)

    data: dict[str, Any] = dict(raw)
    for env_name, value in _read_env(env_file, environ).items():
        section, field = _ENV_FIELDS[env_name]
        sec = data.get(section)
        if sec is None:
            sec = {}
        if not isinstance(sec, dict):
            continue  # 类型错误由校验报告
        data[section] = {**sec, field: value}

    try:
        settings = Settings(**data)
    except ValidationError as e:
        raise SettingsError(_validation_errors(e)) from None

    register_secrets(settings.secret_values())
    return settings


def load_settings_or_exit(
    config_path: str | Path = "config.yaml",
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """CLI 使用：校验失败时向 stderr 输出参数名并以非零退出码终止。"""
    try:
        return load_settings(config_path, env_file, environ)
    except SettingsError as e:
        print(e.format(), file=sys.stderr)
        raise SystemExit(EXIT_CONFIG_ERROR) from None


# ---------------------------------------------------------------------------
# 掩码
# ---------------------------------------------------------------------------

_secrets_lock = threading.Lock()
_registered_secrets: set[str] = set()


def register_secrets(values: Iterable[str]) -> None:
    """登记需要掩码的原始密钥值（进程级）。"""
    with _secrets_lock:
        _registered_secrets.update(v for v in values if v)


def clear_registered_secrets() -> None:
    """测试用：清空掩码表。"""
    with _secrets_lock:
        _registered_secrets.clear()


def _all_secrets(extra: Iterable[str] | None) -> list[str]:
    with _secrets_lock:
        s = set(_registered_secrets)
    if extra:
        s.update(v for v in extra if v)
    # 长的先替换，避免一个密钥是另一个的子串时残留
    return sorted(s, key=len, reverse=True)


def _redact_str(s: str, secrets: list[str]) -> str:
    for sec in secrets:
        if sec in s:
            s = s.replace(sec, MASK)
    return s


def redact(value: Any, secrets: Iterable[str] | None = None) -> Any:
    """递归替换密钥：SecretStr → MASK；字符串中出现的已登记密钥子串 → MASK。

    支持 str / bytes / dict / list / tuple / set / pydantic 模型；其他类型原样返回。
    """
    secs = _all_secrets(secrets)
    return _redact(value, secs)


def _redact(value: Any, secs: list[str]) -> Any:
    if isinstance(value, SecretStr):
        return MASK
    if isinstance(value, str):
        return _redact_str(value, secs)
    if isinstance(value, bytes):
        return _redact_str(value.decode("utf-8", errors="replace"), secs).encode("utf-8")
    if isinstance(value, BaseModel):
        return _redact(value.model_dump(mode="json"), secs)
    if isinstance(value, Mapping):
        return {_redact(k, secs): _redact(v, secs) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, secs) for v in value]
    if isinstance(value, tuple):
        return tuple(_redact(v, secs) for v in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_redact(v, secs) for v in value)
    return value


def config_snapshot(settings: Settings) -> dict[str, Any]:
    """去除密钥后的 JSON 兼容配置快照（用于 ``runs.config_json`` 与报告）。"""
    dumped = settings.model_dump(mode="json")
    dumped["azure"]["api_key"] = MASK
    return redact(dumped, settings.secret_values())


def config_snapshot_json(settings: Settings) -> str:
    return json.dumps(config_snapshot(settings), ensure_ascii=False, sort_keys=True)


class RedactingFilter(logging.Filter):
    """日志过滤器：在格式化后的消息中掩码所有已登记密钥。"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - 交给 logging 自身处理格式错误
            return True
        redacted = redact(msg)
        if redacted != msg or record.args:
            record.msg = redacted
            record.args = None
        return True


__all__ = [
    "Lang",
    "MASK",
    "EXIT_CONFIG_ERROR",
    "ENV_AZURE_API_KEY",
    "ENV_AZURE_ENDPOINT",
    "ENV_GOOGLE_CREDENTIALS",
    "AzureSettings",
    "GeminiSettings",
    "LoadSettings",
    "ThrottleSettings",
    "CorpusSettings",
    "CostSettings",
    "ModelPrice",
    "Settings",
    "SettingsError",
    "load_settings",
    "load_settings_or_exit",
    "register_secrets",
    "clear_registered_secrets",
    "redact",
    "config_snapshot",
    "config_snapshot_json",
    "RedactingFilter",
]
