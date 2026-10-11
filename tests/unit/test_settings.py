"""Task 1.3：配置加载与校验（Requirements 1.2–1.9）。"""
from __future__ import annotations

import copy
import io
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from loadtest.settings import (
    ENV_AZURE_API_KEY,
    ENV_AZURE_ENDPOINT,
    ENV_GOOGLE_CREDENTIALS,
    EXIT_CONFIG_ERROR,
    MASK,
    RedactingFilter,
    SettingsError,
    ThrottleSettings,
    clear_registered_secrets,
    config_snapshot,
    config_snapshot_json,
    load_settings,
    load_settings_or_exit,
    redact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "config.example.yaml"

API_KEY = "sk-test-SECRET-0123456789abcdefXYZ"
ENDPOINT = "https://example-resource.openai.azure.com/"
CREDS_CONTENT = '{"type": "service_account", "private_key": "PRIVATE-KEY-CONTENT-QWERTY"}'


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_secrets():
    clear_registered_secrets()
    yield
    clear_registered_secrets()


@pytest.fixture
def base_cfg() -> dict[str, Any]:
    with open(EXAMPLE_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def creds_file(tmp_path: Path) -> Path:
    p = tmp_path / "sa.json"
    p.write_text(CREDS_CONTENT, encoding="utf-8")
    return p


@pytest.fixture
def environ(creds_file: Path) -> dict[str, str]:
    return {
        ENV_AZURE_API_KEY: API_KEY,
        ENV_AZURE_ENDPOINT: ENDPOINT,
        ENV_GOOGLE_CREDENTIALS: str(creds_file),
    }


def write_cfg(tmp_path: Path, cfg: dict[str, Any], name: str = "config.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def load(tmp_path: Path, cfg: dict[str, Any], environ: dict[str, str]):
    return load_settings(write_cfg(tmp_path, cfg), env_file=None, environ=environ)


def expect_error(tmp_path: Path, cfg: dict[str, Any], environ: dict[str, str]) -> SettingsError:
    with pytest.raises(SettingsError) as ei:
        load(tmp_path, cfg, environ)
    err = ei.value
    text = str(err)
    # 错误信息绝不包含密钥或凭据文件内容
    assert API_KEY not in text
    assert "PRIVATE-KEY-CONTENT-QWERTY" not in text
    return err


def with_change(cfg: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    *parents, leaf = path.split(".")
    node = out
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value
    return out


# ---------------------------------------------------------------------------
# 基本加载
# ---------------------------------------------------------------------------


def test_example_config_loads(tmp_path, base_cfg, environ):
    s = load(tmp_path, base_cfg, environ)
    assert s.azure.api_key.get_secret_value() == API_KEY
    assert str(s.azure.endpoint).startswith("https://example-resource.openai.azure.com")
    assert s.gemini.credentials_file == Path(environ[ENV_GOOGLE_CREDENTIALS])


def test_env_file_is_read(tmp_path, base_cfg, creds_file):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"{ENV_AZURE_API_KEY}={API_KEY}\n{ENV_AZURE_ENDPOINT}={ENDPOINT}\n"
        f"{ENV_GOOGLE_CREDENTIALS}={creds_file.as_posix()}\n",
        encoding="utf-8",
    )
    s = load_settings(write_cfg(tmp_path, base_cfg), env_file=env_file, environ={})
    assert s.azure.api_key.get_secret_value() == API_KEY


def test_defaults_with_minimal_yaml(tmp_path, environ):
    s = load(tmp_path, {"gemini": {"project": "p"}}, environ)
    assert s.azure.quota_tpm == 1_000_000
    assert s.azure.quota_rpm == 1_000
    assert s.load.target_tpm == 800_000
    assert s.azure.deployment == "gpt-4.1-nano"
    assert s.azure.model_family == "gpt-4.1-nano"
    assert s.load.duration_s == 1200
    assert s.load.lang_ratio == {"zh": 0.60, "en": 0.40}
    assert s.throttle.mode == "probe"
    assert s.gemini.throttle.mode == "level"


def test_default_recover_step(tmp_path, environ, base_cfg):
    s = load(tmp_path, {"gemini": {"project": "p"}}, environ)
    assert s.throttle.recover_step_s == pytest.approx(15.0)
    assert s.gemini.throttle.recover_step_s == pytest.approx(10.0)
    s2 = load(tmp_path, base_cfg, environ)
    assert s2.throttle.recover_step_s == pytest.approx(15.0)
    assert s2.gemini.throttle.recover_step_s == pytest.approx(10.0)
    # 直接构造的默认值即 Azure 档位
    assert ThrottleSettings().recover_step_s == pytest.approx(15.0)


def test_gemini_partial_throttle_uses_gemini_defaults(tmp_path, base_cfg, environ):
    cfg = copy.deepcopy(base_cfg)
    cfg["gemini"]["throttle"] = {"min_cooldown_s": 12.0}
    s = load(tmp_path, cfg, environ)
    assert s.gemini.throttle.mode == "level"
    assert s.gemini.throttle.levels == [0.10, 0.50, 1.00]
    assert s.gemini.throttle.recover_step_s == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 缺失 / 非法密钥
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [ENV_AZURE_API_KEY, ENV_AZURE_ENDPOINT, ENV_GOOGLE_CREDENTIALS])
def test_missing_secret_reports_param_name(tmp_path, base_cfg, environ, missing):
    env = {k: v for k, v in environ.items() if k != missing}
    err = expect_error(tmp_path, base_cfg, env)
    assert missing in err.params


def test_empty_secret_treated_as_missing(tmp_path, base_cfg, environ):
    env = {**environ, ENV_AZURE_API_KEY: ""}
    err = expect_error(tmp_path, base_cfg, env)
    assert ENV_AZURE_API_KEY in err.params


@pytest.mark.parametrize(
    "path, env_name",
    [
        ("azure.api_key", ENV_AZURE_API_KEY),
        ("azure.endpoint", ENV_AZURE_ENDPOINT),
        ("gemini.credentials_file", ENV_GOOGLE_CREDENTIALS),
    ],
)
def test_secrets_in_yaml_rejected(tmp_path, base_cfg, environ, path, env_name):
    value = API_KEY if path == "azure.api_key" else "x"
    err = expect_error(tmp_path, with_change(base_cfg, path, value), environ)
    assert env_name in err.params


def test_credentials_file_missing(tmp_path, base_cfg, environ):
    env = {**environ, ENV_GOOGLE_CREDENTIALS: str(tmp_path / "nope" / "missing.json")}
    err = expect_error(tmp_path, base_cfg, env)
    assert err.params == [ENV_GOOGLE_CREDENTIALS]


def test_credentials_file_is_directory(tmp_path, base_cfg, environ):
    env = {**environ, ENV_GOOGLE_CREDENTIALS: str(tmp_path)}
    err = expect_error(tmp_path, base_cfg, env)
    assert ENV_GOOGLE_CREDENTIALS in err.params


def test_credentials_content_never_in_errors(tmp_path, base_cfg, environ):
    # 凭据文件存在但另有错误：错误中不得包含凭据内容
    err = expect_error(tmp_path, with_change(base_cfg, "load.duration_s", 0), environ)
    assert err.params == ["load.duration_s"]
    for _, msg in err.errors:
        assert "PRIVATE-KEY" not in msg


# ---------------------------------------------------------------------------
# 非法参数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ratio",
    [
        {"zh": 0.6, "en": 0.5},  # 和 ≠ 1
        {"zh": 1.2, "en": -0.2},  # 越界
        {"zh": 1.0},  # 缺少 en
        {"zh": 0.5, "en": 0.3, "fr": 0.2},  # 未知语言
    ],
)
def test_invalid_lang_ratio(tmp_path, base_cfg, environ, ratio):
    err = expect_error(tmp_path, with_change(base_cfg, "load.lang_ratio", ratio), environ)
    assert any(p.startswith("load.lang_ratio") for p in err.params)


@pytest.mark.parametrize(
    "ramp",
    [
        [],
        [[1.0, 0.5], [10.0, 1.0]],  # 首项时间 ≠ 0
        [[0.0, 0.5], [0.0, 1.0]],  # 时间非递增
        [[0.0, 0.5], [10.0, 0.8]],  # 末项 ≠ 1.0
        [[0.0, 0.0], [10.0, 1.0]],  # 档位 ∉ (0, 1]
        [[0.0, 1.5]],
    ],
)
def test_invalid_startup_ramp(tmp_path, base_cfg, environ, ramp):
    err = expect_error(tmp_path, with_change(base_cfg, "load.startup_ramp", ramp), environ)
    assert any(p.startswith("load.startup_ramp") for p in err.params)


def test_valid_startup_ramp_single_step(tmp_path, base_cfg, environ):
    s = load(tmp_path, with_change(base_cfg, "load.startup_ramp", [[0.0, 1.0]]), environ)
    assert s.load.startup_ramp == [(0.0, 1.0)]


@pytest.mark.parametrize("field", ["input_per_m", "output_per_m", "cached_input_per_m"])
def test_negative_price(tmp_path, base_cfg, environ, field):
    cfg = copy.deepcopy(base_cfg)
    cfg["cost"]["prices"]["gpt-4.1-nano"][field] = -0.01
    err = expect_error(tmp_path, cfg, environ)
    assert any(p.startswith("cost.prices") and field in p for p in err.params)


def test_missing_price_for_used_model(tmp_path, base_cfg, environ):
    cfg = copy.deepcopy(base_cfg)
    del cfg["cost"]["prices"]["gemini-3.8-flash"]
    err = expect_error(tmp_path, cfg, environ)
    assert any("gemini-3.8-flash" in msg for _, msg in err.errors)


def test_gemini3_defaults(tmp_path, base_cfg, environ):
    from loadtest.settings import GeminiSettings

    f = GeminiSettings.model_fields
    assert f["model_chain"].default_factory() == ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]
    assert f["expensive_models"].default_factory() == ["gemini-3.8-flash"]
    assert f["temperature"].default == 1.0
    levels = f["thinking_levels"].default_factory()
    assert levels["gemini-3.1-flash-lite"] == "minimal" and levels["gemini-3.8-flash"] == "low"


@pytest.mark.parametrize("field", ["azure_budget_cap_usd", "gemini_budget_cap_usd", "per_run_cap_usd"])
@pytest.mark.parametrize("value", [0, -1.0])
def test_invalid_budget(tmp_path, base_cfg, environ, field, value):
    err = expect_error(tmp_path, with_change(base_cfg, f"cost.{field}", value), environ)
    assert f"cost.{field}" in err.params


def test_negative_safety_margin(tmp_path, base_cfg, environ):
    err = expect_error(tmp_path, with_change(base_cfg, "cost.safety_margin_usd", -0.1), environ)
    assert "cost.safety_margin_usd" in err.params


def test_unknown_key_rejected(tmp_path, base_cfg, environ):
    err = expect_error(tmp_path, with_change(base_cfg, "load.target_tmp", 1), environ)
    assert "load.target_tmp" in err.params


def test_multiple_errors_reported_together(tmp_path, base_cfg, environ):
    cfg = with_change(base_cfg, "cost.azure_budget_cap_usd", -1)
    cfg = with_change(cfg, "load.lang_ratio", {"zh": 0.9, "en": 0.9})
    err = expect_error(tmp_path, cfg, environ)
    assert "cost.azure_budget_cap_usd" in err.params
    assert "load.lang_ratio" in err.params


def test_missing_config_file(tmp_path, environ):
    with pytest.raises(SettingsError) as ei:
        load_settings(tmp_path / "missing.yaml", env_file=None, environ=environ)
    assert ei.value.params == ["config"]


# ---------------------------------------------------------------------------
# ThrottleSettings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"levels": [0.10, 0.50, 0.25, 1.00]},  # 非递增
        {"levels": [0.10, 0.50, 0.50, 1.00]},  # 重复
        {"levels": [0.20, 0.50, 1.00]},  # 首项 ≠ 0.10
        {"levels": [0.10, 0.50, 0.90]},  # 末项 ≠ 1.00
        {"levels": [1.00]},  # 不足 2 项
        {"levels": [0.10, 0.25, 0.75, 1.00], "soft_down_floor": 0.50},  # floor ∉ levels
        {"soft_down_floor": 0.30},
        {"min_cooldown_s": 130.0, "max_retry_after_s": 120.0},
        {"probe_inflight_wait_s": 0.0},
        {"probe_inflight_wait_s": -1.0},
        {"probe_error_retry_s": 0.0},
        {"probe_error_retry_s": -5.0},
        {"recover_full_s": 0.0},
        {"min_cooldown_s": -1.0},
    ],
)
def test_invalid_throttle_settings(kwargs):
    with pytest.raises(ValidationError):
        ThrottleSettings(**kwargs)


def test_valid_throttle_settings():
    t = ThrottleSettings(levels=[0.10, 0.50, 1.00], recover_full_s=20.0, min_cooldown_s=120.0)
    assert t.recover_step_s == pytest.approx(10.0)


@pytest.mark.parametrize(
    "path, value",
    [
        ("throttle.levels", [0.10, 0.75, 0.50, 1.00]),
        ("throttle.levels", [0.05, 0.50, 1.00]),
        ("throttle.soft_down_floor", 0.6),
        ("throttle.min_cooldown_s", 200.0),
        ("throttle.probe_inflight_wait_s", 0),
        ("gemini.throttle.levels", [0.10, 0.50, 0.95]),
        ("gemini.throttle.probe_error_retry_s", 0),
    ],
)
def test_invalid_throttle_in_yaml(tmp_path, base_cfg, environ, path, value):
    err = expect_error(tmp_path, with_change(base_cfg, path, value), environ)
    section = path.rsplit(".", 1)[0]
    assert any(p.startswith(section) for p in err.params)


def test_azure_throttle_level_mode_rejected(tmp_path, base_cfg, environ):
    err = expect_error(tmp_path, with_change(base_cfg, "throttle.mode", "level"), environ)
    assert "throttle" in err.params


def test_gemini_throttle_probe_mode_rejected(tmp_path, base_cfg, environ):
    err = expect_error(tmp_path, with_change(base_cfg, "gemini.throttle.mode", "probe"), environ)
    assert "gemini.throttle" in err.params


# ---------------------------------------------------------------------------
# 掩码：快照与日志中原始 key 子串出现 0 次
# ---------------------------------------------------------------------------


def test_snapshot_has_no_raw_key(tmp_path, base_cfg, environ):
    s = load(tmp_path, base_cfg, environ)
    snap = config_snapshot(s)
    assert snap["azure"]["api_key"] == MASK
    text = config_snapshot_json(s)
    assert text.count(API_KEY) == 0
    assert "PRIVATE-KEY-CONTENT-QWERTY" not in text
    assert MASK in text
    assert API_KEY not in repr(s) and API_KEY not in str(s)


def test_redact_nested_structures(tmp_path, base_cfg, environ):
    load(tmp_path, base_cfg, environ)  # 登记密钥
    data = {"a": [f"x{API_KEY}y", (API_KEY,)], API_KEY: {API_KEY}, "b": API_KEY.encode()}
    out = redact(data)
    assert str(out).count(API_KEY) == 0
    assert out["a"][0] == f"x{MASK}y"


def test_logging_filter_masks_key(tmp_path, base_cfg, environ):
    s = load(tmp_path, base_cfg, environ)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.addFilter(RedactingFilter())
    logger = logging.getLogger("loadtest.test_settings.redact")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)
    try:
        logger.info("key=%s", API_KEY)
        logger.warning(f"inline {API_KEY} here")
        logger.error("snapshot %s", config_snapshot_json(s))
        logger.info("dict %r", {"api_key": API_KEY})
    finally:
        logger.removeHandler(handler)
    out = buf.getvalue()
    assert out.count(API_KEY) == 0
    assert out.count(MASK) >= 3


def test_redact_without_registration_uses_extra():
    assert redact(f"a{API_KEY}b", [API_KEY]) == f"a{MASK}b"
    # 未登记时原样返回
    assert redact(f"a{API_KEY}b") == f"a{API_KEY}b"


# ---------------------------------------------------------------------------
# load_settings_or_exit
# ---------------------------------------------------------------------------


def test_load_settings_or_exit_nonzero(tmp_path, base_cfg, environ, capsys):
    cfg_path = write_cfg(tmp_path, with_change(base_cfg, "load.lang_ratio", {"zh": 0.7, "en": 0.7}))
    with pytest.raises(SystemExit) as ei:
        load_settings_or_exit(cfg_path, env_file=None, environ=environ)
    assert ei.value.code == EXIT_CONFIG_ERROR
    assert ei.value.code != 0
    err = capsys.readouterr().err
    assert "load.lang_ratio" in err
    assert API_KEY not in err


def test_load_settings_or_exit_missing_key(tmp_path, base_cfg, environ, capsys):
    env = {k: v for k, v in environ.items() if k != ENV_AZURE_API_KEY}
    with pytest.raises(SystemExit) as ei:
        load_settings_or_exit(write_cfg(tmp_path, base_cfg), env_file=None, environ=env)
    assert ei.value.code not in (0, None)
    assert ENV_AZURE_API_KEY in capsys.readouterr().err


def test_load_settings_or_exit_success(tmp_path, base_cfg, environ):
    s = load_settings_or_exit(write_cfg(tmp_path, base_cfg), env_file=None, environ=environ)
    assert s.azure.quota_tpm == 1_000_000
