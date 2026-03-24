"""Configuration loading, merging, and validation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

from whaleclaw.config.paths import CONFIG_FILE
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.types import ConfigError

_config: WhaleclawConfig | None = None

_ENV_PREFIX = "WHALECLAW_"


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value
    return base


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"配置文件读取失败: {path} — {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件格式错误 (期望 JSON 对象): {path}")
    return data  # type: ignore[return-value]


def _env_overrides() -> dict[str, object]:
    """Build a nested dict from ``WHALECLAW_*`` environment variables.

    Example: ``WHALECLAW_GATEWAY_PORT=9000`` → ``{"gateway": {"port": 9000}}``.
    """
    result: dict[str, object] = {}
    for key, value in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        parts = key[len(_ENV_PREFIX) :].lower().split("_")
        node: dict[str, object] = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})  # type: ignore[assignment]
        raw: object = value
        if value.isdigit():
            raw = int(value)
        elif value.lower() in ("true", "false"):
            raw = value.lower() == "true"
        node[parts[-1]] = raw
    return result


def _persist_auto_generated_auth(cfg: WhaleclawConfig, user_cfg_path: Path) -> None:
    """首次自动生成的 token / jwt_secret 持久化到用户配置文件，避免每次重启变化。"""
    auth = cfg.gateway.auth
    if auth.mode != "token" or not auth.token:
        return

    existing = _load_json(user_cfg_path) if user_cfg_path.is_file() else {}
    gw = existing.get("gateway")
    if not isinstance(gw, dict):
        gw = {}
        existing["gateway"] = gw
    gw_dict = cast(dict[str, object], gw)
    auth_dict_raw = gw_dict.get("auth")
    if not isinstance(auth_dict_raw, dict):
        auth_dict_raw = {}
        gw_dict["auth"] = auth_dict_raw
    auth_dict = cast(dict[str, object], auth_dict_raw)

    changed = False
    if not auth_dict.get("token"):
        auth_dict["token"] = auth.token
        changed = True
    if not auth_dict.get("jwt_secret"):
        auth_dict["jwt_secret"] = auth.jwt_secret
        changed = True
    if "mode" not in auth_dict:
        auth_dict["mode"] = auth.mode
        changed = True

    if changed:
        user_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        user_cfg_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def load_config(
    *,
    config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
) -> WhaleclawConfig:
    """Load, merge and validate configuration.

    Priority (high → low):
    1. *cli_overrides*
    2. Environment variables (``WHALECLAW_`` prefix)
    3. User config file (``~/.whaleclaw/whaleclaw.json``)
    4. Project config file (``./whaleclaw.json``)
    5. Schema defaults
    """
    merged: dict[str, object] = {}

    project_cfg = Path("whaleclaw.json")
    _deep_merge(merged, _load_json(project_cfg))

    user_cfg = config_path or CONFIG_FILE
    _deep_merge(merged, _load_json(user_cfg))

    _deep_merge(merged, _env_overrides())

    if cli_overrides:
        _deep_merge(merged, cli_overrides)

    try:
        cfg = WhaleclawConfig.model_validate(merged)
    except Exception as exc:
        raise ConfigError(f"配置校验失败: {exc}") from exc

    _persist_auto_generated_auth(cfg, config_path or CONFIG_FILE)

    global _config  # noqa: PLW0603
    _config = cfg
    return cfg


def get_config() -> WhaleclawConfig:
    """Return the current global config, loading defaults if needed."""
    global _config  # noqa: PLW0603
    if _config is None:
        _config = load_config()
    return _config


def set_default_agent_model(model: str, *, config_path: Path | None = None) -> None:
    """Persist ``agent.model`` into the user config file."""
    model_id = model.strip()
    if not model_id:
        raise ConfigError("默认模型不能为空")

    user_cfg = config_path or CONFIG_FILE
    user_cfg.parent.mkdir(parents=True, exist_ok=True)

    raw_cfg: dict[str, object] = _load_json(user_cfg) if user_cfg.exists() else {}

    agent_cfg = raw_cfg.get("agent")
    agent_node: dict[str, object]
    if agent_cfg is None:
        agent_node = {}
        raw_cfg["agent"] = agent_node
    elif isinstance(agent_cfg, dict):
        raw_agent_cfg = cast(dict[object, object], agent_cfg)
        agent_node = {}
        for item_key, item_value in raw_agent_cfg.items():
            if isinstance(item_key, str):
                agent_node[item_key] = item_value
        raw_cfg["agent"] = agent_node
    else:
        raise ConfigError("配置文件格式错误: agent 必须是对象")

    agent_node["model"] = model_id
    user_cfg.write_text(
        json.dumps(raw_cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def reset_config() -> None:
    """Reset global config (mainly for testing)."""
    global _config  # noqa: PLW0603
    _config = None
