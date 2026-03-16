"""Tests for configuration loading, merging, and environment overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from whaleclaw.config.loader import (
    _deep_merge,
    _env_overrides,
    load_config,
    reset_config,
    set_default_agent_model,
)
from whaleclaw.types import ConfigError


class TestDeepMerge:
    def test_flat(self) -> None:
        base = {"a": 1, "b": 2}
        result = _deep_merge(base, {"b": 3, "c": 4})
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested(self) -> None:
        base: dict[str, object] = {"gateway": {"port": 18666, "bind": "127.0.0.1"}}
        result = _deep_merge(base, {"gateway": {"port": 9000}})
        assert result == {"gateway": {"port": 9000, "bind": "127.0.0.1"}}


class TestEnvOverrides:
    def test_parses_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WHALECLAW_GATEWAY_PORT", "9000")
        monkeypatch.setenv("WHALECLAW_GATEWAY_VERBOSE", "true")
        result = _env_overrides()
        assert result == {"gateway": {"port": 9000, "verbose": True}}


class TestLoadConfig:
    def test_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WHALECLAW_GATEWAY_PORT", raising=False)
        monkeypatch.chdir(tmp_path)
        reset_config()
        cfg = load_config(config_path=tmp_path / "nonexistent.json")
        assert cfg.gateway.port == 18666

    def test_cli_overrides(self) -> None:
        reset_config()
        cfg = load_config(cli_overrides={"gateway": {"port": 9000}})
        assert cfg.gateway.port == 9000

    def test_file_loading(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "whaleclaw.json"
        cfg_file.write_text(json.dumps({"gateway": {"port": 12345}}))
        reset_config()
        cfg = load_config(config_path=cfg_file)
        assert cfg.gateway.port == 12345

    def test_invalid_json(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("not json")
        reset_config()
        with pytest.raises(ConfigError):
            load_config(config_path=cfg_file)

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WHALECLAW_GATEWAY_PORT", "7777")
        reset_config()
        cfg = load_config()
        assert cfg.gateway.port == 7777


class TestSetDefaultAgentModel:
    def test_create_and_write(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "whaleclaw.json"
        set_default_agent_model("openai/gpt-5.2", config_path=cfg_file)
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert data["agent"]["model"] == "openai/gpt-5.2"

    def test_update_existing_agent_model(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "whaleclaw.json"
        cfg_file.write_text(
            json.dumps({"agent": {"model": "qwen/qwen3-plus"}, "gateway": {"port": 18666}}),
            encoding="utf-8",
        )
        set_default_agent_model("zhipu/glm-5", config_path=cfg_file)
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        assert data["agent"]["model"] == "zhipu/glm-5"
        assert data["gateway"]["port"] == 18666

    def test_reject_non_object_agent(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "whaleclaw.json"
        cfg_file.write_text(json.dumps({"agent": "bad"}), encoding="utf-8")
        with pytest.raises(ConfigError, match="agent 必须是对象"):
            set_default_agent_model("openai/gpt-5.2", config_path=cfg_file)
