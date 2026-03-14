"""Tests for the model router."""

from __future__ import annotations

import pytest

from whaleclaw.config.schema import ModelsConfig
from whaleclaw.providers.router import ModelRouter
from whaleclaw.types import ProviderError


class TestModelRouter:
    def test_resolve_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        router = ModelRouter(ModelsConfig())
        provider, model = router.resolve("anthropic/claude-sonnet-4-20250514")
        assert model == "claude-sonnet-4-20250514"

    def test_resolve_default_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        router = ModelRouter(ModelsConfig())
        provider, model = router.resolve("claude-sonnet-4-20250514")
        assert model == "claude-sonnet-4-20250514"

    def test_resolve_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        router = ModelRouter(ModelsConfig())
        provider, model = router.resolve("openai/gpt-5.2")
        assert model == "gpt-5.2"

    def test_resolve_nvidia_subpath(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
        router = ModelRouter(ModelsConfig())
        provider, model = router.resolve("nvidia/meta/llama-3.1-8b-instruct")
        assert model == "meta/llama-3.1-8b-instruct"

    def test_unknown_provider(self) -> None:
        router = ModelRouter(ModelsConfig())
        with pytest.raises(ProviderError, match="未配置 api_key"):
            router.resolve("unknown/model")

    def test_caches_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        router = ModelRouter(ModelsConfig())
        p1, _ = router.resolve("anthropic/model-a")
        p2, _ = router.resolve("anthropic/model-b")
        assert p1 is p2
