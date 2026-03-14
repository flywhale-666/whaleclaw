"""Model router — resolves model IDs to provider instances."""

from __future__ import annotations

from whaleclaw.config.schema import ModelsConfig, ProviderConfig
from whaleclaw.providers.anthropic import AnthropicProvider
from whaleclaw.providers.base import AgentResponse, LLMProvider, Message, ToolSchema
from whaleclaw.providers.deepseek import DeepSeekProvider
from whaleclaw.providers.google import GoogleProvider
from whaleclaw.providers.minimax import MiniMaxProvider
from whaleclaw.providers.moonshot import MoonshotProvider
from whaleclaw.providers.nvidia import NvidiaProvider
from whaleclaw.providers.openai import OpenAIProvider
from whaleclaw.providers.openai_compat import OpenAICompatProvider
from whaleclaw.providers.qwen import QwenProvider
from whaleclaw.providers.zhipu import ZhipuProvider
from whaleclaw.types import ProviderError, StreamCallback
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_PROVIDER_MAP: dict[str, type[LLMProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "zhipu": ZhipuProvider,
    "minimax": MiniMaxProvider,
    "moonshot": MoonshotProvider,
    "google": GoogleProvider,
    "nvidia": NvidiaProvider,
}


def _get_provider_config(models_cfg: ModelsConfig, name: str) -> ProviderConfig:
    return models_cfg.get_provider(name)


class ModelRouter:
    """Route model IDs to the correct provider instance."""

    def __init__(self, models_config: ModelsConfig | None = None) -> None:
        self._models_config = models_config or ModelsConfig()
        self._cache: dict[str, LLMProvider] = {}

    def resolve(self, model_id: str) -> tuple[LLMProvider, str]:
        """Parse ``<provider>/<model>`` and return ``(provider_instance, model_name)``.

        For NVIDIA models like ``nvidia/meta/llama-3.1-8b-instruct``,
        the model_name keeps the sub-path (``meta/llama-3.1-8b-instruct``).

        自定义 provider（不在内置列表中的）会使用 OpenAICompatProvider
        作为通用适配器，需要配置中提供 base_url 和 api_key。
        """
        if "/" in model_id:
            provider_name, model_name = model_id.split("/", 1)
        else:
            provider_name, model_name = "anthropic", model_id

        if provider_name in self._cache:
            return self._cache[provider_name], model_name

        cfg = _get_provider_config(self._models_config, provider_name)
        is_custom = provider_name not in _PROVIDER_MAP

        if is_custom:
            if not cfg.api_key:
                raise ProviderError(
                    f"自定义提供商 '{provider_name}' 未配置 api_key"
                )
            if not cfg.base_url:
                raise ProviderError(
                    f"自定义提供商 '{provider_name}' 未配置 base_url"
                )
            try:
                instance: LLMProvider = OpenAICompatProvider(
                    api_key=cfg.api_key,
                    base_url=cfg.base_url,
                    timeout=cfg.timeout,
                )
            except Exception as exc:
                raise ProviderError(f"{provider_name} 初始化失败: {exc}") from exc
        else:
            cls = _PROVIDER_MAP[provider_name]
            kwargs: dict[str, object] = {"timeout": cfg.timeout}
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            if cfg.base_url and cls not in (AnthropicProvider, GoogleProvider):
                kwargs["base_url"] = cfg.base_url

            if provider_name == "openai" and cfg.auth_mode == "oauth":
                kwargs["auth_mode"] = "oauth"
                kwargs["oauth_access"] = cfg.oauth_access
                kwargs["oauth_refresh"] = cfg.oauth_refresh
                kwargs["oauth_expires"] = cfg.oauth_expires
                kwargs["oauth_account_id"] = cfg.oauth_account_id

            try:
                instance = cls(**kwargs)  # type: ignore[arg-type]
            except Exception as exc:
                raise ProviderError(f"{provider_name} 初始化失败: {exc}") from exc

        self._cache[provider_name] = instance
        return instance, model_name

    def supports_native_tools(self, model_id: str) -> bool:
        """Check if the provider for *model_id* supports native tools API.

        For NVIDIA NIM, this is model-dependent (only Llama/Mistral families).
        """
        provider, model_name = self.resolve(model_id)
        if isinstance(provider, NvidiaProvider):
            return NvidiaProvider.model_supports_tools(model_name)
        return provider.supports_native_tools

    async def chat(
        self,
        model_id: str,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        """Route to the correct provider and call chat."""
        provider, model_name = self.resolve(model_id)
        return await provider.chat(
            messages, model_name, tools=tools, on_stream=on_stream
        )

    async def chat_with_failover(
        self,
        model_ids: list[str],
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        """Try models in order, falling back on failure."""
        last_error: Exception | None = None
        for mid in model_ids:
            try:
                return await self.chat(mid, messages, tools=tools, on_stream=on_stream)
            except ProviderError as exc:
                log.warning("failover", model=mid, error=str(exc))
                last_error = exc
        raise last_error or ProviderError("所有模型均不可用")
