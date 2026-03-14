"""Pydantic v2 configuration schema for WhaleClaw."""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from whaleclaw.config.paths import WORKSPACE_DIR


class AuthConfig(BaseModel):
    """Authentication configuration for the Gateway."""

    mode: str = "none"
    password: str | None = None
    token: str | None = None
    jwt_secret: str = "whaleclaw-default-secret"
    jwt_expire_hours: int = 24


class GatewayConfig(BaseModel):
    """Gateway server configuration."""

    port: int = Field(default=18666, ge=1, le=65535)
    bind: str = "127.0.0.1"
    verbose: bool = False
    auth: AuthConfig = Field(default_factory=AuthConfig)


class ProviderModelEntry(BaseModel):
    """A single validated model under a provider (e.g. one NVIDIA NIM model)."""

    id: str
    name: str = ""
    base_url: str | None = None
    verified: bool = False
    thinking: str = "off"


class ProviderConfig(BaseModel):
    """Per-provider configuration."""

    api_key: str | None = None
    base_url: str | None = None
    timeout: int = 120
    configured_models: list[ProviderModelEntry] = Field(default_factory=list)
    auth_mode: Literal["api_key", "oauth"] = "api_key"
    oauth_access: str | None = None
    oauth_refresh: str | None = None
    oauth_expires: int = 0
    oauth_account_id: str | None = None


class ModelsConfig(BaseModel):
    """Configuration for all LLM providers.

    内置 provider 用显式字段声明；用户通过「自定义」添加的 OpenAI
    兼容 provider 存储在 ``model_extra`` 中（Pydantic ``extra='allow'``），
    可通过 :meth:`all_provider_names` / :meth:`get_provider` 统一访问。
    """

    model_config = {"extra": "allow"}

    anthropic: ProviderConfig = Field(default_factory=ProviderConfig)
    openai: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    qwen: ProviderConfig = Field(default_factory=ProviderConfig)
    zhipu: ProviderConfig = Field(default_factory=ProviderConfig)
    minimax: ProviderConfig = Field(default_factory=ProviderConfig)
    moonshot: ProviderConfig = Field(default_factory=ProviderConfig)
    google: ProviderConfig = Field(default_factory=ProviderConfig)
    nvidia: ProviderConfig = Field(default_factory=ProviderConfig)

    BUILTIN_PROVIDERS: ClassVar[tuple[str, ...]] = (
        "anthropic", "openai", "deepseek", "qwen", "zhipu",
        "minimax", "moonshot", "google", "nvidia",
    )

    def custom_provider_names(self) -> list[str]:
        """返回所有用户自定义 provider 名称。"""
        extra = self.model_extra or {}
        return [k for k, v in extra.items() if isinstance(v, dict)]

    def all_provider_names(self) -> list[str]:
        """返回所有 provider 名称（内置 + 自定义）。"""
        return list(self.BUILTIN_PROVIDERS) + self.custom_provider_names()

    def get_provider(self, name: str) -> ProviderConfig:
        """按名称获取 ProviderConfig，自定义 provider 从 extra 中解析。"""
        if name in self.BUILTIN_PROVIDERS:
            return getattr(self, name)
        extra = self.model_extra or {}
        raw = extra.get(name)
        if isinstance(raw, dict):
            return ProviderConfig.model_validate(raw)
        return ProviderConfig()


class SummarizerConfig(BaseModel):
    """Configuration for L0/L1 hierarchical context compression.

    Uses a cheap, fast model to generate layered summaries of older
    conversation history, persisted in SQLite for on-demand loading.
    Set ``enabled = false`` or leave the model provider unconfigured
    to disable; context will simply be truncated when it overflows.
    """

    model: str = "zhipu/glm-4.7-flash"
    enabled: bool = True


class AgentMemoryConfig(BaseModel):
    """Long-term memory capture/recall settings."""

    enabled: bool = True
    auto_capture_mode: Literal["conservative", "balanced", "aggressive"] = "balanced"
    recall_limit: int = 8
    recall_max_tokens: int = 1600
    recall_profile_max_tokens: int = 1600
    recall_raw_max_tokens: int = 1600
    cooldown_seconds: int = 180
    max_per_hour: int = 12
    capture_batch_size: int = 2
    capture_merge_window_seconds: int = 45
    organizer_enabled: bool = True
    organizer_background: bool = True
    global_style_enabled: bool = True
    organizer_model: str = "zhipu/glm-4.7-flash"
    organizer_min_new_entries: int = 3
    organizer_interval_seconds: int = 180
    organizer_max_raw_window: int = 120
    keep_profile_versions: int = 3
    max_raw_entries: int = 800


class AgentConfig(BaseModel):
    """Agent runtime configuration."""

    model: str = "anthropic/claude-sonnet-4-20250514"
    max_tool_rounds: int = 50
    workspace: str = str(WORKSPACE_DIR)
    thinking_level: str = "off"
    auto_self_heal_on_error: bool = True
    self_heal_max_retries: int = Field(default=1, ge=0, le=3)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    memory: AgentMemoryConfig = Field(default_factory=AgentMemoryConfig)


class FeishuGroupConfig(BaseModel):
    """Per-group settings for Feishu."""

    require_mention: bool = True
    activation: Literal["mention", "always"] = "mention"


class FeishuChannelConfig(BaseModel):
    """Feishu bot channel configuration."""

    mode: Literal["ws", "webhook"] = "ws"
    app_id: str = ""
    app_secret: str = ""
    verification_token: str | None = None
    encrypt_key: str | None = None
    allow_from: list[str] = Field(default_factory=list)
    groups: dict[str, FeishuGroupConfig] = Field(default_factory=dict)
    dm_policy: Literal["pairing", "open", "closed"] = "pairing"
    webhook_path: str = "/webhook/feishu"


class ChannelsConfig(BaseModel):
    """Configuration for all message channels."""

    feishu: FeishuChannelConfig = Field(default_factory=FeishuChannelConfig)


class SecurityConfig(BaseModel):
    """Security configuration."""

    sandbox_mode: str = "non-main"
    dm_policy: str = "pairing"
    audit: bool = True


class RoutingRuleConfig(BaseModel):
    """Single routing rule in config."""

    name: str
    priority: int = 0
    match: dict[str, object] = Field(default_factory=dict)
    target: dict[str, object] = Field(default_factory=dict)


class RoutingConfig(BaseModel):
    """Routing configuration."""

    rules: list[RoutingRuleConfig] = Field(default_factory=list)


class WhaleclawConfig(BaseModel):
    """Root configuration model."""

    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    channels: ChannelsConfig = Field(default_factory=ChannelsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    plugins: dict[str, dict[str, object]] = Field(default_factory=dict)
