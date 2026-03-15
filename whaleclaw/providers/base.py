"""Abstract base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field

from whaleclaw.types import StreamCallback


class CacheControl(BaseModel):
    """Prompt caching hint (Anthropic / Google)."""

    type: Literal["ephemeral"] = "ephemeral"


class ToolSchema(BaseModel):
    """Tool JSON Schema passed to the LLM via native ``tools`` parameter."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    """A single tool-use request from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class ImageContent(BaseModel):
    """An inline image attached to a message."""

    mime: str
    data: str  # base64-encoded


class Message(BaseModel):
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    cache_control: CacheControl | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    images: list[ImageContent] | None = None


class AgentResponse(BaseModel):
    """Structured response from an LLM provider."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


def repair_tool_call_pairs(messages: list[Message]) -> list[Message]:
    """移除裁剪后失去配对的 tool_calls / tool result，防止 API 报 400。

    ContextWindow 裁剪可能导致 assistant(tool_calls) 和 tool(tool_call_id)
    不成对：
    - 孤儿 tool result → 直接删除
    - 孤儿 assistant.tool_calls 条目 → 从 tool_calls 列表中移除
    """
    provided_ids: set[str] = set()
    consumed_ids: set[str] = set()
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                provided_ids.add(tc.id)
        elif m.role == "tool" and m.tool_call_id:
            consumed_ids.add(m.tool_call_id)

    orphan_outputs = consumed_ids - provided_ids
    orphan_calls = provided_ids - consumed_ids
    if not orphan_outputs and not orphan_calls:
        return messages

    result: list[Message] = []
    for m in messages:
        if m.role == "tool" and m.tool_call_id in orphan_outputs:
            continue
        if m.role == "assistant" and m.tool_calls and orphan_calls:
            kept = [tc for tc in m.tool_calls if tc.id not in orphan_calls]
            m = Message(
                role=m.role,
                content=m.content,
                cache_control=m.cache_control,
                tool_calls=kept or None,
                tool_call_id=m.tool_call_id,
                images=m.images,
            )
        result.append(m)
    return result


class LLMProvider(ABC):
    """Abstract base for all LLM provider adapters."""

    supports_native_tools: bool = True
    supports_cache_control: bool = False

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        """Send messages and return a complete response.

        Args:
            messages: Conversation history including system prompt.
            model: Model identifier (e.g. ``claude-sonnet-4-20250514``).
            tools: Tool JSON schemas via native API parameter.
            on_stream: Optional callback invoked with each text chunk.
        """
