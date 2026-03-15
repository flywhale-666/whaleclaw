"""Anthropic Claude provider adapter using httpx streaming."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from whaleclaw.providers.base import (
    AgentResponse,
    LLMProvider,
    Message,
    ToolCall,
    ToolSchema,
    repair_tool_call_pairs,
)
from whaleclaw.types import ProviderAuthError, ProviderError, ProviderRateLimitError, StreamCallback
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 8192


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API adapter with streaming support."""

    supports_native_tools = True
    supports_cache_control = True

    def __init__(self, api_key: str | None = None, *, timeout: int = 120) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._timeout = timeout
        if not self._api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY 未配置")

    def _build_body(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any]:
        messages = repair_tool_call_pairs(messages)
        system_parts: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                block: dict[str, Any] = {"type": "text", "text": msg.content}
                if msg.cache_control:
                    block["cache_control"] = msg.cache_control.model_dump()
                system_parts.append(block)
            elif msg.role == "assistant" and msg.tool_calls:
                content_blocks: list[dict[str, Any]] = []
                if msg.content:
                    content_blocks.append(
                        {"type": "text", "text": msg.content}
                    )
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    })
                conversation.append({
                    "role": "assistant",
                    "content": content_blocks,
                })
            elif msg.role == "tool" and msg.tool_call_id:
                conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    ],
                })
            elif msg.images and msg.role == "user":
                blocks: list[dict[str, Any]] = []
                for img in msg.images:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.mime,
                            "data": img.data,
                        },
                    })
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                conversation.append({"role": "user", "content": blocks})
            else:
                conversation.append(
                    {"role": msg.role, "content": msg.content}
                )

        body: dict[str, Any] = {
            "model": model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "stream": True,
        }
        if system_parts:
            body["system"] = system_parts
        if conversation:
            body["messages"] = conversation
        else:
            body["messages"] = [{"role": "user", "content": ""}]

        if tools:
            body["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in tools
            ]

        return body

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        """Call Anthropic Messages API with SSE streaming."""
        body = self._build_body(messages, model, tools)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None

        current_block_type: str | None = None
        current_tool_id: str = ""
        current_tool_name: str = ""
        tool_input_json: list[str] = []
        tool_calls: list[ToolCall] = []

        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", _API_URL, json=body, headers=headers) as resp,
        ):
                if resp.status_code == 401:
                    raise ProviderAuthError("Anthropic API Key 无效")
                if resp.status_code == 429:
                    raise ProviderRateLimitError("Anthropic API 速率限制")
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise ProviderError(
                        f"Anthropic API error {resp.status_code}: "
                        f"{error_body.decode(errors='replace')}"
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        break

                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif event_type == "content_block_start":
                        cb = event.get("content_block", {})
                        current_block_type = cb.get("type")
                        if current_block_type == "tool_use":
                            current_tool_id = cb.get("id", "")
                            current_tool_name = cb.get("name", "")
                            tool_input_json = []

                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        delta_type = delta.get("type", "")
                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            collected.append(text)
                            if on_stream and text:
                                await on_stream(text)
                        elif delta_type == "input_json_delta":
                            tool_input_json.append(
                                delta.get("partial_json", "")
                            )

                    elif event_type == "content_block_stop":
                        if current_block_type == "tool_use":
                            raw = "".join(tool_input_json)
                            try:
                                args = json.loads(raw) if raw else {}
                            except json.JSONDecodeError:
                                args = {}
                            tool_calls.append(ToolCall(
                                id=current_tool_id,
                                name=current_tool_name,
                                arguments=args,
                            ))
                        current_block_type = None

                    elif event_type == "message_delta":
                        delta = event.get("delta", {})
                        stop_reason = delta.get("stop_reason")
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

        full_text = "".join(collected)
        log.debug(
            "anthropic.response",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=len(tool_calls),
        )
        return AgentResponse(
            content=full_text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )
