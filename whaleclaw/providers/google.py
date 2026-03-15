"""Google Gemini provider adapter.

Gemini uses a different API format from OpenAI, so this is a standalone
implementation using the ``generateContent`` endpoint with SSE streaming.
"""

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

_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


class GoogleProvider(LLMProvider):
    """Google Generative AI (Gemini) adapter with streaming."""

    supports_native_tools = True
    supports_cache_control = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._timeout = timeout
        if not self._api_key:
            raise ProviderAuthError("GOOGLE_API_KEY 未配置")

    def _build_body(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any]:
        messages = repair_tool_call_pairs(messages)
        system_text = ""
        contents: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_text += msg.content + "\n"
            elif msg.role == "assistant" and msg.tool_calls:
                parts: list[dict[str, Any]] = []
                if msg.content:
                    parts.append({"text": msg.content})
                for tc in msg.tool_calls:
                    parts.append({
                        "functionCall": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    })
                contents.append({"role": "model", "parts": parts})
            elif msg.role == "tool" and msg.tool_call_id:
                contents.append({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.tool_call_id,
                                "response": {"result": msg.content},
                            }
                        }
                    ],
                })
            elif msg.role == "user":
                parts_list: list[dict[str, Any]] = []
                if msg.images:
                    for img in msg.images:
                        parts_list.append({
                            "inline_data": {
                                "mime_type": img.mime,
                                "data": img.data,
                            },
                        })
                if msg.content:
                    parts_list.append({"text": msg.content})
                contents.append({"role": "user", "parts": parts_list or [{"text": ""}]})
            elif msg.role == "assistant":
                contents.append(
                    {"role": "model", "parts": [{"text": msg.content}]}
                )

        body: dict[str, Any] = {"contents": contents}
        if system_text.strip():
            body["system_instruction"] = {"parts": [{"text": system_text.strip()}]}

        if tools:
            body["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                        for t in tools
                    ]
                }
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
        body = self._build_body(messages, tools)
        url = f"{_BASE_URL}/models/{model}:streamGenerateContent?alt=sse&key={self._api_key}"

        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        tool_calls: list[ToolCall] = []
        tc_idx = 0

        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", url, json=body) as resp,
        ):
                if resp.status_code == 401:
                    raise ProviderAuthError("Google API Key 无效")
                if resp.status_code == 429:
                    raise ProviderRateLimitError("Google API 速率限制")
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise ProviderError(
                        f"Google API error {resp.status_code}: "
                        f"{error_body.decode(errors='replace')}"
                    )

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if not payload:
                        continue

                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    for candidate in event.get("candidates", []):
                        content = candidate.get("content", {})
                        for part in content.get("parts", []):
                            text = part.get("text", "")
                            if text:
                                collected.append(text)
                                if on_stream:
                                    await on_stream(text)
                            fc = part.get("functionCall")
                            if fc:
                                tool_calls.append(ToolCall(
                                    id=f"gemini_call_{tc_idx}",
                                    name=fc.get("name", ""),
                                    arguments=fc.get("args", {}),
                                ))
                                tc_idx += 1

                    usage = event.get("usageMetadata", {})
                    input_tokens = usage.get("promptTokenCount", input_tokens)
                    output_tokens = usage.get(
                        "candidatesTokenCount", output_tokens
                    )

        full_text = "".join(collected)
        stop = "tool_use" if tool_calls else "stop"
        log.debug(
            "google.response", model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            tool_calls=len(tool_calls),
        )
        return AgentResponse(
            content=full_text,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop,
            tool_calls=tool_calls,
        )
