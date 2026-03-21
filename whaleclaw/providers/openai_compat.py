"""OpenAI-compatible chat completions adapter.

Many providers (OpenAI, DeepSeek, Zhipu, MiniMax, Moonshot, NVIDIA)
expose an OpenAI-compatible ``/chat/completions`` endpoint.  This module
provides a reusable base that all of them inherit.
"""

from __future__ import annotations

import asyncio
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



class _ToolCallAccumulator:
    """Accumulates streamed tool_call deltas into complete ToolCall objects."""

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def feed(self, index: int, delta: dict[str, Any]) -> None:
        if index not in self._calls:
            self._calls[index] = {"id": "", "name": "", "arguments": ""}
        acc = self._calls[index]
        if delta.get("id"):
            acc["id"] = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            acc["name"] += fn["name"]
        if fn.get("arguments"):
            acc["arguments"] += fn["arguments"]

    def build(self) -> list[ToolCall]:
        result: list[ToolCall] = []
        for _idx in sorted(self._calls):
            acc = self._calls[_idx]
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            result.append(ToolCall(
                id=acc["id"] or f"call_{_idx}",
                name=acc["name"],
                arguments=args,
            ))
        return result


class OpenAICompatProvider(LLMProvider):
    """Base adapter for any OpenAI-compatible chat completions API."""

    supports_native_tools = True
    supports_cache_control = False

    provider_name: str = "openai"
    default_base_url: str = "https://api.openai.com/v1"
    env_key: str = "OPENAI_API_KEY"
    max_network_retries: int = 2

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key or os.environ.get(self.env_key, "")
        self._base_url = (base_url or self.default_base_url).rstrip("/")
        self._timeout = timeout
        if not self._api_key:
            raise ProviderAuthError(f"{self.env_key} 未配置")

    def _build_body(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any]:
        messages = repair_tool_call_pairs(messages)
        msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "assistant" and m.tool_calls:
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in m.tool_calls
                    ],
                }
                msgs.append(msg)
            elif m.role == "tool" and m.tool_call_id:
                msgs.append({
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content or "",
                })
            elif m.images and m.role == "user":
                parts: list[dict[str, Any]] = []
                for img in m.images:
                    parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.mime};base64,{img.data}",
                        },
                    })
                if m.content:
                    parts.append({"type": "text", "text": m.content})
                msgs.append({"role": "user", "content": parts})
            else:
                msgs.append({"role": m.role, "content": m.content or ""})

        body: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "stream": True,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        return body

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        body = self._build_body(messages, model, tools)
        headers = self._build_headers()
        url = f"{self._base_url}/chat/completions"

        retryable_errors: tuple[type[Exception], ...] = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
        )
        last_exc: Exception | None = None
        max_attempts = self.max_network_retries + 1

        for attempt in range(1, max_attempts + 1):
            t0 = asyncio.get_event_loop().time()
            collected: list[str] = []
            input_tokens = 0
            output_tokens = 0
            stop_reason: str | None = None
            tc_acc = _ToolCallAccumulator()
            try:
                async with (
                    httpx.AsyncClient(timeout=self._timeout) as client,
                    client.stream("POST", url, json=body, headers=headers) as resp,
                ):
                        if resp.status_code == 401:
                            raise ProviderAuthError(f"{self.provider_name} API Key 无效")
                        if resp.status_code == 429:
                            raise ProviderRateLimitError(f"{self.provider_name} API 速率限制")
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            raise ProviderError(
                                f"{self.provider_name} API error {resp.status_code}: "
                                f"{error_body.decode(errors='replace')}"
                            )

                        async for line in resp.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[6:].strip()
                            if payload == "[DONE]":
                                break

                            try:
                                event = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            usage = event.get("usage")
                            if usage:
                                input_tokens = usage.get(
                                    "prompt_tokens",
                                    usage.get("input_tokens", input_tokens),
                                )
                                output_tokens = usage.get(
                                    "completion_tokens",
                                    usage.get("output_tokens", output_tokens),
                                )
                                if input_tokens == 0 and output_tokens == 0:
                                    total = usage.get("total_tokens", 0)
                                    if total > 0:
                                        input_tokens = total
                                        output_tokens = 0

                            for choice in event.get("choices") or []:
                                delta = choice.get("delta", {})
                                text = delta.get("content", "")
                                if text:
                                    collected.append(text)
                                    if on_stream:
                                        await on_stream(text)

                                for tc_delta in delta.get("tool_calls") or []:
                                    tc_acc.feed(tc_delta.get("index", 0), tc_delta)

                                fr = choice.get("finish_reason")
                                if fr:
                                    stop_reason = fr
            except retryable_errors as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                backoff_seconds = 0.25 * attempt
                log.warning(
                    f"{self.provider_name}.network_retry",
                    model=model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    error=exc.__class__.__name__,
                )
                await asyncio.sleep(backoff_seconds)
                continue

            full_text = "".join(collected)
            tool_calls = tc_acc.build()
            elapsed_ms = int((asyncio.get_event_loop().time() - t0) * 1000)
            log.debug(
                f"{self.provider_name}.response",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=len(tool_calls),
                elapsed_ms=elapsed_ms,
                attempt=attempt,
            )
            return AgentResponse(
                content=full_text,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                stop_reason=stop_reason,
                tool_calls=tool_calls,
            )

        if last_exc is not None:
            raise ProviderError(
                f"{self.provider_name} 网络连接失败({last_exc.__class__.__name__})"
            ) from last_exc
        raise ProviderError(f"{self.provider_name} 请求失败")
