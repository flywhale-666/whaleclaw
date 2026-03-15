"""OpenAI GPT provider adapter with OAuth and Responses API support."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from whaleclaw.providers.base import AgentResponse, Message, ToolCall, ToolSchema, repair_tool_call_pairs
from whaleclaw.providers.openai_compat import OpenAICompatProvider
from whaleclaw.types import ProviderAuthError, ProviderError, ProviderRateLimitError, StreamCallback
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_OAUTH_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def _is_responses_model(model: str) -> bool:
    """Codex models require the Responses API instead of Chat Completions."""
    return "codex" in model


class OpenAIProvider(OpenAICompatProvider):
    """OpenAI provider supporting Chat Completions and Responses API.

    Codex models (gpt-5.2-codex, gpt-5.3-codex) are routed to the
    ``/responses`` endpoint automatically.  All other models use the
    standard ``/chat/completions`` endpoint.

    Supports both API key and OAuth (ChatGPT account) authentication.
    """

    provider_name = "openai"
    default_base_url = "https://api.openai.com/v1"
    env_key = "OPENAI_API_KEY"
    supports_cache_control = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        timeout: int = 120,
        auth_mode: str = "api_key",
        oauth_access: str | None = None,
        oauth_refresh: str | None = None,
        oauth_expires: int = 0,
        oauth_account_id: str | None = None,
    ) -> None:
        self._auth_mode = auth_mode
        self._oauth_access = oauth_access or ""
        self._oauth_refresh = oauth_refresh or ""
        self._oauth_expires = oauth_expires
        self._oauth_account_id = oauth_account_id or ""

        if auth_mode == "oauth" and self._oauth_access:
            self._api_key = self._oauth_access
            self._base_url = (base_url or self.default_base_url).rstrip("/")
            self._timeout = timeout
        else:
            super().__init__(api_key=api_key, base_url=base_url, timeout=timeout)

    # ── OAuth token management ──────────────────────────────────────

    def _ensure_oauth_token(self) -> None:
        """Refresh the OAuth token if expired."""
        if self._auth_mode != "oauth":
            return
        now = int(time.time())
        if now < self._oauth_expires - 60:
            return

        log.info("openai.oauth_refresh", reason="token expired or expiring")
        try:
            from whaleclaw.utils.openai_oauth import refresh_access_token, save_oauth_to_config
            result = refresh_access_token(self._oauth_refresh)
            self._oauth_access = result.access
            self._oauth_refresh = result.refresh
            self._oauth_expires = result.expires
            self._oauth_account_id = result.account_id
            self._api_key = result.access
            save_oauth_to_config(result)
            log.info("openai.oauth_refreshed", expires=result.expires)
        except Exception as exc:
            log.error("openai.oauth_refresh_failed", error=str(exc))

    def _build_headers(self) -> dict[str, str]:
        headers = super()._build_headers()
        if self._auth_mode == "oauth" and self._oauth_account_id:
            headers["ChatGPT-Account-Id"] = self._oauth_account_id
        return headers

    # ── Responses API helpers ───────────────────────────────────────

    def _build_responses_input(
        self,
        messages: list[Message],
        *,
        include_system: bool = True,
    ) -> list[dict[str, Any]]:
        """Convert internal Message list to Responses API input items."""
        items: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool" and m.tool_call_id:
                items.append({
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content or "",
                })
            elif m.role == "assistant" and m.tool_calls:
                if m.content:
                    items.append({
                        "role": "assistant",
                        "type": "message",
                        "content": [{"type": "output_text", "text": m.content}],
                    })
                for tc in m.tool_calls:
                    if not tc.name.strip():
                        continue
                    call_item: dict[str, Any] = {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                    # chatgpt codex backend rejects `id=call_*`; call_id is enough for replay.
                    if self._auth_mode != "oauth":
                        call_item["id"] = tc.id
                    items.append(call_item)
            else:
                if not include_system and m.role in ("system", "developer"):
                    continue
                role = m.role if m.role in ("user", "developer", "system") else "user"
                content_parts: list[dict[str, Any]] = []
                if m.images:
                    for img in m.images:
                        content_parts.append({
                            "type": "input_image",
                            "image_url": f"data:{img.mime};base64,{img.data}",
                        })
                content_parts.append({"type": "input_text", "text": m.content or ""})
                items.append({
                    "role": role,
                    "type": "message",
                    "content": content_parts,
                })
        return items

    @staticmethod
    def _repair_responses_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Responses API dict 层面的二次保险：移除孤儿 function_call / function_call_output。"""
        call_ids: set[str] = set()
        output_ids: set[str] = set()
        for item in items:
            itype = item.get("type", "")
            if itype == "function_call":
                cid = item.get("call_id", "")
                if cid:
                    call_ids.add(cid)
            elif itype == "function_call_output":
                cid = item.get("call_id", "")
                if cid:
                    output_ids.add(cid)

        orphan_calls = call_ids - output_ids
        orphan_outputs = output_ids - call_ids
        if not orphan_calls and not orphan_outputs:
            return items

        log.warning(
            "openai.repair_responses_items",
            orphan_calls=len(orphan_calls),
            orphan_outputs=len(orphan_outputs),
        )
        return [
            item for item in items
            if not (
                (item.get("type") == "function_call" and item.get("call_id", "") in orphan_calls)
                or (item.get("type") == "function_call_output" and item.get("call_id", "") in orphan_outputs)
            )
        ]

    def _build_responses_body(
        self,
        messages: list[Message],
        model: str,
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any]:
        messages = repair_tool_call_pairs(messages)
        include_system = self._auth_mode != "oauth"
        input_items = self._build_responses_input(messages, include_system=include_system)
        input_items = self._repair_responses_items(input_items)

        # OAuth 模式下 system/developer 消息被移到 instructions 字段，
        # 如果 context window 裁剪过于激进导致 input 为空，
        # 至少保留最后一条 user 消息，否则 Responses API 会报 400。
        if not input_items:
            for m in reversed(messages):
                if m.role == "user":
                    input_items = self._build_responses_input([m], include_system=False)
                    break
            if not input_items:
                input_items = [{"role": "user", "type": "message",
                                "content": [{"type": "input_text", "text": "..."}]}]

        body: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "stream": True,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
                for t in tools
            ]
        if self._auth_mode == "oauth":
            # Codex OAuth backend requires store=false.
            body["store"] = False
            instructions = "\n\n".join(
                (m.content or "").strip()
                for m in messages
                if m.role in ("system", "developer") and (m.content or "").strip()
            )
            # chatgpt.com codex backend requires top-level instructions.
            body["instructions"] = instructions or "You are a helpful assistant."
        return body

    def _responses_urls(self) -> list[str]:
        if self._auth_mode == "oauth":
            return [_OAUTH_RESPONSES_URL]
        return [f"{self._base_url}/responses"]

    async def _chat_responses_once(
        self,
        url: str,
        body: dict[str, Any],
        headers: dict[str, str],
        *,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        collected: list[str] = []
        input_tokens = 0
        output_tokens = 0
        stop_reason: str | None = None
        pending_calls: dict[str, dict[str, Any]] = {}

        async with (
            httpx.AsyncClient(timeout=self._timeout) as client,
            client.stream("POST", url, json=body, headers=headers) as resp,
        ):
            if resp.status_code == 401:
                raise ProviderAuthError(f"{self.provider_name} API Key / OAuth 无效")
            if resp.status_code == 429:
                raise ProviderRateLimitError(f"{self.provider_name} API 速率限制")
            if resp.status_code != 200:
                error_body = await resp.aread()
                raise ProviderError(
                    f"{self.provider_name} Responses API error {resp.status_code}: "
                    f"{error_body.decode(errors='replace')}"
                )

            async for line in resp.aiter_lines():
                if not line.startswith("event:") and not line.startswith("data:"):
                    continue

                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type", "")

                    if etype == "response.output_text.delta":
                        text = event.get("delta", "")
                        if text:
                            collected.append(text)
                            if on_stream:
                                await on_stream(text)

                    elif etype == "response.function_call_arguments.delta":
                        cid = event.get("call_id", "") or event.get("item_id", "")
                        if not cid:
                            continue
                        if cid not in pending_calls:
                            pending_calls[cid] = {"id": cid, "name": "", "arguments": ""}
                        pending_calls[cid]["arguments"] += event.get("delta", "")

                    elif etype == "response.function_call_arguments.done":
                        cid = event.get("call_id", "") or event.get("item_id", "")
                        if not cid:
                            continue
                        if cid not in pending_calls:
                            pending_calls[cid] = {"id": cid, "name": "", "arguments": ""}
                        done_args = event.get("arguments", "")
                        if isinstance(done_args, str) and done_args:
                            pending_calls[cid]["arguments"] = done_args

                    elif etype == "response.output_item.added":
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            cid = item.get("call_id", "") or item.get("id", "")
                            if not cid:
                                continue
                            pending_calls[cid] = {
                                "id": cid,
                                "name": item.get("name", ""),
                                "arguments": "",
                            }

                    elif etype in ("response.output_item.done", "response.output_item.completed"):
                        item = event.get("item", {})
                        if item.get("type") == "function_call":
                            cid = item.get("call_id", "") or item.get("id", "")
                            if not cid:
                                continue
                            if cid not in pending_calls:
                                pending_calls[cid] = {"id": cid, "name": "", "arguments": ""}
                            if item.get("name"):
                                pending_calls[cid]["name"] = item.get("name", "")
                            item_args = item.get("arguments", "")
                            if isinstance(item_args, str) and item_args:
                                pending_calls[cid]["arguments"] = item_args

                    elif etype == "response.completed":
                        r = event.get("response", {})
                        usage = r.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        stop_reason = r.get("status", "completed")

        full_text = "".join(collected)
        tool_calls: list[ToolCall] = []
        dropped_invalid = 0
        for acc in pending_calls.values():
            name = str(acc.get("name", "")).strip()
            if not name:
                dropped_invalid += 1
                continue
            try:
                args = json.loads(acc["arguments"]) if acc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=acc["id"], name=name, arguments=args))

        log.debug(
            "openai.responses_api",
            model=body.get("model"),
            url=url,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=len(tool_calls),
            dropped_invalid_tool_calls=dropped_invalid,
        )
        return AgentResponse(
            content=full_text,
            model=str(body.get("model", "")),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
        )

    async def _chat_responses(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        """Call the Responses API (``/responses``) for codex models."""
        body = self._build_responses_body(messages, model, tools)
        headers = self._build_headers()
        urls = self._responses_urls()
        last_auth_error: ProviderAuthError | None = None
        for idx, url in enumerate(urls):
            try:
                return await self._chat_responses_once(url, body, headers, on_stream=on_stream)
            except ProviderAuthError as exc:
                last_auth_error = exc
                if idx < len(urls) - 1:
                    log.warning(
                        "openai.responses_auth_retry",
                        from_url=url,
                        to_url=urls[idx + 1],
                    )
                    continue
                raise
        raise last_auth_error or ProviderAuthError(f"{self.provider_name} API Key / OAuth 无效")

    # ── Main entry point ────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        model: str,
        *,
        tools: list[ToolSchema] | None = None,
        on_stream: StreamCallback | None = None,
    ) -> AgentResponse:
        self._ensure_oauth_token()
        if self._auth_mode == "oauth" and not model.lower().startswith("gpt-5"):
            raise ProviderError(
                "OpenAI OAuth 模式仅支持 GPT-5 系列模型"
            )
        # For ChatGPT OAuth, GPT-5 family is served via Responses API.
        use_responses = _is_responses_model(model) or (
            self._auth_mode == "oauth" and model.lower().startswith("gpt-5")
        )
        if use_responses:
            return await self._chat_responses(
                messages, model, tools=tools, on_stream=on_stream,
            )
        return await super().chat(
            messages, model, tools=tools, on_stream=on_stream,
        )
