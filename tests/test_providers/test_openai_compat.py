"""Tests for the OpenAI-compatible provider base (mocked HTTP)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from whaleclaw.providers.base import ImageContent, Message, ToolCall
from whaleclaw.providers.openai import OpenAIProvider
from whaleclaw.providers.qwen import QwenProvider
from whaleclaw.types import ProviderAuthError, ProviderError


class _FakeResponse:
    def __init__(self, status_code: int, lines: list[str]) -> None:
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b"error"

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *a: object) -> None:
        pass


class _FakeClient:
    def __init__(self, resp: _FakeResponse) -> None:
        self._resp = resp

    def stream(self, *a: object, **kw: object) -> _FakeResponse:
        return self._resp

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *a: object) -> None:
        pass


class _RouteClient:
    def __init__(self, routes: dict[str, _FakeResponse]) -> None:
        self._routes = routes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def stream(self, method: str, url: str, **kw: object) -> _FakeResponse:
        self.calls.append((url, kw))
        return self._routes[url]

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *a: object) -> None:
        pass


class _FlakyClient:
    def __init__(self, failures: int, resp: _FakeResponse) -> None:
        self._failures = failures
        self._resp = resp
        self.calls = 0

    def stream(self, *a: object, **kw: object) -> _FakeResponse:
        self.calls += 1
        if self.calls <= self._failures:
            raise httpx.ConnectError("boom")
        return self._resp

    async def __aenter__(self):  # noqa: ANN204
        return self

    async def __aexit__(self, *a: object) -> None:
        pass


@pytest.fixture()
def provider() -> OpenAIProvider:
    return OpenAIProvider(api_key="test-key")


@pytest.mark.asyncio
async def test_streaming(provider: OpenAIProvider) -> None:
    lines = [
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "Hi"}, "finish_reason": None}],
        }),
        "data: " + json.dumps({
            "choices": [{"delta": {"content": " there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }),
        "data: [DONE]",
    ]
    fake = _FakeClient(_FakeResponse(200, lines))
    chunks: list[str] = []

    async def on_stream(c: str) -> None:
        chunks.append(c)

    with patch("whaleclaw.providers.openai_compat.httpx.AsyncClient", return_value=fake):
        result = await provider.chat(
            [Message(role="user", content="hi")],
            "gpt-5.2",
            on_stream=on_stream,
        )

    assert result.content == "Hi there"
    assert chunks == ["Hi", " there"]
    assert result.input_tokens == 5
    assert result.output_tokens == 3


@pytest.mark.asyncio
async def test_auth_error(provider: OpenAIProvider) -> None:
    fake = _FakeClient(_FakeResponse(401, []))
    with (
        patch("whaleclaw.providers.openai_compat.httpx.AsyncClient", return_value=fake),
        pytest.raises(ProviderAuthError),
    ):
        await provider.chat([Message(role="user", content="hi")], "gpt-5.2")


@pytest.mark.asyncio
async def test_network_retry_then_success(provider: OpenAIProvider) -> None:
    lines = [
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }),
        "data: [DONE]",
    ]
    fake = _FlakyClient(failures=1, resp=_FakeResponse(200, lines))
    with patch("whaleclaw.providers.openai_compat.httpx.AsyncClient", return_value=fake):
        result = await provider.chat([Message(role="user", content="hi")], "gpt-5.2")

    assert result.content == "ok"
    assert fake.calls == 2


@pytest.mark.asyncio
async def test_streaming_usage_input_output_fields(provider: OpenAIProvider) -> None:
    lines = [
        "data: " + json.dumps({
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"input_tokens": 7, "output_tokens": 4},
        }),
        "data: [DONE]",
    ]
    fake = _FakeClient(_FakeResponse(200, lines))
    with patch("whaleclaw.providers.openai_compat.httpx.AsyncClient", return_value=fake):
        result = await provider.chat([Message(role="user", content="hi")], "gpt-5.2")

    assert result.input_tokens == 7
    assert result.output_tokens == 4


def test_qwen_build_body_enables_stream_usage() -> None:
    provider = QwenProvider(api_key="test-key")
    body = provider._build_body(
        [Message(role="user", content="hi")],
        "qwen3.5-plus",
        None,
    )
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_openai_compat_build_body_includes_user_images_as_data_urls() -> None:
    provider = OpenAIProvider(api_key="test-key")
    body = provider._build_body(
        [
            Message(
                role="user",
                content="描述这张图",
                images=[ImageContent(mime="image/jpeg", data="ZmFrZS1pbWFnZQ==")],
            )
        ],
        "gpt-4o",
        None,
    )

    user_msg = body["messages"][0]
    assert user_msg["role"] == "user"
    assert user_msg["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,ZmFrZS1pbWFnZQ=="},
    }
    assert user_msg["content"][1] == {"type": "text", "text": "描述这张图"}


def test_openai_responses_body_includes_user_images_as_input_images() -> None:
    provider = OpenAIProvider(api_key="test-key")
    body = provider._build_responses_body(
        [
            Message(
                role="user",
                content="描述这张图",
                images=[ImageContent(mime="image/png", data="cG5nLWJ5dGVz")],
            )
        ],
        "gpt-5.4",
        None,
    )

    user_msg = body["input"][0]
    assert user_msg["role"] == "user"
    assert user_msg["type"] == "message"
    assert user_msg["content"][0] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,cG5nLWJ5dGVz",
    }
    assert user_msg["content"][1] == {"type": "input_text", "text": "描述这张图"}


@pytest.mark.asyncio
async def test_oauth_rejects_non_gpt52_model() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    with pytest.raises(ProviderError, match="仅支持"):
        await provider.chat([Message(role="user", content="hi")], "gpt-4o")


@pytest.mark.asyncio
async def test_oauth_gpt52_uses_chatgpt_codex_endpoint() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    routes = {
        "https://chatgpt.com/backend-api/codex/responses": _FakeResponse(
            200,
            [
                "data: " + json.dumps({
                    "type": "response.output_text.delta",
                    "delta": "hello",
                }),
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                }),
            ],
        ),
    }
    fake = _RouteClient(routes)
    with patch("whaleclaw.providers.openai.httpx.AsyncClient", return_value=fake):
        result = await provider.chat([Message(role="user", content="hi")], "gpt-5.2")

    assert result.content == "hello"
    assert [url for (url, _kw) in fake.calls] == ["https://chatgpt.com/backend-api/codex/responses"]
    body = fake.calls[0][1]["json"]
    assert body["model"] == "gpt-5.2"
    assert body["instructions"] == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_oauth_responses_moves_system_to_instructions() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    routes = {
        "https://chatgpt.com/backend-api/codex/responses": _FakeResponse(
            200,
            [
                "data: " + json.dumps({
                    "type": "response.output_text.delta",
                    "delta": "ok",
                }),
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    },
                }),
            ],
        ),
    }
    fake = _RouteClient(routes)
    with patch("whaleclaw.providers.openai.httpx.AsyncClient", return_value=fake):
        _ = await provider.chat(
            [
                Message(role="system", content="你是一个中文助手"),
                Message(role="user", content="hi"),
            ],
            "gpt-5.2",
        )

    body = fake.calls[0][1]["json"]
    assert body["instructions"] == "你是一个中文助手"
    roles = [item.get("role") for item in body["input"] if isinstance(item, dict)]
    assert "system" not in roles


@pytest.mark.asyncio
async def test_oauth_responses_omits_function_call_id_field() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    routes = {
        "https://chatgpt.com/backend-api/codex/responses": _FakeResponse(
            200,
            [
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }),
            ],
        ),
    }
    fake = _RouteClient(routes)
    with patch("whaleclaw.providers.openai.httpx.AsyncClient", return_value=fake):
        _ = await provider.chat(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call_abc",
                            name="bash",
                            arguments={"command": "echo 1"},
                        )
                    ],
                ),
                Message(role="tool", content="1", tool_call_id="call_abc"),
                Message(role="user", content="继续"),
            ],
            "gpt-5.2",
        )

    body = fake.calls[0][1]["json"]
    fn_calls = [item for item in body["input"] if item.get("type") == "function_call"]
    assert len(fn_calls) == 1
    assert fn_calls[0]["call_id"] == "call_abc"
    assert "id" not in fn_calls[0]


@pytest.mark.asyncio
async def test_oauth_responses_skips_empty_function_name() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    routes = {
        "https://chatgpt.com/backend-api/codex/responses": _FakeResponse(
            200,
            [
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }),
            ],
        ),
    }
    fake = _RouteClient(routes)
    with patch("whaleclaw.providers.openai.httpx.AsyncClient", return_value=fake):
        _ = await provider.chat(
            [
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id="call_empty", name="", arguments={"a": 1})],
                ),
                Message(role="tool", content="x", tool_call_id="call_empty"),
                Message(role="user", content="继续"),
            ],
            "gpt-5.2",
        )

    body = fake.calls[0][1]["json"]
    fn_calls = [item for item in body["input"] if item.get("type") == "function_call"]
    assert fn_calls == []


@pytest.mark.asyncio
async def test_oauth_responses_parses_function_args_from_done_events() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    routes = {
        "https://chatgpt.com/backend-api/codex/responses": _FakeResponse(
            200,
            [
                "data: " + json.dumps({
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "call_id": "call_1", "name": "browser"},
                }),
                "data: " + json.dumps({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "browser",
                        "arguments": "{\"action\":\"search_images\",\"text\":\"杨幂近照\"}",
                    },
                }),
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }),
            ],
        ),
    }
    fake = _RouteClient(routes)
    with patch("whaleclaw.providers.openai.httpx.AsyncClient", return_value=fake):
        result = await provider.chat([Message(role="user", content="给我杨幂近照")], "gpt-5.2")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "browser"
    assert result.tool_calls[0].arguments == {"action": "search_images", "text": "杨幂近照"}


@pytest.mark.asyncio
async def test_oauth_responses_drops_invalid_function_call_events() -> None:
    provider = OpenAIProvider(
        auth_mode="oauth",
        oauth_access="oauth-token",
        oauth_refresh="refresh-token",
        oauth_expires=4_102_444_800,  # 2100-01-01
        oauth_account_id="acct-1",
    )
    routes = {
        "https://chatgpt.com/backend-api/codex/responses": _FakeResponse(
            200,
            [
                # invalid: missing call_id/item_id should be ignored
                "data: " + json.dumps({
                    "type": "response.function_call_arguments.delta",
                    "delta": "{\"action\":\"search_images\"}",
                }),
                # invalid: has id but empty name, should be dropped at build stage
                "data: " + json.dumps({
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "call_id": "call_bad", "name": ""},
                }),
                # valid tool call
                "data: " + json.dumps({
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "call_id": "call_ok", "name": "browser"},
                }),
                "data: " + json.dumps({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_ok",
                        "name": "browser",
                        "arguments": "{\"action\":\"search_images\",\"text\":\"刘亦菲 近照\"}",
                    },
                }),
                "data: " + json.dumps({
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                }),
            ],
        ),
    }
    fake = _RouteClient(routes)
    with patch("whaleclaw.providers.openai.httpx.AsyncClient", return_value=fake):
        result = await provider.chat([Message(role="user", content="给我刘亦菲近照")], "gpt-5.2")

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_ok"
    assert result.tool_calls[0].name == "browser"
    assert result.tool_calls[0].arguments == {"action": "search_images", "text": "刘亦菲 近照"}
