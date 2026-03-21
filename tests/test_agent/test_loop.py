"""Tests for the Agent main loop (mocked provider)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

import whaleclaw.agent.loop as loop_mod  # pyright: ignore[reportUnusedImport]  # 保留供 patch 使用
from tests.test_agent.loop_helpers import (
    BashProbeTool,
    BrowserProbeTool,
    DesktopCaptureNoopTool,
    EchoTool,
    NameMemoryManager,
    make_router,
)
from whaleclaw.agent.helpers.tool_execution import (
    is_nano_banana_cli_param_error,
    is_nano_banana_preflight_error,
    is_transient_cli_usage_error,
    parse_fallback_tool_calls,
    repair_tool_call,
)
from whaleclaw.agent.helpers.tool_guards import (
    ToolGuardState,
    apply_tool_result_guards,
)

def _nb_hooks() -> list[object]:
    import importlib.util, sys
    from pathlib import Path
    from whaleclaw.config.paths import WORKSPACE_DIR
    candidates = [
        WORKSPACE_DIR / "skills" / "nano-banana-image-t8" / "hooks.py",
        Path("whaleclaw/skills/bundled/nano-banana-image-t8/hooks.py"),
    ]
    hooks_path = next((p for p in candidates if p.is_file()), None)
    if hooks_path is None:
        raise FileNotFoundError("nano-banana hooks.py not found")
    key = "nb_hooks_loop_test"
    if key in sys.modules:
        return [sys.modules[key].Hooks()]
    spec = importlib.util.spec_from_file_location(key, str(hooks_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return [mod.Hooks()]
from whaleclaw.agent.helpers.office_rules import is_image_generation_request
from whaleclaw.agent.helpers.skill_helpers import select_native_tool_names
from whaleclaw.agent.loop import run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, ImageContent, Message, ToolCall
from whaleclaw.sessions.manager import Session
from whaleclaw.tools.base import ToolResult
from whaleclaw.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_run_agent_returns_reply() -> None:
    mock_response = AgentResponse(
        content="你好！我是 WhaleClaw。",
        model="claude-sonnet-4-20250514",
        input_tokens=50,
        output_tokens=20,
    )

    router = make_router(response=mock_response)

    result = await run_agent(
        message="你好",
        session_id="test-session",
        config=WhaleclawConfig(),
        router=router,
    )

    assert result == "你好！我是 WhaleClaw。"
    router.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_agent_reuses_persisted_current_user_message_with_images() -> None:
    now = datetime.now(UTC)
    session = Session(
        id="test-session-images",
        channel="feishu",
        peer_id="u1",
        messages=[Message(role="user", content="详细描述这张图")],
        model="openai/gpt-5.4",
        created_at=now,
        updated_at=now,
        metadata={},
    )
    image = ImageContent(mime="image/jpeg", data="ZmFrZS1pbWFnZQ==")

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        user_messages = [m for m in messages if isinstance(m, Message) and m.role == "user"]
        assert len(user_messages) == 1
        assert user_messages[0].content == "详细描述这张图"
        assert user_messages[0].images is not None
        assert len(user_messages[0].images) == 1
        assert user_messages[0].images[0].mime == "image/jpeg"
        return AgentResponse(content="已看到图片。", model="test-model")

    router = make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="详细描述这张图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
        images=[image],
    )

    assert result == "已看到图片。"


def test_select_native_tool_names_prefers_desktop_capture_for_desktop_screenshot() -> None:
    registry = ToolRegistry()
    registry.register(BashProbeTool())
    registry.register(DesktopCaptureNoopTool())

    selected = select_native_tool_names(registry, "把桌面截图一下")

    assert "desktop_capture" in selected


@pytest.mark.asyncio
async def test_run_agent_retries_once_on_empty_reply_then_recovers() -> None:
    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AgentResponse(content="", model="test-model", input_tokens=0, output_tokens=0)
        return AgentResponse(content="请告诉我你要我做什么。", model="test-model")

    router = make_router(chat_fn=fake_chat)
    result = await run_agent(
        message="？？？",
        session_id="test-empty-retry",
        config=WhaleclawConfig(),
        router=router,
    )
    assert result == "请告诉我你要我做什么。"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_returns_fallback_after_two_empty_replies() -> None:
    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        return AgentResponse(content="", model="test-model", input_tokens=0, output_tokens=0)

    router = make_router(chat_fn=fake_chat)
    result = await run_agent(
        message="？？？",
        session_id="test-empty-fallback",
        config=WhaleclawConfig(),
        router=router,
    )
    assert result == "我这边没收到模型有效回复。请再发一次需求，我会继续处理。"


@pytest.mark.asyncio
async def test_run_agent_streams() -> None:
    mock_response = AgentResponse(
        content="Hello world",
        model="claude-sonnet-4-20250514",
        input_tokens=10,
        output_tokens=5,
    )

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,
    ) -> AgentResponse:
        if on_stream:
            await on_stream("Hello ")
            await on_stream("world")
        return mock_response

    router = make_router(chat_fn=fake_chat)

    chunks: list[str] = []

    async def collect(chunk: str) -> None:
        chunks.append(chunk)

    result = await run_agent(
        message="hi",
        session_id="test-session",
        config=WhaleclawConfig(),
        on_stream=collect,
        router=router,
    )

    assert result == "Hello world"
    assert chunks == ["Hello ", "world"]


def test_is_transient_cli_usage_error_detects_argparse_banner() -> None:
    result = ToolResult(
        success=False,
        output="[stderr]\nusage: test_nano_banana_2.py [-h]\nerror: unrecognized arguments: --bad",
        error="usage: test_nano_banana_2.py [-h]\nerror: unrecognized arguments: --bad",
    )

    assert is_transient_cli_usage_error(result) is True


def test_is_nano_banana_cli_param_error_requires_script_banner_and_argparse_marker() -> None:
    result = ToolResult(
        success=False,
        output=(
            "[stderr]\nHTTP 500 请求失败: https://ai.t8star.cn/v1/images/generations\n"
            '响应体: {"error":"upstream error"}'
        ),
        error="HTTP 500 请求失败",
    )

    assert is_nano_banana_cli_param_error(result) is False


def test_is_nano_banana_preflight_error_detects_python_script_run_as_shell() -> None:
    result = ToolResult(
        success=False,
        output=(
            "[stderr]\n"
            "/tmp/test_nano_banana_2.py: line 1: from: command not found\n"
            "/tmp/test_nano_banana_2.py: line 3: import: command not found"
        ),
        error=(
            "/tmp/test_nano_banana_2.py: line 1: from: command not found\n"
            "/tmp/test_nano_banana_2.py: line 3: import: command not found"
        ),
    )

    assert is_nano_banana_preflight_error(result) is True


def test_repair_tool_call_normalizes_nano_banana_cli_args() -> None:
    tc = ToolCall(
        id="tc_nano",
        name="bash",
        arguments={
            "command": (
                'bash -lc \'./python/bin/python3.12 '
                '~/.whaleclaw/workspace/skills/nano-banana-image-t8/scripts/test_nano_banana_2.py '
                '--mode text2image --api-base https://ai.t8star.cn '
                '--prompt "刘亦菲大战奥特曼" --size "4:3"\''
            )
        },
    )

    repaired, reason = repair_tool_call(tc, "使用香蕉生图画两张图，比例 4:3")

    assert reason is not None
    command = str(repaired.arguments["command"])
    assert "--mode text" in command
    assert "--base-url https://ai.t8star.cn" in command
    assert '--aspect-ratio "4:3"' in command
    assert "--mode text2image" not in command
    assert "--api-base" not in command


def test_nano_banana_cli_usage_error_allows_fixed_retry() -> None:
    state = ToolGuardState()
    tc = ToolCall(
        id="tc_nano",
        name="bash",
        arguments={"command": "./python/bin/python3.12 test_nano_banana_2.py --mode text2image"},
    )
    result = ToolResult(
        success=False,
        output=(
            "[stderr]\nusage: test_nano_banana_2.py [-h]\n"
            "error: argument --mode: invalid choice: 'text2image'"
        ),
        error=(
            "usage: test_nano_banana_2.py [-h]\n"
            "error: argument --mode: invalid choice: 'text2image'"
        ),
    )

    update = apply_tool_result_guards(
        state,
        tc,
        result,
        session_id="test",
        skill_hooks=_nb_hooks(),
    )

    assert update.conversation_messages == [
        "[系统提示] 上一次 Nano Banana 命令在进入实际生图流程前就失败了。"
        "如果是参数错误或脚本调用方式错误，允许你立刻修正后重试一次；"
        "但必须修改错误点，禁止原样重试。"
    ]


def test_nano_banana_runtime_failure_still_blocks_auto_retry() -> None:
    state = ToolGuardState()
    tc = ToolCall(
        id="tc_nano",
        name="bash",
        arguments={"command": "./python/bin/python3.12 test_nano_banana_2.py --mode text"},
    )
    result = ToolResult(
        success=False,
        output="[ERROR] HTTP 500 请求失败: https://ai.t8star.cn/v1/images/generations",
        error="HTTP 500 请求失败",
    )

    update = apply_tool_result_guards(
        state,
        tc,
        result,
        session_id="test",
        skill_hooks=_nb_hooks(),
    )

    assert update.conversation_messages == [
        "[系统提示] 本次生图失败，禁止自动重试。"
        "继续执行剩余任务，最终将成功和失败的结果一并回复用户，"
        "由用户决定是否对失败的图重新操作。"
    ]


def test_nano_banana_python_shell_mismatch_allows_fixed_retry() -> None:
    state = ToolGuardState()
    tc = ToolCall(
        id="tc_nano",
        name="bash",
        arguments={"command": "~/.whaleclaw/workspace/skills/nano-banana-image-t8/scripts/test_nano_banana_2.py"},
    )
    result = ToolResult(
        success=False,
        output=(
            "[stderr]\n"
            "/tmp/test_nano_banana_2.py: line 1: from: command not found\n"
            "/tmp/test_nano_banana_2.py: line 3: import: command not found"
        ),
        error=(
            "/tmp/test_nano_banana_2.py: line 1: from: command not found\n"
            "/tmp/test_nano_banana_2.py: line 3: import: command not found"
        ),
    )

    update = apply_tool_result_guards(
        state,
        tc,
        result,
        session_id="test",
        skill_hooks=_nb_hooks(),
    )

    assert update.conversation_messages == [
        "[系统提示] 上一次 Nano Banana 命令在进入实际生图流程前就失败了。"
        "如果是参数错误或脚本调用方式错误，允许你立刻修正后重试一次；"
        "但必须修改错误点，禁止原样重试。"
    ]


@pytest.mark.asyncio
async def test_run_agent_tool_call_loop() -> None:
    """Agent should execute tools and loop back to LLM."""
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(id="tc_1", name="echo", arguments={"text": "hello"})
        ],
    )
    final_response = AgentResponse(
        content="Echo result: hello",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = make_router(chat_fn=fake_chat)

    registry = ToolRegistry()
    registry.register(EchoTool())

    tool_calls_seen: list[str] = []
    tool_results_seen: list[bool] = []

    async def on_tc(name: str, _args: dict[str, Any]) -> None:
        tool_calls_seen.append(name)

    async def on_tr(name: str, result: ToolResult) -> None:
        tool_results_seen.append(result.success)

    result = await run_agent(
        message="echo hello",
        session_id="test-tool",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        on_tool_call=on_tc,
        on_tool_result=on_tr,
    )

    assert result == "Echo result: hello"
    assert call_count == 2
    assert tool_calls_seen == ["echo"]
    assert tool_results_seen == [True]


@pytest.mark.asyncio
async def test_run_agent_updates_assistant_name_from_user_message() -> None:
    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        system_text = messages[0].content if messages else ""
        return AgentResponse(content=system_text, model="test-model")

    cfg = WhaleclawConfig()
    cfg.agent.memory.enabled = False
    mm = NameMemoryManager()
    router = make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="以后你叫旺财",
        session_id="test-rename",
        config=cfg,
        router=router,
        memory_manager=mm,  # type: ignore[arg-type]
    )

    assert "你是 旺财" in result
    assert mm.name == "旺财"
    assert mm.set_calls == 1


@pytest.mark.asyncio
async def test_run_agent_does_not_rename_on_plain_name_question() -> None:
    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        system_text = messages[0].content if messages else ""
        return AgentResponse(content=system_text, model="test-model")

    cfg = WhaleclawConfig()
    cfg.agent.memory.enabled = False
    mm = NameMemoryManager("WhaleClaw")
    router = make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="你叫什么名字？",
        session_id="test-no-rename",
        config=cfg,
        router=router,
        memory_manager=mm,  # type: ignore[arg-type]
    )

    assert "你是 WhaleClaw" in result
    assert mm.set_calls == 0


@pytest.mark.asyncio
async def test_run_agent_unknown_tool() -> None:
    """Unknown tool should not crash, returns error to LLM."""
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(id="tc_bad", name="nonexistent", arguments={})
        ],
    )
    final_response = AgentResponse(
        content="I could not find that tool.",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()

    result = await run_agent(
        message="do something",
        session_id="test-unknown",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "I could not find that tool."
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_fallback_mode() -> None:
    """Provider without native tools: parse JSON from text output."""
    json_text = (
        '我来查一下。\n'
        '```json\n'
        '{"tool": "echo", "arguments": {"text": "hello"}}\n'
        '```'
    )
    tool_response = AgentResponse(
        content=json_text,
        model="test-model",
    )
    final_response = AgentResponse(
        content="查到了: hello",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    router = make_router(chat_fn=fake_chat, native_tools=False)

    registry = ToolRegistry()
    registry.register(EchoTool())

    result = await run_agent(
        message="echo hello",
        session_id="test-fallback",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "查到了: hello"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_retries_when_tool_args_invalid_then_succeeds() -> None:
    invalid_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[ToolCall(id="tc_browser", name="browser", arguments={})],
    )
    valid_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_browser_2",
                name="browser",
                arguments={"action": "search_images", "text": "杨幂近照"},
            )
        ],
    )
    final_response = AgentResponse(
        content="已完成",
        model="test-model",
    )

    call_count = 0

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return invalid_tool_response
        if call_count == 2:
            return valid_tool_response
        return final_response

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(BrowserProbeTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-repair",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "已完成"
    assert call_count == 3


def test_is_image_generation_request_matches_expected_queries() -> None:
    assert is_image_generation_request("请帮我文生图，主题是赛博朋克街景") is True
    assert is_image_generation_request("这张图做图生图，风格改成宫崎骏") is True
    assert is_image_generation_request("帮我改这个 ppt 第三页文案") is False
    assert is_image_generation_request("帮我测试一下 API key 是否可用") is False


class TestParseFallbackToolCalls:
    def test_fenced_json(self) -> None:
        text = '```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```'
        calls = parse_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls"}

    def test_bare_json(self) -> None:
        text = '好的，我来执行 {"tool": "bash", "arguments": {"command": "pwd"}}'
        calls = parse_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_no_tool(self) -> None:
        text = "这是普通文本，没有工具调用。"
        calls = parse_fallback_tool_calls(text)
        assert calls == []
