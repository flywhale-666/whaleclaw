"""Tests for the Agent main loop (mocked provider)."""

from __future__ import annotations

import base64
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

import whaleclaw.agent.loop as loop_mod
from whaleclaw.agent.helpers.tool_execution import (
    is_nano_banana_cli_param_error,
    is_nano_banana_preflight_error,
    is_transient_cli_usage_error,
    repair_tool_call,
)
from whaleclaw.agent.helpers.tool_guards import (
    ToolGuardState,
    apply_tool_result_guards,
)
from whaleclaw.agent.loop import _is_image_generation_request, _parse_fallback_tool_calls, run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, Message, ToolCall
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.skills.parser import Skill, SkillParamGuard, SkillParamItem
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.tools.registry import ToolRegistry


def _make_router(
    chat_fn: Any = None,
    response: AgentResponse | None = None,
    native_tools: bool = True,
) -> MagicMock:
    """Build a mock ModelRouter with proper sync/async methods."""
    router = MagicMock()
    router.supports_native_tools = MagicMock(return_value=native_tools)
    if chat_fn is not None:
        router.chat = chat_fn
    elif response is not None:
        router.chat = AsyncMock(return_value=response)
    return router


@pytest.mark.asyncio
async def test_run_agent_returns_reply() -> None:
    mock_response = AgentResponse(
        content="你好！我是 WhaleClaw。",
        model="claude-sonnet-4-20250514",
        input_tokens=50,
        output_tokens=20,
    )

    router = _make_router(response=mock_response)

    result = await run_agent(
        message="你好",
        session_id="test-session",
        config=WhaleclawConfig(),
        router=router,
    )

    assert result == "你好！我是 WhaleClaw。"
    router.chat.assert_awaited_once()


def test_select_native_tool_names_prefers_desktop_capture_for_desktop_screenshot() -> None:
    registry = ToolRegistry()
    registry.register(_BashProbeTool())
    registry.register(_DesktopCaptureNoopTool())

    selected = loop_mod.select_native_tool_names(registry, "把桌面截图一下")

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

    router = _make_router(chat_fn=fake_chat)
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

    router = _make_router(chat_fn=fake_chat)
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

    router = _make_router(chat_fn=fake_chat)

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
        office_loop_guard_enabled=False,
        image_api_probe_guard_enabled=False,
        session_id="test",
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
        office_loop_guard_enabled=False,
        image_api_probe_guard_enabled=False,
        session_id="test",
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
        office_loop_guard_enabled=False,
        image_api_probe_guard_enabled=False,
        session_id="test",
    )

    assert update.conversation_messages == [
        "[系统提示] 上一次 Nano Banana 命令在进入实际生图流程前就失败了。"
        "如果是参数错误或脚本调用方式错误，允许你立刻修正后重试一次；"
        "但必须修改错误点，禁止原样重试。"
    ]


class _EchoTool(Tool):
    """Dummy tool that echoes its input."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="echo",
            description="Echo text back.",
            parameters=[
                ToolParameter(
                    name="text", type="string", description="Text to echo."
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=kwargs.get("text", ""))


class _LoopTool(Tool):
    """Dummy tool used to simulate repeated successful loops."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="loop_tool",
            description="Repeatable tool for loop tests.",
            parameters=[
                ToolParameter(
                    name="text", type="string", description="Loop payload."
                )
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=str(kwargs.get("text", "")))


class _BrowserProbeTool(Tool):
    """Dummy browser tool to assert required browser arguments."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser",
            description="Probe browser arguments.",
            parameters=[
                ToolParameter(name="action", type="string", description="action"),
                ToolParameter(name="text", type="string", description="text"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        action = kwargs.get("action", "")
        text = kwargs.get("text", "")
        if action == "search_images" and bool(text):
            return ToolResult(success=True, output=f"ok:{text}")
        return ToolResult(success=False, output="", error="bad args")


class _BrowserAlwaysFailTool(Tool):
    """Dummy browser tool that always fails."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="browser",
            description="Always fails.",
            parameters=[
                ToolParameter(name="action", type="string", description="action"),
                ToolParameter(name="text", type="string", description="text"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=False, output="", error="browser failed")


class _BashProbeTool(Tool):
    """Dummy bash tool to assert command arg exists."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Probe bash arguments.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "")).strip()
        if command:
            return ToolResult(success=True, output=f"ok:{command}")
        return ToolResult(success=False, output="", error="bad command")


class _DelayedNanoBananaBashTool(Tool):
    """Dummy bash tool that simulates slow nano-banana commands."""

    def __init__(self, delay: float = 0.08) -> None:
        self.delay = delay
        self.started_at: list[float] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Delayed bash for parallel nano banana tests.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", ""))
        self.started_at.append(time.monotonic())
        await asyncio.sleep(self.delay)
        return ToolResult(success=True, output=f"ok:{command}")


class _DesktopCaptureNoopTool(Tool):
    """Dummy desktop_capture tool for tool-selection assertions."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="desktop_capture",
            description="Capture macOS desktop screenshot.",
            parameters=[
                ToolParameter(name="wake", type="boolean", description="wake display"),
                ToolParameter(name="delay_ms", type="integer", description="delay"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=True, output="desktop captured")


class _BashAlwaysFailTool(Tool):
    """Dummy bash tool that always fails."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Always fails.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=False, output="", error="bash failed")


class _BashPyScriptRetryTool(Tool):
    """Dummy bash tool that fails on direct .py invocation then succeeds once rewritten."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Fails first for direct python script invocation.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "")).strip()
        self.commands.append(command)
        if command.startswith("/tmp/test_nano_banana_2.py "):
            return ToolResult(
                success=False,
                output="[stderr]\nfrom: command not found\nimport: command not found\n[exit_code: 127]",
                error="from: command not found\nimport: command not found",
            )
        if "python3.12 /tmp/test_nano_banana_2.py --mode edit" in command:
            return ToolResult(success=True, output="ok")
        return ToolResult(success=False, output="", error=f"unexpected command: {command}")


class _NanoBananaFixedRunnerTool(Tool):
    """Dummy bash tool that simulates fixed-template nano-banana execution."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.commands: list[str] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Executes fixed nano-banana command.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "")).strip()
        self.commands.append(command)
        return ToolResult(
            success=True,
            output=(
                "当前使用模型: 香蕉2\n"
                "[图生图] 测试中...\n"
                f"图生图成功: {self.output_path}\n"
                "任务完成\n"
                "\n[exit_code: 0]"
            ),
        )


import re as _re


def _make_nano_banana_router(
    *,
    model_display: str = "香蕉2",
    output_path: Path | None = None,
    final_content: str | None = None,
) -> MagicMock:
    """构建 nano-banana 测试用的 router mock。

    第一次 chat 调用：从 system messages 中提取命令模板，返回 bash tool_call。
    第二次 chat 调用：返回格式化的成功回复。
    """
    _CMD_RE = _re.compile(r"```\n(.+?)\n```", _re.DOTALL)
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
            cmd = ""
            for m in messages:
                text = m.content if hasattr(m, "content") else str(m)
                match = _CMD_RE.search(text)
                if match:
                    cmd = match.group(1).strip()
                    break
            return AgentResponse(
                content="",
                model="test-model",
                tool_calls=[ToolCall(id="tc_nb_1", name="bash", arguments={"command": cmd})],
            )
        content = final_content
        if content is None:
            parts = [f"当前使用模型：{model_display}", "改好了："]
            if output_path is not None:
                parts.append(f"![结果图]({output_path})")
                parts.append(f"文件路径：{output_path}")
            content = "\n".join(parts)
        return AgentResponse(content=content, model="test-model")

    return _make_router(chat_fn=fake_chat)


class _PptEditNoopTool(Tool):
    """Dummy ppt_edit tool used for tool-selection assertions."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ppt_edit",
            description="noop ppt edit.",
            parameters=[
                ToolParameter(name="path", type="string", description="path"),
                ToolParameter(name="slide_index", type="integer", description="index"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(success=True, output="ok")


class _PptEditBusinessNoHitTool(Tool):
    """Dummy ppt_edit business style tool that reports zero hit."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="ppt_edit",
            description="business no hit",
            parameters=[
                ToolParameter(name="path", type="string", description="path"),
                ToolParameter(name="slide_index", type="integer", description="index"),
                ToolParameter(name="action", type="string", description="action"),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
        return ToolResult(
            success=True,
            output="已应用 /tmp/a.pptx 第 1 页商务风格，重设深色条 0 处",
        )


class _NameMemoryManager:
    def __init__(self, name: str = "") -> None:
        self.name = name
        self.set_calls = 0
        self.clear_calls = 0

    async def get_assistant_name(self) -> str:
        return self.name

    async def set_assistant_name(self, name: str, *, source: str = "manual") -> bool:  # noqa: ARG002
        self.name = name
        self.set_calls += 1
        return True

    async def clear_assistant_name(self) -> int:
        old = 1 if self.name else 0
        self.name = ""
        self.clear_calls += 1
        return old


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

    router = _make_router(chat_fn=fake_chat)

    registry = ToolRegistry()
    registry.register(_EchoTool())

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
    mm = _NameMemoryManager()
    router = _make_router(chat_fn=fake_chat)

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
    mm = _NameMemoryManager("WhaleClaw")
    router = _make_router(chat_fn=fake_chat)

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

    router = _make_router(chat_fn=fake_chat)
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

    router = _make_router(chat_fn=fake_chat, native_tools=False)

    registry = ToolRegistry()
    registry.register(_EchoTool())

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

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserProbeTool())

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
    assert _is_image_generation_request("请帮我文生图，主题是赛博朋克街景") is True
    assert _is_image_generation_request("这张图做图生图，风格改成宫崎骏") is True
    assert _is_image_generation_request("帮我改这个 ppt 第三页文案") is False
    assert _is_image_generation_request("帮我测试一下 API key 是否可用") is False


@pytest.mark.asyncio
async def test_run_agent_circuit_breaker_blocks_repeated_browser_failures() -> None:
    browser_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_browser",
                name="browser",
                arguments={"action": "search_images", "text": "杨幂近照"},
            )
        ],
    )
    final_response = AgentResponse(
        content="改用 bash 处理",
        model="test-model",
    )

    call_count = 0
    prompts_seen: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        prompts_seen.append("\n".join(m.content for m in messages if hasattr(m, "content")))
        if call_count <= 2:
            return browser_tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserAlwaysFailTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-circuit",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "改用 bash 处理"
    assert call_count == 3
    assert any("browser 工具连续失败，已自动熔断" in p for p in prompts_seen)


@pytest.mark.asyncio
async def test_run_agent_circuit_breaker_blocks_repeated_bash_failures() -> None:
    bash_tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_bash",
                name="bash",
                arguments={"command": "python3 /tmp/a.py"},
            )
        ],
    )
    final_response = AgentResponse(
        content="改用 ppt_edit 处理",
        model="test-model",
    )

    call_count = 0
    prompts_seen: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        prompts_seen.append("\n".join(m.content for m in messages if hasattr(m, "content")))
        if call_count <= 3:
            return bash_tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BashAlwaysFailTool())

    result = await run_agent(
        message="给第二页配图",
        session_id="test-bash-circuit",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "改用 ppt_edit 处理"
    assert call_count == 4
    assert any("同一 bash 命令模板已连续失败 3 次" in p for p in prompts_seen)


@pytest.mark.asyncio
async def test_run_agent_includes_ppt_edit_for_followup_office_message() -> None:
    captured_tool_names: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],  # noqa: ARG001
        *,
        tools: Any = None,
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        if isinstance(tools, list):
            for t in tools:
                if hasattr(t, "name"):
                    name = str(getattr(t, "name", "")).strip()
                    if name:
                        captured_tool_names.append(name)
                    continue
                if isinstance(t, dict):
                    name = str(t.get("name", "")).strip()
                    if not name and isinstance(t.get("function"), dict):
                        name = str(t["function"].get("name", "")).strip()
                    if name:
                        captured_tool_names.append(name)
        return AgentResponse(content="收到", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BashProbeTool())
    registry.register(_PptEditNoopTool())

    now = datetime.now(UTC)
    session = Session(
        id="s-followup-office",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="anthropic/claude-sonnet-4-20250514",
        created_at=now,
        updated_at=now,
        metadata={"last_pptx_path": "/tmp/贵州2日游.pptx"},
    )

    result = await run_agent(
        message="第一页的黑色条不好看，换种格式",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "收到"
    assert "ppt_edit" in captured_tool_names


@pytest.mark.asyncio
async def test_run_agent_requires_dark_bar_target_hit_for_ppt_edit() -> None:
    first = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_ppt",
                name="ppt_edit",
                arguments={
                    "path": "/tmp/a.pptx",
                    "slide_index": 1,
                    "action": "apply_business_style",
                },
            )
        ],
    )
    second = AgentResponse(content="继续处理", model="test-model")
    call_count = 0
    prompts_seen: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        nonlocal call_count
        call_count += 1
        prompts_seen.append("\n".join(m.content for m in messages if hasattr(m, "content")))
        if call_count == 1:
            return first
        return second

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_PptEditBusinessNoHitTool())

    now = datetime.now(UTC)
    session = Session(
        id="s-dark-bar",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="anthropic/claude-sonnet-4-20250514",
        created_at=now,
        updated_at=now,
        metadata={"last_pptx_path": "/tmp/a.pptx"},
    )

    result = await run_agent(
        message="第一页封面的黑色横条不好看，换一种方式",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "继续处理"
    assert any("未命中用户指定对象：黑色横条仍未被替换" in p for p in prompts_seen)


@pytest.mark.asyncio
async def test_run_agent_repairs_browser_query_without_action() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[ToolCall(id="tc_browser", name="browser", arguments={"query": "杨幂近照"})],
    )
    final_response = AgentResponse(content="已完成", model="test-model")

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

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserProbeTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-repair-query",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "已完成"
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_repairs_bash_cmd_alias() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[ToolCall(id="tc_bash", name="bash", arguments={"cmd": "echo hi"})],
    )
    final_response = AgentResponse(content="done", model="test-model")

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

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BashProbeTool())

    result = await run_agent(
        message="执行命令",
        session_id="test-bash-repair-cmd",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result.endswith("done")
    assert call_count == 2


@pytest.mark.asyncio
async def test_run_agent_retries_direct_python_script_bash_invocation() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_bash",
                name="bash",
                arguments={"command": "/tmp/test_nano_banana_2.py --mode edit"},
            )
        ],
    )
    final_response = AgentResponse(content="done", model="test-model")

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
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    bash_tool = _BashPyScriptRetryTool()
    registry.register(bash_tool)

    result = await run_agent(
        message="执行 nano banana 图生图",
        session_id="test-bash-retry-py-script",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result.endswith("done")
    assert len(bash_tool.commands) == 2
    assert bash_tool.commands[0] == "/tmp/test_nano_banana_2.py --mode edit"
    assert "python3.12 /tmp/test_nano_banana_2.py --mode edit" in bash_tool.commands[1]


@pytest.mark.asyncio
async def test_run_agent_executes_multiple_nano_banana_bash_calls_in_parallel() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_nb_1",
                name="bash",
                arguments={"command": "./python/bin/python3.12 /tmp/test_nano_banana_2.py --mode text --prompt a"},
            ),
            ToolCall(
                id="tc_nb_2",
                name="bash",
                arguments={"command": "./python/bin/python3.12 /tmp/test_nano_banana_2.py --mode text --prompt b"},
            ),
            ToolCall(
                id="tc_nb_3",
                name="bash",
                arguments={"command": "./python/bin/python3.12 /tmp/test_nano_banana_2.py --mode text --prompt c"},
            ),
        ],
    )
    final_response = AgentResponse(content="done", model="test-model")

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
            return tool_response
        return final_response

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    bash_tool = _DelayedNanoBananaBashTool()
    registry.register(bash_tool)

    result = await run_agent(
        message="并发生成三张香蕉图片",
        session_id="test-nano-banana-parallel",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result.endswith("done")
    assert call_count == 2
    assert len(bash_tool.started_at) == 3
    assert max(bash_tool.started_at) - min(bash_tool.started_at) < 0.04


@pytest.mark.asyncio
async def test_run_agent_uses_fixed_nano_banana_command_when_params_are_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    image_path = tmp_path / "input.png"
    image_path.write_bytes(b"image")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": f"把这张图改成天使翅膀\n\n(用户发送了图片)\n![飞书图片1]({image_path})",
                    "images": 1,
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message=f"把这张图改成天使翅膀\n\n(用户发送了图片)\n![飞书图片1]({image_path})",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "当前使用模型：香蕉2" in result
    assert str(output_path) in result
    assert len(bash_tool.commands) == 1
    assert "--mode edit" in bash_tool.commands[0]
    assert f"--input-image {image_path}" in bash_tool.commands[0]


@pytest.mark.asyncio
async def test_run_agent_repairs_garbled_browser_query_to_user_message() -> None:
    tool_response = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_browser",
                name="browser",
                arguments={"action": "search_images", "text": "2026 \\n0\\n0\\n0\\n0"},
            )
        ],
    )
    final_response = AgentResponse(content="ok", model="test-model")

    call_count = 0
    captured: list[str] = []

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

    class _BrowserCaptureTool(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="browser",
                description="capture",
                parameters=[
                    ToolParameter(name="action", type="string", description="action"),
                    ToolParameter(name="text", type="string", description="text"),
                ],
            )

        async def execute(self, **kwargs: Any) -> ToolResult:
            captured.append(str(kwargs.get("text", "")))
            return ToolResult(success=True, output="ok")

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserCaptureTool())

    result = await run_agent(
        message="给我杨幂新年写真高清图",
        session_id="test-browser-repair-garbled",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "ok"
    assert call_count == 2
    assert captured and captured[0] == "给我杨幂新年写真高清图"


@pytest.mark.asyncio
async def test_run_agent_rejects_escaped_block_file_edit_args() -> None:
    bad_file_edit = AgentResponse(
        content="",
        model="test-model",
        tool_calls=[
            ToolCall(
                id="tc_edit",
                name="file_edit",
                arguments={
                    "path": "/tmp/a.py",
                    "old_string": "line1\\nline2\\nline3\\nline4",
                    "new_string": "x\\ny\\nz\\nw",
                },
            )
        ],
    )
    final_response = AgentResponse(content="我改用 file_write 重写脚本", model="test-model")

    call_count = 0
    tool_called = False

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
            return bad_file_edit
        return final_response

    class _FileEditProbeTool(Tool):
        @property
        def definition(self) -> ToolDefinition:
            return ToolDefinition(
                name="file_edit",
                description="probe file_edit",
                parameters=[
                    ToolParameter(name="path", type="string", description="path"),
                    ToolParameter(name="old_string", type="string", description="old"),
                    ToolParameter(name="new_string", type="string", description="new"),
                ],
            )

        async def execute(self, **kwargs: Any) -> ToolResult:  # noqa: ARG002
            nonlocal tool_called
            tool_called = True
            return ToolResult(success=True, output="edited")

    router = _make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_FileEditProbeTool())

    result = await run_agent(
        message="重做这个 python 脚本",
        session_id="test-file-edit-escaped-block",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "我改用 file_write 重写脚本"
    assert call_count == 2
    assert not tool_called


class TestParseFallbackToolCalls:
    def test_fenced_json(self) -> None:
        text = '```json\n{"tool": "bash", "arguments": {"command": "ls"}}\n```'
        calls = _parse_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "ls"}

    def test_bare_json(self) -> None:
        text = '好的，我来执行 {"tool": "bash", "arguments": {"command": "pwd"}}'
        calls = _parse_fallback_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"

    def test_no_tool(self) -> None:
        text = "这是普通文本，没有工具调用。"
        calls = _parse_fallback_tool_calls(text)
        assert calls == []


class _DummyMemoryManager:
    def __init__(self, recalled: str = "") -> None:
        self._recalled = recalled
        self.recall_calls = 0
        self.capture_calls = 0
        self.capture_payloads: list[str] = []
        self.policy_calls = 0
        self.style_calls = 0

    def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
        self.policy_calls += 1
        return (True, True)

    async def get_global_style_directive(self) -> str:
        self.style_calls += 1
        return ""

    async def recall(  # noqa: PLR0913
        self,
        query: str,  # noqa: ARG002
        max_tokens: int = 500,  # noqa: ARG002
        limit: int = 10,  # noqa: ARG002
        *,
        include_profile: bool = True,
        include_raw: bool = True,
    ) -> str:
        self.recall_calls += 1
        if include_profile and not include_raw:
            return "【长期记忆画像】\n用户偏好简洁。"
        if include_raw and not include_profile:
            return self._recalled
        return self._recalled

    async def build_profile_for_injection(  # noqa: PLR0913
        self,
        *,
        max_tokens: int,  # noqa: ARG002
        router: Any = None,  # noqa: ARG002
        model_id: str = "",  # noqa: ARG002
        exclude_style: bool = False,  # noqa: ARG002
    ) -> str:
        self.recall_calls += 1
        return "【长期记忆画像】\n用户偏好简洁。"

    async def auto_capture_user_message(  # noqa: PLR0913
        self,
        content: str,
        *,
        source: str,  # noqa: ARG002
        mode: str = "balanced",  # noqa: ARG002
        cooldown_seconds: int = 180,  # noqa: ARG002
        max_per_hour: int = 12,  # noqa: ARG002
        batch_size: int = 3,  # noqa: ARG002
        merge_window_seconds: int = 120,  # noqa: ARG002
    ) -> bool:
        self.capture_calls += 1
        self.capture_payloads.append(content)
        return True

    async def organize_if_needed(self, **kwargs: Any) -> bool:  # noqa: ARG002
        return False


@pytest.mark.asyncio
async def test_run_agent_injects_recalled_memory_into_system_prompt() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="收到", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _DummyMemoryManager(recalled="- 用户喜欢简洁回答")
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    result = await run_agent(
        message="继续上次的话题",
        session_id="test-memory-recall",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert result == "收到"
    assert memory.policy_calls == 1
    assert memory.recall_calls == 2
    assert any(
        m.role == "system" and "长期记忆召回" in m.content
        for m in captured_messages
    )


@pytest.mark.asyncio
async def test_run_agent_auto_captures_user_fact_into_memory() -> None:
    router = _make_router(response=AgentResponse(content="记住了", model="test-model"))
    memory: Any = _DummyMemoryManager()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="我喜欢 Rust，请记住",
        session_id="test-memory-compact",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert memory.capture_calls == 1
    assert "我喜欢 Rust" in memory.capture_payloads[0]


@pytest.mark.asyncio
async def test_run_agent_skips_recall_when_policy_not_triggered() -> None:
    class _NoRecallMemory(_DummyMemoryManager):
        def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
            self.policy_calls += 1
            return (False, False)

    router = _make_router(response=AgentResponse(content="ok", model="test-model"))
    memory: Any = _NoRecallMemory(recalled="- should_not_be_used")
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    result = await run_agent(
        message="你好",
        session_id="test-memory-no-recall",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert result == "ok"
    assert memory.policy_calls == 1
    assert memory.recall_calls == 0


@pytest.mark.asyncio
async def test_run_agent_creation_task_auto_injects_profile_memory() -> None:
    class _NoRecallMemory(_DummyMemoryManager):
        def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
            self.policy_calls += 1
            return (False, False)

    router = _make_router(response=AgentResponse(content="已开始制作", model="test-model"))
    memory: Any = _NoRecallMemory(recalled="- raw_should_not_be_used")
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    result = await run_agent(
        message="帮我做一份香港两日游PPT",
        session_id="test-memory-creation-auto-l0",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    assert result == "已开始制作"
    assert memory.policy_calls == 1
    assert memory.recall_calls == 1


@pytest.mark.asyncio
async def test_run_agent_injects_global_style_directive() -> None:
    captured_messages: list[Any] = []

    class _StyleMemory(_DummyMemoryManager):
        async def get_global_style_directive(self) -> str:
            self.style_calls += 1
            return "回答风格：简洁明了，先结论后细节。"

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _StyleMemory()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="你好",
        session_id="test-memory-style-inject",
        config=cfg,
        router=router,
        memory_manager=memory,
    )
    assert memory.style_calls == 1
    assert any(
        m.role == "system" and "全局回复风格偏好" in m.content
        for m in captured_messages
    )


@pytest.mark.asyncio
async def test_run_agent_excludes_style_lines_from_profile_when_global_style_exists() -> None:
    captured_messages: list[Any] = []

    class _StyleAwareMemory(_DummyMemoryManager):
        async def get_global_style_directive(self) -> str:
            self.style_calls += 1
            return "普通问答默认简洁紧凑，避免冗余客套和过多空行。"

        async def build_profile_for_injection(  # noqa: PLR0913
            self,
            *,
            max_tokens: int,  # noqa: ARG002
            router: Any = None,  # noqa: ARG002
            model_id: str = "",  # noqa: ARG002
            exclude_style: bool = False,
        ) -> str:
            self.recall_calls += 1
            if exclude_style:
                return "【长期记忆画像】\n制作PPT时图片仅允许裁剪和等比缩放。"
            return (
                "【长期记忆画像】\n普通问答默认简洁紧凑，避免冗余客套和过多空行；"
                "制作PPT时图片仅允许裁剪和等比缩放。"
            )

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _StyleAwareMemory()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="帮我做一份PPT",
        session_id="test-memory-style-dedupe",
        config=cfg,
        router=router,
        memory_manager=memory,
    )

    memory_prompt = next(
        m.content
        for m in captured_messages
        if m.role == "system" and "长期记忆召回" in m.content
    )
    assert "制作PPT时图片仅允许裁剪和等比缩放" in memory_prompt
    assert "普通问答默认简洁紧凑" not in memory_prompt


@pytest.mark.asyncio
async def test_run_agent_injects_external_memory_hint() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    memory: Any = _DummyMemoryManager()
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    _ = await run_agent(
        message="帮我优化这个脚本",
        session_id="test-external-memory",
        config=cfg,
        router=router,
        memory_manager=memory,
        extra_memory="【EvoMap 协作经验候选】\n- 遇到超时优先增加重试和退避",
    )

    assert any(
        m.role == "system" and "协作网络的外部经验候选" in m.content
        for m in captured_messages
    )


@pytest.mark.asyncio
async def test_run_agent_truncates_external_memory_when_compressor_unavailable() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    router.resolve = MagicMock(side_effect=RuntimeError("compress model missing"))
    cfg = WhaleclawConfig()
    cfg.agent.memory.organizer_background = False

    huge = "X" * 12000
    _ = await run_agent(
        message="测试外部经验注入",
        session_id="test-external-memory-truncate",
        config=cfg,
        router=router,
        extra_memory=huge,
    )

    ext_msg = next(
        m for m in captured_messages
        if m.role == "system" and "协作网络的外部经验候选" in m.content
    )
    assert ext_msg.content.count("X") <= 3000


@pytest.mark.asyncio
async def test_run_agent_keeps_short_external_memory_without_compress() -> None:
    captured_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        if messages and messages[0].role == "system" and "外部经验压缩器" in messages[0].content:
            return AgentResponse(content="压缩后经验", model="compress-model")
        captured_messages[:] = messages
        return AgentResponse(content="ok", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    cfg = WhaleclawConfig()
    cfg.agent.summarizer.enabled = False

    _ = await run_agent(
        message="测试短经验压缩",
        session_id="test-external-memory-short-compress",
        config=cfg,
        router=router,
        extra_memory="【EvoMap 协作经验候选】\n- 原始经验文本",
    )

    ext_msg = next(
        m for m in captured_messages
        if m.role == "system" and "协作网络的外部经验候选" in m.content
    )
    assert "压缩后经验" not in ext_msg.content
    assert "原始经验文本" in ext_msg.content


@pytest.mark.asyncio
async def test_run_agent_skill_lock_requires_explicit_done_confirmation() -> None:
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
            return AgentResponse(
                content="",
                model="test-model",
                tool_calls=[ToolCall(id="tc_bash", name="bash", arguments={"command": "echo ok"})],
            )
        return AgentResponse(content="已出图", model="test-model")

    registry = ToolRegistry()
    registry.register(_BashProbeTool())
    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-1",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    first = await run_agent(
        message="/use nano-banana-image-t8 一只熊猫在上海街头跳舞",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )
    assert "已出图" in first
    assert "任务完成" in first
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    assert session.metadata.get("skill_lock_waiting_done") is True
    assert call_count == 2

    router2 = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    second = await run_agent(
        message="任务结束",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router2,
        registry=registry,
        session=session,
    )
    assert second == "已确认任务完成，已解除本轮技能锁定。"
    assert "locked_skill_ids" not in session.metadata
    assert "skill_lock_waiting_done" not in session.metadata
    router2.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_reports_unlock_not_completed_for_task_done_intent_near_miss() -> None:
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-near-miss",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_lock_waiting_done": True,
        },
    )

    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    result = await run_agent(
        message="本轮结束啦",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "还没有完成正式解锁" in result
    assert "请直接回复“任务完成”或“任务结束”" in result
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_unlocks_locked_skill_even_when_waiting_done_flag_is_false() -> None:
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-unlock-without-waiting-flag",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_lock_waiting_done": False,
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "旧任务",
                    "images": 2,
                }
            },
        },
    )

    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    result = await run_agent(
        message="任务完成",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert result == "已确认任务完成，已解除本轮技能锁定。"
    assert "locked_skill_ids" not in session.metadata
    assert "skill_lock_waiting_done" not in session.metadata
    assert "skill_param_state" not in session.metadata
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_task_done_clears_persisted_skill_lock(tmp_path: Path) -> None:
    store = SessionStore(db_path=tmp_path / "unlock.db")
    await store.open()
    try:
        manager = SessionManager(store, WhaleclawConfig())
        session = await manager.create("webchat", "unlock-user")
        session.metadata = {
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_lock_waiting_done": True,
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "旧任务",
                }
            },
        }
        await manager.update_metadata(session, session.metadata)

        router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
        result = await run_agent(
            message="任务完成",
            session_id=session.id,
            config=WhaleclawConfig(),
            router=router,
            session=session,
            session_manager=manager,
        )

        assert result == "已确认任务完成，已解除本轮技能锁定。"
        reloaded = await manager.get(session.id)
        assert reloaded is not None
        assert "locked_skill_ids" not in reloaded.metadata
        assert "skill_lock_waiting_done" not in reloaded.metadata
        assert "skill_param_state" not in reloaded.metadata
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_run_agent_skill_lock_status_question_uses_persisted_metadata(
    tmp_path: Path,
) -> None:
    store = SessionStore(db_path=tmp_path / "lock_status.db")
    await store.open()
    try:
        manager = SessionManager(store, WhaleclawConfig())
        persisted = await manager.create("webchat", "lock-status-user")
        stale_session = Session(
            id=persisted.id,
            channel=persisted.channel,
            peer_id=persisted.peer_id,
            messages=[],
            model=persisted.model,
            created_at=persisted.created_at,
            updated_at=persisted.updated_at,
            metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
        )
        await store.update_session_field(persisted.id, metadata={})

        router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
        result = await run_agent(
            message="现在被锁定在哪个技能里面？",
            session_id=stale_session.id,
            config=WhaleclawConfig(),
            router=router,
            session=stale_session,
            session_manager=manager,
        )

        assert result == "当前没有技能锁定。"
        router.chat.assert_not_called()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_run_agent_applies_locked_skill_set_to_system_prompt() -> None:
    seen_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        seen_messages.extend(messages)
        return AgentResponse(content="继续处理", model="test-model")

    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-2",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["skill-a", "skill-b"]},
    )
    router = _make_router(chat_fn=fake_chat)
    await run_agent(
        message="继续改一下",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    joined = "\n".join(
        str(m.content) for m in seen_messages if getattr(m, "role", "") == "system"
    )
    assert "当前会话已锁定技能：skill-a, skill-b" in joined


@pytest.mark.asyncio
async def test_run_agent_applies_nano_banana_model_and_recent_image_hints_to_system_prompt(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "banana-ref.png"
    image_path.write_bytes(b"png-bytes")
    seen_messages: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        seen_messages.extend(messages)
        return AgentResponse(content="继续处理", model="test-model")

    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-nano-system",
        channel="feishu",
        peer_id="u1",
        messages=[
            Message(
                role="user",
                content=f"(用户发送了图片)\n![飞书图片1]({image_path})",
            )
        ],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_input_image_paths": [str(image_path)],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "一只熊猫在上海跳舞",
                    "__model_display__": "香蕉pro",
                }
            },
        },
    )
    router = _make_router(chat_fn=fake_chat)
    await run_agent(
        message="用香蕉pro重试",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    joined = "\n".join(
        str(m.content) for m in seen_messages if getattr(m, "role", "") == "system"
    )
    assert "当前本轮模型是：香蕉pro" in joined
    assert "--model` 和 `--edit-model` 都设置为 `香蕉pro`" in joined
    assert str(image_path) in joined


@pytest.mark.asyncio
async def test_run_agent_auto_locks_when_user_explicitly_mentions_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana", "文生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/SKILL.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-3",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )
    result = await run_agent(
        message="使用nanobanana的技能，文生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "nano-banana-image-t8" in result
    assert "技能" in result
    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    assert session.metadata.get("skill_lock_waiting_done") is False


@pytest.mark.asyncio
async def test_run_agent_auto_locks_when_user_hits_specific_skill_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉文生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    prompt="请提供 Nano Banana API Key",
                ),
            ],
        ),
        source_path=Path("/tmp/nano-trigger.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-trigger-1",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    result = await run_agent(
        message="我要用香蕉生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "我将使用 nano-banana-image-t8 技能继续完成任务。" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_auto_locks_even_for_one_shot_skill_in_task_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/SKILL2.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-4",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    result = await run_agent(
        message="使用ppt-generator技能，帮我制作个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "ppt-generator" in result
    assert "技能" in result
    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") == ["ppt-generator"]


@pytest.mark.asyncio
async def test_run_agent_rejects_skill_switch_without_user_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/a.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        is_user_installed=True,
        source_path=Path("/tmp/b.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-5",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="我在想是不是该用ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "请先回复“任务完成”" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]


@pytest.mark.asyncio
async def test_run_agent_rejects_other_skill_trigger_without_explicit_skill_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/a-trigger.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        is_user_installed=True,
        source_path=Path("/tmp/b-trigger.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="不应直接执行", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-trigger-remind",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="帮我做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "当前会话仍锁定在 nano-banana-image-t8 技能" in result
    assert "请先回复“任务完成”" in result
    assert session.metadata.get("pending_skill_switch_ids") == ["ppt-generator"]
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]


@pytest.mark.asyncio
async def test_run_agent_keeps_locked_skill_when_user_did_not_explicitly_request_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        is_user_installed=True,
        source_path=Path("/tmp/b1.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="继续生图处理", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-5b",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="继续生成一张横版香蕉海报图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "同意切换技能" not in result
    assert "继续生图处理" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]


@pytest.mark.asyncio
async def test_run_agent_requires_task_done_before_switching_skills(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/a2.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        is_user_installed=True,
        source_path=Path("/tmp/b2.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-6",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="改用ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "请先回复“任务完成”" in result
    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_unlocks_then_requires_reenter_command_for_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/a3.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        is_user_installed=True,
        source_path=Path("/tmp/b3.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    seen_messages: list[str] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        seen_messages.extend(str(m.content) for m in messages if getattr(m, "role", "") == "user")
        return AgentResponse(content="开始做PPT", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-6b",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    first = await run_agent(
        message="我在想是不是该用ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "请先回复“任务完成”" in first
    assert session.metadata.get("pending_skill_switch_ids") == ["ppt-generator"]

    second = await run_agent(
        message="任务完成",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert second == "已确认任务完成，已解除本轮技能锁定。"
    assert session.metadata.get("locked_skill_ids") is None
    assert "pending_skill_switch_ids" not in session.metadata
    assert "pending_skill_switch_message" not in session.metadata
    assert seen_messages == []


@pytest.mark.asyncio
async def test_run_agent_switches_after_unlock_and_reenter_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_a = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/a3.md"),
    )
    skill_b = Skill(
        id="ppt-generator",
        name="PPT Generator",
        triggers=["ppt"],
        instructions="x",
        lock_session=False,
        is_user_installed=True,
        source_path=Path("/tmp/b3.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill_b] if "ppt" in user_message.lower() else [skill_a],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="收到", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-skill-lock-7",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    first = await run_agent(
        message="换成ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "请先回复“任务完成”" in first
    unlocked = await run_agent(
        message="任务完成",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )
    assert unlocked == "已确认任务完成，已解除本轮技能锁定。"

    result = await run_agent(
        message="换成ppt-generator技能做个PPT",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "收到" in result
    assert session.metadata.get("locked_skill_ids") is None


@pytest.mark.asyncio
async def test_non_lock_skill_does_not_persist_lock_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_skill = Skill(
        id="browser-control",
        name="浏览器控制",
        triggers=["截图"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/browser.md"),
    )
    banana_skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        source_path=Path("/tmp/banana.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [browser_skill] if "截图" in user_message else [banana_skill],  # noqa: ARG005,E501
    )

    router = _make_router(response=AgentResponse(content="截图成功", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-non-lock-browser-skill",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    first = await run_agent(
        message="截图一下桌面给我",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "截图成功" in first
    assert session.metadata.get("locked_skill_ids") is None

    second = await run_agent(
        message="使用香蕉生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert session.metadata.get("locked_skill_ids") == ["nano-banana-image-t8"]
    assert "nano-banana-image-t8" in second


@pytest.mark.asyncio
async def test_use_non_lock_skill_does_not_create_skill_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser_skill = Skill(
        id="browser-control",
        name="浏览器控制",
        triggers=["截图"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/browser-use.md"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [browser_skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="浏览器技能执行", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-non-lock-browser-use",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    result = await run_agent(
        message="/use browser-control 截图一下桌面",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "浏览器技能执行" in result
    assert session.metadata.get("locked_skill_ids") is None


@pytest.mark.asyncio
async def test_nano_banana_guard_lists_missing_params_before_execution() -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    prompt="请提供 API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="ratio",
                    label="尺寸/比例",
                    type="ratio",
                    required=False,
                    aliases=["比例", "尺寸", "size"],
                    prompt="可选填写比例或尺寸",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard.md"),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-1",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="使用nano banana制作文生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "API Key" in result
    assert "当前模型：香蕉2（0.1元）可切换模型香蕉pro（0.2元）" in result
    assert "提示词" in result
    assert "图生图图片：已收到 0 张（至少 1 张）" in result
    assert "切换本次模型：切换香蕉2（pro）。设置默认模型：默认模型香蕉2（pro）" in result
    router.chat.assert_not_called()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_nano_banana_guard_uses_saved_key_without_asking_again(
    tmp_path: Path,
) -> None:
    saved_key = tmp_path / "nano_banana_api_key.txt"
    saved_key.write_text("sk-test-saved-key", encoding="utf-8")
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉文生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    saved_file=str(saved_key),
                    prompt="请提供 Nano Banana API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="images",
                    label="图生图图片",
                    type="images",
                    required=False,
                    min_count=1,
                    prompt="图生图时请上传至少 1 张图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_saved_key.md"),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-saved-key",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {},
            },
        },
    )

    result = await run_agent(
        message="尺寸 1:1",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "API Key" in result
    assert "已就绪" in result
    assert "请提供 Nano Banana API Key" not in result
    assert "请提供提示词" in result
    router.chat.assert_not_called()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_nano_banana_guard_keeps_fixed_template_for_activation_only_message(
    tmp_path: Path,
) -> None:
    saved_key = tmp_path / "nano_banana_api_key.txt"
    saved_key.write_text("sk-test-saved-key", encoding="utf-8")
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉文生图", "香蕉图生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    saved_file=str(saved_key),
                    prompt="请提供 Nano Banana API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="images",
                    label="图生图图片",
                    type="images",
                    required=False,
                    min_count=1,
                    prompt="图生图时请上传至少 1 张图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_activation_only.md"),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-activation-only",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={"locked_skill_ids": ["nano-banana-image-t8"]},
    )

    result = await run_agent(
        message="使用香蕉生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "当前会话仍在香蕉生图技能里。" in result
    assert "如果要继续生图，请直接发送提示词或图片" in result
    assert "请回复“任务完成”解除技能锁定" in result
    router.chat.assert_not_called()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_nano_banana_control_message_does_not_overwrite_existing_prompt() -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉pro"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    prompt="请提供 Nano Banana API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="images",
                    label="图生图图片",
                    type="images",
                    required=False,
                    min_count=1,
                    prompt="图生图时请上传至少 1 张图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_retry.md"),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = _make_router(response=AgentResponse(content="继续执行", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-retry",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "把男孩衣服改成紫色",
                    "images": 1,
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="用香蕉pro重试",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert result == "继续执行"
    assert (
        cast(dict[str, object], session.metadata["skill_param_state"]["nano-banana-image-t8"])[
            "prompt"
        ]
        == "把男孩衣服改成紫色"
    )
    router.chat.assert_called_once()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_nano_banana_activation_message_reminds_when_session_is_already_locked() -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉pro"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    prompt="请提供 Nano Banana API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="images",
                    label="图生图图片",
                    type="images",
                    required=False,
                    min_count=1,
                    prompt="图生图时请上传至少 1 张图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_activation_complete.md"),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-activation-complete",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "算了，继续讲笑话给我",
                    "images": 4,
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="使用香蕉生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "当前会话仍在香蕉生图技能里。" in result
    assert "当前模型：香蕉2。" in result
    assert "如果要继续生图，请直接发送提示词或图片" in result
    assert "请回复“任务完成”解除技能锁定" in result
    router.chat.assert_not_called()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_nano_banana_activation_message_prefers_last_used_model_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = _make_router()
    monkeypatch.setattr(
        loop_mod,
        "_load_saved_nano_banana_model_display",
        lambda: "香蕉pro",
    )

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-activation-last-model",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_nano_banana_model_display": "香蕉2",
        },
    )

    result = await run_agent(
        message="使用香蕉生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "当前会话仍在香蕉生图技能里。" in result
    assert "当前模型：香蕉2。" in result
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_returns_referenced_previous_image_under_locked_nano_banana(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.png"
    current_path.write_bytes(b"current")
    previous_path = tmp_path / "previous.png"
    previous_path.write_bytes(b"previous")
    older_path = tmp_path / "older.png"
    older_path.write_bytes(b"older")

    router = _make_router()
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-history-lookup",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "image_reference_history": [
                str(current_path),
                str(previous_path),
                str(older_path),
            ],
        },
    )

    result = await run_agent(
        message="那再上一张呢？",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert str(older_path) in result
    assert "![历史图片]" in result
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_reuses_recent_session_images_for_locked_image_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="images",
                    label="图片",
                    type="images",
                    required=True,
                    prompt="请上传图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_images.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    image_path = tmp_path / "ref.png"
    image_path.write_bytes(b"png-bytes")
    previous_user_message = Message(
        role="user",
        content=f"(用户发送了图片)\n![飞书图片1]({image_path})",
    )

    seen_user_images: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        for item in messages:
            if getattr(item, "role", "") == "user":
                seen_user_images.append(getattr(item, "images", None))
        return AgentResponse(content="开始图生图", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-images-1",
        channel="feishu",
        peer_id="u1",
        messages=[previous_user_message],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "处理这张图",
                    "images": 1,
                },
            },
            "last_input_image_paths": [str(image_path)],
        },
    )

    result = await run_agent(
        message="用 nano banana 处理这张图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "开始图生图" in result
    assert any(images and len(images) == 1 for images in seen_user_images)
    last_non_empty = next(images for images in reversed(seen_user_images) if images)
    assert last_non_empty[0].mime == "image/png"


@pytest.mark.asyncio
async def test_run_agent_prefers_latest_generated_image_for_locked_image_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="images",
                    label="图片",
                    type="images",
                    required=True,
                    prompt="请上传图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_generated_image.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    original_path = tmp_path / "original.png"
    original_path.write_bytes(b"original-image")
    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")

    previous_user_message = Message(
        role="user",
        content=f"(用户发送了图片)\n![飞书图片1]({original_path})",
    )
    previous_assistant_message = Message(
        role="assistant",
        content=f"结果图：\n文件路径：`{generated_path}`",
    )

    seen_user_images: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        for item in messages:
            if getattr(item, "role", "") == "user":
                seen_user_images.append(getattr(item, "images", None))
        return AgentResponse(content="继续图生图", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-generated-1",
        channel="feishu",
        peer_id="u1",
        messages=[previous_user_message, previous_assistant_message],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "sk-test",
                    "prompt": "让猫更有气势",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="姿势改成胜利手势",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "继续图生图" in result
    assert any(images for images in seen_user_images)
    last_non_empty = next(images for images in reversed(seen_user_images) if images)
    assert base64.b64decode(last_non_empty[0].data) == b"generated-image"


@pytest.mark.asyncio
async def test_run_agent_regenerate_reuses_last_input_image_set_for_locked_image_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="images",
                    label="图片",
                    type="images",
                    required=True,
                    prompt="请上传图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_regenerate.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    original_1 = tmp_path / "original-1.png"
    original_1.write_bytes(b"original-1")
    original_2 = tmp_path / "original-2.png"
    original_2.write_bytes(b"original-2")
    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")

    seen_user_images: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        for item in messages:
            if getattr(item, "role", "") == "user":
                seen_user_images.append(getattr(item, "images", None))
        return AgentResponse(content="重新图生图", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-regenerate-1",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_input_image_paths": [str(original_1), str(original_2)],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "sk-test",
                    "prompt": "让图1和图2组合",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="这图不好看，重新生成下",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "重新图生图" in result
    last_non_empty = next(images for images in reversed(seen_user_images) if images)
    assert len(last_non_empty) == 2
    assert base64.b64decode(last_non_empty[0].data) == b"original-1"
    assert base64.b64decode(last_non_empty[1].data) == b"original-2"


@pytest.mark.asyncio
async def test_run_agent_regenerate_keeps_text_mode_for_fixed_nano_banana_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_regenerate_text.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    stale_input_path = tmp_path / "stale-input.png"
    stale_input_path.write_bytes(b"stale-input")
    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-regenerate-text",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_input_image_paths": [str(stale_input_path)],
            "last_nano_banana_mode": "text",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "一只熊猫在上海街头跳舞",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "text",
                }
            },
        },
    )

    result = await run_agent(
        message="重新生成，背景改成夜景霓虹",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "当前使用模型：香蕉2" in result
    assert str(output_path) in result
    assert len(bash_tool.commands) == 1
    assert "--mode text" in bash_tool.commands[0]
    assert "--input-image" not in bash_tool.commands[0]


@pytest.mark.asyncio
async def test_resolve_nano_banana_input_paths_uses_previous_history_reference(
    tmp_path: Path,
) -> None:
    current_path = tmp_path / "current.png"
    current_path.write_bytes(b"current")
    previous_path = tmp_path / "previous.png"
    previous_path.write_bytes(b"previous")
    older_path = tmp_path / "older.png"
    older_path.write_bytes(b"older")

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-image-history-ref",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "image_reference_history": [
                str(current_path),
                str(previous_path),
                str(older_path),
            ],
            "last_generated_image_path": str(current_path),
            "last_input_image_paths": [str(current_path)],
            "last_nano_banana_mode": "edit",
        },
    )

    resolved = loop_mod._resolve_nano_banana_input_paths(  # noqa: SLF001
        "用上一张图重试，改成水墨画风格",
        session,
    )

    assert resolved == [str(previous_path)]


@pytest.mark.asyncio
async def test_resolve_nano_banana_input_paths_uses_numbered_uploaded_images(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.png"
    first_path.write_bytes(b"first")
    second_path = tmp_path / "second.png"
    second_path.write_bytes(b"second")

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-numbered-input-ref",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "last_input_image_paths": [str(first_path), str(second_path)],
            "last_nano_banana_mode": "edit",
        },
    )

    resolved = loop_mod._resolve_nano_banana_input_paths(  # noqa: SLF001
        "让图1的女子穿着图2的衣服，站在一望无际的沙漠中",
        session,
    )

    assert resolved == [str(first_path), str(second_path)]


@pytest.mark.asyncio
async def test_run_agent_regenerate_merges_new_prompt_delta_for_text_nano_banana(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_prompt_merge.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-prompt-merge",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_nano_banana_mode": "text",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "海贼王风格，路飞和娜美在篮球场打篮球",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "text",
                }
            },
        },
    )

    await run_agent(
        message="重新生成，我要真人风格",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "海贼王风格，路飞和娜美在篮球场打篮球" in command
    assert "我要真人风格" in command
    assert "--aspect-ratio auto" in command


@pytest.mark.asyncio
async def test_run_agent_new_prompt_replaces_previous_text_prompt_for_nano_banana(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_prompt_replace.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-prompt-replace",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_nano_banana_mode": "text",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "海贼王风格，路飞和娜美在篮球场打篮球",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "text",
                }
            },
        },
    )

    await run_agent(
        message="给我一张赛博朋克风格的城市夜景",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "--mode text" in command
    assert "给我一张赛博朋克风格的城市夜景" in command
    assert "海贼王风格" not in command


@pytest.mark.asyncio
async def test_run_agent_regenerate_merges_prompt_and_updates_ratio_for_text_nano_banana(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(
                    key="ratio",
                    type="ratio",
                    required=False,
                    aliases=["比例", "尺寸", "size"],
                ),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_prompt_ratio_merge.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-prompt-ratio-merge",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_nano_banana_mode": "text",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "海贼王风格，路飞和娜美在篮球场打篮球",
                    "ratio": "16:9",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "text",
                }
            },
        },
    )

    await run_agent(
        message="重新生成，我要真人风格的海贼王，路飞跟娜美在篮球场打篮球，图片比例是9:16",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "海贼王风格，路飞和娜美在篮球场打篮球" in command
    assert "我要真人风格的海贼王，路飞跟娜美在篮球场打篮球" in command
    assert "--aspect-ratio 9:16" in command


@pytest.mark.asyncio
async def test_run_agent_regenerate_reuses_original_inputs_and_merges_prompt_for_edit_nano_banana(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_regenerate_edit.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    original_path = tmp_path / "original.png"
    original_path.write_bytes(b"original-image")
    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-regenerate-edit",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_input_image_paths": [str(original_path)],
            "last_nano_banana_mode": "edit",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "将背景改成生化危机9的场景",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "edit",
                }
            },
        },
    )

    await run_agent(
        message="重新生成，再加一些烟雾和火花",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "--mode edit" in command
    assert f"--input-image {original_path}" in command
    assert f"--input-image {generated_path}" not in command
    assert "将背景改成生化危机9的场景" in command
    assert "再加一些烟雾和火花" in command


@pytest.mark.asyncio
async def test_run_agent_edit_followup_add_object_reuses_latest_generated_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_add_object.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-add-object",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_nano_banana_mode": "edit",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "将背景改成生化危机9的场景",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "edit",
                }
            },
        },
    )

    await run_agent(
        message="后面加一只暴君",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "--mode edit" in command
    assert f"--input-image {generated_path}" in command
    assert "将背景改成生化危机9的场景" in command
    assert "后面加一只暴君" in command


@pytest.mark.asyncio
async def test_run_agent_edit_followup_continues_last_edit_mode_without_explicit_edit_keyword(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_continue_edit_mode.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-continue-edit-mode",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_nano_banana_mode": "edit",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "将背景改成生化危机9的场景",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "edit",
                }
            },
        },
    )

    await run_agent(
        message="把压迫感再加强一点",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "--mode edit" in command
    assert f"--input-image {generated_path}" in command
    assert "把压迫感再加强一点" in command


@pytest.mark.asyncio
async def test_run_agent_text_to_image_followup_with_subject_reference_switches_to_edit_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_subject_reference_edit.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-subject-reference-edit",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_nano_banana_mode": "text",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "一只猩猩在丛林里荡秋千",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "text",
                }
            },
        },
    )

    await run_agent(
        message="让这猩猩穿着钢铁盔甲",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "--mode edit" in command
    assert f"--input-image {generated_path}" in command
    assert "一只猩猩在丛林里荡秋千" in command
    assert "让这猩猩穿着钢铁盔甲" in command


@pytest.mark.asyncio
async def test_run_agent_explicit_new_image_request_breaks_previous_edit_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_break_edit_chain.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-break-edit-chain",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_nano_banana_mode": "edit",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "把这个场景改成在水里面",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "edit",
                }
            },
        },
    )

    await run_agent(
        message="做一张图，海贼王路飞跟娜美正在掰手腕",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert "--mode text" in command
    assert "--input-image" not in command
    assert "海贼王路飞跟娜美正在掰手腕" in command
    assert "把这个场景改成在水里面" not in command


@pytest.mark.asyncio
async def test_run_agent_new_uploaded_image_starts_fresh_image_to_image_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_fresh_uploaded_image.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    previous_generated = tmp_path / "previous-generated.png"
    previous_generated.write_bytes(b"previous-generated")
    new_input = tmp_path / "new-input.png"
    new_input.write_bytes(b"new-input")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-fresh-uploaded-image",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(previous_generated),
            "last_input_image_paths": [str(previous_generated)],
            "last_nano_banana_mode": "edit",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "把这个场景改成在水里面",
                    "images": 1,
                    "__model_display__": "香蕉2",
                    "__last_mode__": "edit",
                }
            },
        },
    )

    await run_agent(
        message=(
            f"做一张图，赛博朋克城市里的机甲猫\n\n"
            f"(用户发送了图片)\n![飞书图片1]({new_input})"
        ),
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    command = bash_tool.commands[0]
    assert f"--input-image {new_input}" in command
    assert f"--input-image {previous_generated}" not in command
    assert "赛博朋克城市里的机甲猫" in command
    assert "把这个场景改成在水里面" not in command


@pytest.mark.asyncio
async def test_run_agent_uses_2k_image_size_for_nano_banana_pro_fixed_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_2k_pro.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉pro", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-2k-pro",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张海报",
                    "__model_display__": "香蕉pro",
                }
            },
        },
    )

    result = await run_agent(
        message="用香蕉pro生成一张海报",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "当前使用模型：香蕉pro" in result
    assert len(bash_tool.commands) == 1
    assert "--image-size 2K" in bash_tool.commands[0]


@pytest.mark.asyncio
async def test_run_agent_does_not_use_2k_image_size_for_nano_banana_2_fixed_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_fixed_runner_2k_gemini.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-fixed-runner-2k-gemini",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张海报",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="用香蕉2生成一张海报",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "当前使用模型：香蕉2" in result
    assert len(bash_tool.commands) == 1
    assert "--image-size 2K" not in bash_tool.commands[0]


@pytest.mark.asyncio
async def test_nano_banana_control_message_switches_model_without_running_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉pro"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_switch_model_without_generation.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-switch-model-without-generation",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张海报",
                    "__model_display__": "香蕉pro",
                }
            },
        },
    )

    result = await run_agent(
        message="不对呀，我的模型应该是香蕉2啊",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "已切换本次生图模型为：香蕉2" in result
    assert bash_tool.commands == []


@pytest.mark.asyncio
async def test_nano_banana_model_switch_phrase_does_not_run_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉pro"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_switch_model_phrase.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-switch-model-phrase",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张海报",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="模型切换香蕉Pro",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "已切换本次生图模型为：香蕉pro" in result
    assert bash_tool.commands == []


@pytest.mark.asyncio
async def test_nano_banana_activation_message_uses_guard_reply_after_unlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉文生图", "香蕉图生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="api_key",
                    label="API Key",
                    type="api_key",
                    required=True,
                    prompt="请提供 Nano Banana API Key",
                ),
                SkillParamItem(
                    key="prompt",
                    label="提示词",
                    type="text",
                    required=True,
                    aliases=["提示词", "prompt"],
                    prompt="请提供提示词",
                ),
                SkillParamItem(
                    key="images",
                    label="图生图图片",
                    type="images",
                    required=False,
                    min_count=1,
                    prompt="图生图时请上传至少 1 张图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_activation_after_unlock.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    monkeypatch.setattr(
        loop_mod,
        "_load_saved_nano_banana_model_display",
        lambda: "香蕉2",
    )

    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-activation-after-unlock",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    result = await run_agent(
        message="使用香蕉生图",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "我将使用 nano-banana-image-t8 技能继续完成任务。" in result
    assert "我先确认参数（缺啥补啥）：" in result
    assert "提示词：未提供" in result
    assert "请补充：" in result
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_keeps_locked_nano_banana_for_numbered_image_reference_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nano_skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉文生图", "香蕉图生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_numbered_ref.md"),
    )
    search_skill = Skill(
        id="search_images",
        name="搜索并下载图片",
        triggers=["图片", "照片", "搜图", "找图"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/search_images.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [nano_skill]
        if forced_skill_ids
        else [search_skill, nano_skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-numbered-image-ref",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "旧提示词",
                    "images": 2,
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="让图1的女子穿着图2的衣服，站在一望无际的草原里",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "如果你确实要切换到 search_images" not in result
    assert "当前会话仍锁定在 nano-banana-image-t8 技能" not in result


@pytest.mark.asyncio
async def test_run_agent_keeps_locked_nano_banana_for_regenerate_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nano_skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图", "香蕉文生图", "香蕉图生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
                SkillParamItem(key="images", type="images", required=False, min_count=1),
            ],
        ),
        source_path=Path("/tmp/nano_regenerate_followup.md"),
    )
    search_skill = Skill(
        id="search_images",
        name="搜索并下载图片",
        triggers=["图片", "照片", "搜图", "找图"],
        instructions="x",
        lock_session=False,
        source_path=Path("/tmp/search_images_regen.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [nano_skill]
        if forced_skill_ids
        else [search_skill, nano_skill],  # noqa: ARG005
    )

    router = _make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-regenerate-followup",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_nano_banana_mode": "edit",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "让图1和图2组合",
                    "images": 2,
                    "__model_display__": "香蕉2",
                    "__last_mode__": "edit",
                }
            },
        },
    )

    result = await run_agent(
        message="重新生成，让图片充满电影感",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "如果你确实要切换到 search_images" not in result
    assert "当前会话仍锁定在 nano-banana-image-t8 技能" not in result


@pytest.mark.asyncio
async def test_run_agent_regenerate_does_not_reuse_images_for_text_mode_nano_banana(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_guard_regenerate_text.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    stale_input_path = tmp_path / "stale-input.png"
    stale_input_path.write_bytes(b"stale-input")
    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")

    seen_user_images: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        for item in messages:
            if getattr(item, "role", "") == "user":
                seen_user_images.append(getattr(item, "images", None))
        return AgentResponse(content="继续文生图", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-regenerate-text",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_input_image_paths": [str(stale_input_path)],
            "last_nano_banana_mode": "text",
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "sk-test",
                    "prompt": "一只熊猫在上海街头跳舞",
                    "__model_display__": "香蕉2",
                    "__last_mode__": "text",
                }
            },
        },
    )

    result = await run_agent(
        message="重新生成，改成水彩风",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=ToolRegistry(),
        session=session,
    )

    assert "继续文生图" in result
    assert all(not images for images in seen_user_images)


@pytest.mark.asyncio
async def test_run_agent_does_not_reuse_images_for_plain_chat_under_locked_image_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="images",
                    label="图片",
                    type="images",
                    required=True,
                    prompt="请上传图片",
                ),
            ],
        ),
        source_path=Path("/tmp/nano_guard_plain_chat.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    original_path = tmp_path / "original.png"
    original_path.write_bytes(b"original-image")
    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")

    seen_user_images: list[Any] = []

    async def fake_chat(
        model_id: str,  # noqa: ARG001
        messages: list[Any],
        *,
        tools: Any = None,  # noqa: ARG001
        on_stream: Any = None,  # noqa: ARG001
    ) -> AgentResponse:
        for item in messages:
            if getattr(item, "role", "") == "user":
                seen_user_images.append(getattr(item, "images", None))
        return AgentResponse(content="讲个冷笑话", model="test-model")

    router = _make_router(chat_fn=fake_chat)
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-plain-chat-1",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "last_input_image_paths": [str(original_path)],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "sk-test",
                    "prompt": "生成一张猫图",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="讲个笑话",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "讲个冷笑话" in result
    assert all(not images for images in seen_user_images)


@pytest.mark.asyncio
async def test_run_agent_plain_chat_under_locked_nano_banana_does_not_run_fixed_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_plain_chat_no_runner.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    generated_path = tmp_path / "generated.png"
    generated_path.write_bytes(b"generated-image")
    output_path = tmp_path / "image_to_image.png"
    output_path.write_bytes(b"out")

    router = _make_router(response=AgentResponse(content="讲个冷笑话", model="test-model"))
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-plain-chat-no-runner",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_generated_image_path": str(generated_path),
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张猫图",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="讲个笑话",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "讲个冷笑话"
    assert bash_tool.commands == []


@pytest.mark.asyncio
async def test_run_agent_prompt_like_description_under_locked_nano_banana_runs_fixed_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_plain_description_no_runner.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = _make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-plain-description-no-runner",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张猫图",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="海贼王路飞跟娜美正在掰手腕",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "当前使用模型：香蕉2" in result
    assert str(output_path) in result
    assert len(bash_tool.commands) == 1
    assert "--mode text" in bash_tool.commands[0]


@pytest.mark.asyncio
async def test_run_agent_question_under_locked_nano_banana_does_not_run_fixed_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_question_no_runner.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = _make_router(
        response=AgentResponse(content="这是问句，不直接生图", model="test-model")
    )
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-question-no-runner",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张猫图",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="这张图现在是什么样的？",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "这是问句，不直接生图"
    assert bash_tool.commands == []


@pytest.mark.asyncio
async def test_run_agent_desktop_screenshot_request_under_locked_nano_banana_does_not_run_fixed_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["nanobanana"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True),
                SkillParamItem(key="prompt", type="text", required=True),
            ],
        ),
        source_path=Path("/tmp/nano_screenshot_no_runner.md"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = _make_router(response=AgentResponse(content="这是桌面截图请求", model="test-model"))
    registry = ToolRegistry()
    bash_tool = _NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-screenshot-no-runner",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "skill_param_state": {
                "nano-banana-image-t8": {
                    "api_key": "__present__",
                    "prompt": "生成一张猫图",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="将桌面截图给我",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert result == "这是桌面截图请求"
    assert bash_tool.commands == []


@pytest.mark.asyncio
async def test_run_agent_breaks_repeated_identical_tool_loop() -> None:
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
        return AgentResponse(
            content="",
            model="test-model",
            tool_calls=[
                ToolCall(
                    id=f"loop-{call_count}",
                    name="loop_tool",
                    arguments={"text": "same"},
                )
            ],
        )

    registry = ToolRegistry()
    registry.register(_LoopTool())
    router = _make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="执行循环任务",
        session_id="test-loop-repeat",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert "工具调用连续无效" in result
    assert call_count <= 6


@pytest.mark.asyncio
async def test_run_agent_blocks_repeated_same_search_query() -> None:
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
        return AgentResponse(
            content="",
            model="test-model",
            tool_calls=[
                ToolCall(
                    id=f"browser-{call_count}",
                    name="browser",
                    arguments={"action": "search_images", "text": "same query"},
                )
            ],
        )

    registry = ToolRegistry()
    registry.register(_BrowserProbeTool())
    router = _make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="给我搜图",
        session_id="test-search-images-loop",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert "工具调用连续无效" in result
    assert call_count <= 5


@pytest.mark.asyncio
async def test_run_agent_blocks_search_images_over_planned_count() -> None:
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
        return AgentResponse(
            content="共 2 张配图",
            model="test-model",
            tool_calls=[
                ToolCall(
                    id=f"browser-over-{call_count}",
                    name="browser",
                    arguments={"action": "search_images", "text": f"query {call_count}"},
                )
            ],
        )

    registry = ToolRegistry()
    registry.register(_BrowserProbeTool())
    router = _make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="做个带配图的PPT",
        session_id="test-search-images-over-plan",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert "工具调用连续无效" in result
    assert call_count <= 7
