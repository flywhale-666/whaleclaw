"""Shared helpers and dummy tools for agent loop tests."""

from __future__ import annotations

import asyncio
import re as _re
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from whaleclaw.providers.base import AgentResponse, ToolCall
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult


def make_router(
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


# ── Dummy Tool 类 ─────────────────────────────────────────────


class EchoTool(Tool):
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


class LoopTool(Tool):
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


class BrowserProbeTool(Tool):
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


class BrowserAlwaysFailTool(Tool):
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


class BashProbeTool(Tool):
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


class DelayedNanoBananaBashTool(Tool):
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


class DesktopCaptureNoopTool(Tool):
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


class BashAlwaysFailTool(Tool):
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


class BashPyScriptRetryTool(Tool):
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


class NanoBananaFixedRunnerTool(Tool):
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


def make_nano_banana_router(
    *,
    model_display: str = "香蕉2",
    output_path: Path | None = None,
    final_content: str | None = None,
) -> MagicMock:
    """构建 nano-banana 测试用的 router mock。"""
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
            # 从后往前搜索，优先匹配 nano-banana 执行模板（靠后的 system message）
            for m in reversed(messages):
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

    return make_router(chat_fn=fake_chat)


class PptEditNoopTool(Tool):
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


class PptEditBusinessNoHitTool(Tool):
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


class NameMemoryManager:
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


class DummyMemoryManager:
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
