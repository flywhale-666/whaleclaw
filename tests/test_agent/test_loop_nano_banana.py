"""Tests for Nano Banana image generation in agent loop."""
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, cast

import pytest

import whaleclaw.agent.single_agent as loop_mod
from tests.test_agent.loop_helpers import (
    BashPyScriptRetryTool,
    DelayedNanoBananaBashTool,
    NanoBananaFixedRunnerTool,
    make_nano_banana_router,
    make_router,
)
from whaleclaw.agent.single_agent import run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, Message, ToolCall
from whaleclaw.sessions.manager import Session
from whaleclaw.skills.parser import Skill, SkillParamGuard, SkillParamItem
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.tools.registry import ToolRegistry


def _load_nb_module() -> Any:
    """加载 nano-banana hooks 模块（缓存到 sys.modules）。"""
    import importlib.util, sys as _sys
    from whaleclaw.config.paths import WORKSPACE_DIR
    key = "nb_hooks_nano_test"
    if key in _sys.modules:
        return _sys.modules[key]
    candidates = [
        WORKSPACE_DIR / "skills" / "nano-banana-image-t8" / "hooks.py",
        Path(__file__).resolve().parents[2] / "whaleclaw" / "skills" / "bundled" / "nano-banana-image-t8" / "hooks.py",
    ]
    hooks_path = next((p for p in candidates if p.is_file()), None)
    if hooks_path is None:
        raise FileNotFoundError("nano-banana hooks.py not found")
    _spec = importlib.util.spec_from_file_location(key, str(hooks_path))
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    _sys.modules[key] = _mod
    try:
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    except Exception:
        _sys.modules.pop(key, None)
        raise
    return _mod


def _load_nb_hooks() -> object:
    from whaleclaw.skills.hooks import register_hooks
    _mod = _load_nb_module()
    instance = _mod.Hooks()
    register_hooks(instance)
    return instance


class _BrowserCaptureTool(Tool):
    """Dummy browser tool that captures text argument for repair tests."""

    def __init__(self, captured: list[str]) -> None:
        self._captured = captured

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
        self._captured.append(str(kwargs.get("text", "")))
        return ToolResult(success=True, output="ok")


class _FileEditProbeTool(Tool):
    """Dummy file_edit tool for escaped block args rejection tests."""

    def __init__(self, tool_called: list[bool]) -> None:
        self._tool_called = tool_called

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

    async def execute(self, **kwargs: Any) -> ToolResult:
        self._tool_called[0] = True
        return ToolResult(success=True, output="edited")


class _NanoBananaRuntimeFailTool(Tool):
    """Dummy bash tool that simulates a runtime nano-banana network failure."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Always fails like the remote nano-banana API.",
            parameters=[ToolParameter(name="command", type="string", description="command")],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command", "")).strip()
        self.commands.append(command)
        return ToolResult(
            success=False,
            output=(
                "[bash] [ERROR] 请求失败: RemoteProtocolError: "
                "Server disconnected without sending a response."
            ),
            error="请求失败: RemoteProtocolError: Server disconnected without sending a response.",
        )


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

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    bash_tool = BashPyScriptRetryTool()
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
    _load_nb_hooks()
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

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    bash_tool = DelayedNanoBananaBashTool()
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_BrowserCaptureTool(captured))

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
    tool_called = [False]

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

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(_FileEditProbeTool(tool_called))

    result = await run_agent(
        message="重做这个 python 脚本",
        session_id="test-file-edit-escaped-block",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "我改用 file_write 重写脚本"
    assert call_count == 2
    assert not tool_called[0]


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
        hooks=_load_nb_hooks(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
async def test_nano_banana_guard_text_prompt_does_not_require_image() -> None:
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
        source_path=Path("/tmp/nano_guard_text_no_image.md"),
        hooks=_load_nb_hooks(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-guard-text-no-image",
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
                    "prompt": "一只霸王龙在洗澡",
                    "__model_display__": "香蕉2",
                }
            },
        },
    )

    result = await run_agent(
        message="一只霸王龙在洗澡",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "请上传图片" not in result
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
    assert "请回复" in result and "任务完成" in result and "解除技能锁定" in result
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = make_router(response=AgentResponse(content="继续执行", model="test-model"))
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
    assert "请回复" in result and "任务完成" in result and "解除技能锁定" in result
    router.chat.assert_not_called()
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_nano_banana_activation_message_prefers_last_used_model_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = make_router()
    model_file = Path.home() / ".whaleclaw" / "credentials" / "nano_banana_default_model.txt"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    _had_file = model_file.is_file()
    _old_content = model_file.read_text(encoding="utf-8") if _had_file else None
    model_file.write_text("香蕉pro", encoding="utf-8")

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

    if _old_content is not None:
        model_file.write_text(_old_content, encoding="utf-8")
    elif model_file.is_file():
        model_file.unlink()

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

    router = make_router()
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
        hooks=_load_nb_hooks(),
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

    router = make_router(chat_fn=fake_chat)
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
        hooks=_load_nb_hooks(),
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

    router = make_router(chat_fn=fake_chat)
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
        hooks=_load_nb_hooks(),
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

    router = make_router(chat_fn=fake_chat)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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

    nb_mod = _load_nb_module()
    resolved = nb_mod.resolve_input_paths(
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

    nb_mod = _load_nb_module()
    resolved = nb_mod.resolve_input_paths(
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
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

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = make_nano_banana_router(model_display="香蕉pro", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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

    _result = await run_agent(
        message="用香蕉pro生成一张海报",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert len(bash_tool.commands) == 1
    assert "nano-banana-2" in bash_tool.commands[0]
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")

    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
async def test_nano_banana_generation_prefers_skill_state_model_over_stale_last_model(
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
        source_path=Path("/tmp/nano_generation_prefers_skill_state_model.md"),
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = make_nano_banana_router(model_display="香蕉pro", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
    registry.register(bash_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-generation-prefers-skill-state-model",
        channel="feishu",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={
            "locked_skill_ids": ["nano-banana-image-t8"],
            "last_nano_banana_model_display": "香蕉2",
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
        message="霸王龙在唱歌",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert "![结果图]" in result
    assert len(bash_tool.commands) == 1
    assert "--model nano-banana-2" in bash_tool.commands[0]
    assert "--edit-model nano-banana-2" in bash_tool.commands[0]


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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
async def test_nano_banana_default_model_switch_persists_model_file(
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
        source_path=Path("/tmp/nano_default_model_switch.md"),
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    nb_mod = _load_nb_module()
    default_model_file = tmp_path / "nano_banana_default_model.txt"
    monkeypatch.setattr(nb_mod, "_DEFAULT_MODEL_FILE", default_model_file)

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-nano-default-model-switch",
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
        message="默认模型香蕉pro",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "已设置默认生图模型为：香蕉pro" in result
    assert default_model_file.read_text(encoding="utf-8") == "nano-banana-2"
    assert (
        cast(dict[str, object], session.metadata["skill_param_state"]["nano-banana-image-t8"])[
            "__model_display__"
        ]
        == "香蕉pro"
    )
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_nano_banana_runtime_failure_does_not_trigger_second_bash_call(
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
        source_path=Path("/tmp/nano_runtime_fail_no_retry.md"),
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    cmd_re = re.compile(r"```\n(.+?)\n```", re.DOTALL)
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
        command = ""
        for message in reversed(messages):
            text = message.content if hasattr(message, "content") else str(message)
            match = cmd_re.search(text)
            if match:
                command = match.group(1).strip()
                break
        if call_count <= 2:
            return AgentResponse(
                content="",
                model="test-model",
                tool_calls=[ToolCall(id=f"tc_nb_fail_{call_count}", name="bash", arguments={"command": command})],
            )
        return AgentResponse(content="不应走到这里", model="test-model")

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    fail_tool = _NanoBananaRuntimeFailTool()
    registry.register(fail_tool)

    now = datetime.now(UTC)
    session = Session(
        id="s-nano-runtime-fail-no-retry",
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
                    "prompt": "一只霸王龙在洗澡",
                    "__model_display__": "香蕉pro",
                }
            },
        },
    )

    result = await run_agent(
        message="一只霸王龙在洗澡",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
        session=session,
    )

    assert len(fail_tool.commands) == 1
    assert "不会自动切换地址或重试" in result
    assert "RemoteProtocolError" in result


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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )
    _model_file = Path.home() / ".whaleclaw" / "credentials" / "nano_banana_default_model.txt"
    _model_file.parent.mkdir(parents=True, exist_ok=True)
    _had_model = _model_file.is_file()
    _old_model = _model_file.read_text(encoding="utf-8") if _had_model else None
    _model_file.write_text("香蕉2", encoding="utf-8")

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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

    if _old_model is not None:
        _model_file.write_text(_old_model, encoding="utf-8")
    elif _model_file.is_file():
        _model_file.unlink()

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
        hooks=_load_nb_hooks(),
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

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
        hooks=_load_nb_hooks(),
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

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
        hooks=_load_nb_hooks(),
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

    router = make_router(chat_fn=fake_chat)
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
        hooks=_load_nb_hooks(),
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

    router = make_router(chat_fn=fake_chat)
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
        hooks=_load_nb_hooks(),
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

    router = make_router(response=AgentResponse(content="讲个冷笑话", model="test-model"))
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = make_nano_banana_router(model_display="香蕉2", output_path=output_path)
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = make_router(
        response=AgentResponse(content="这是问句，不直接生图", model="test-model")
    )
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
        hooks=_load_nb_hooks(),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill] if forced_skill_ids else [],  # noqa: ARG005
    )

    output_path = tmp_path / "text_to_image.png"
    output_path.write_bytes(b"out")
    router = make_router(response=AgentResponse(content="这是桌面截图请求", model="test-model"))
    registry = ToolRegistry()
    bash_tool = NanoBananaFixedRunnerTool(output_path)
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
