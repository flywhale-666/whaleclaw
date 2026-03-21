"""Tests for agent loop guard / circuit-breaker mechanisms."""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_agent.loop_helpers import (
    BashAlwaysFailTool,
    BrowserAlwaysFailTool,
    BrowserProbeTool,
    LoopTool,
    make_router,
)
from whaleclaw.agent.helpers.tool_guards import (
    BASH_BLOCK_LIMIT,
    BASH_WARN_LIMIT,
    BROWSER_BLOCK_LIMIT,
    BROWSER_WARN_LIMIT,
    REPEAT_ABORT_ROUNDS,
    REPEAT_BLOCK_ROUNDS,
    REPEAT_WARN_ROUNDS,
    ToolGuardState,
    apply_tool_result_guards,
)
from whaleclaw.agent.loop import run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, ToolCall
from whaleclaw.tools.base import ToolResult
from whaleclaw.tools.registry import ToolRegistry


def _nb_hooks() -> list[object]:
    import importlib.util
    import sys
    from pathlib import Path
    from whaleclaw.config.paths import WORKSPACE_DIR
    key = "nb_hooks_guards"
    if key in sys.modules:
        return [sys.modules[key].Hooks()]
    candidates = [
        WORKSPACE_DIR / "skills" / "nano-banana-image-t8" / "hooks.py",
        Path("whaleclaw/skills/bundled/nano-banana-image-t8/hooks.py"),
    ]
    hooks_path = next((p for p in candidates if p.is_file()), None)
    if hooks_path is None:
        raise FileNotFoundError("nano-banana hooks.py not found")
    spec = importlib.util.spec_from_file_location(key, str(hooks_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return [mod.Hooks()]


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
        if call_count <= 3:
            return browser_tool_response
        return final_response

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(BrowserAlwaysFailTool())

    result = await run_agent(
        message="给我张杨幂近照",
        session_id="test-browser-circuit",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert result == "改用 bash 处理"
    assert call_count == 4
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

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(BashAlwaysFailTool())

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
    registry.register(LoopTool())
    router = make_router(chat_fn=fake_chat)

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
    registry.register(BrowserProbeTool())
    router = make_router(chat_fn=fake_chat)

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
    registry.register(BrowserProbeTool())
    router = make_router(chat_fn=fake_chat)

    result = await run_agent(
        message="做个带配图的PPT",
        session_id="test-search-images-over-plan",
        config=WhaleclawConfig(),
        router=router,
        registry=registry,
    )

    assert "工具调用连续无效" in result
    assert call_count <= 7


# ── Guard 单元测试 ─────────────────────────────────────────────


def test_browser_failure_warn_then_block() -> None:
    """browser 连续失败：2 次 warn，3 次 block。"""
    state = ToolGuardState()
    tc = ToolCall(id="tc1", name="browser", arguments={"action": "search_images", "text": "x"})
    fail = ToolResult(success=False, output="", error="browser failed")

    upd = None
    for _ in range(BROWSER_WARN_LIMIT):
        upd = apply_tool_result_guards(state, tc, fail, session_id="test")
    assert upd is not None
    warn_decisions = [d for d in upd.decisions if d.kind == "warn"]
    assert len(warn_decisions) >= 1

    upd = apply_tool_result_guards(state, tc, fail, session_id="test")
    block_decisions = [d for d in upd.decisions if d.kind == "block"]
    assert len(block_decisions) >= 1
    assert any("browser" in d.blocked_tools for d in block_decisions)


def test_bash_same_template_warn_then_block() -> None:
    """bash 同模板连续失败：2 次 warn，3 次 block。"""
    state = ToolGuardState()
    tc = ToolCall(id="tc1", name="bash", arguments={"command": "python3 /tmp/a.py"})
    fail = ToolResult(success=False, output="", error="bash failed")

    upd = None
    for _ in range(BASH_WARN_LIMIT):
        upd = apply_tool_result_guards(state, tc, fail, session_id="test")
    assert upd is not None
    warn_decisions = [d for d in upd.decisions if d.kind == "warn"]
    assert len(warn_decisions) >= 1

    upd = apply_tool_result_guards(state, tc, fail, session_id="test")
    block_decisions = [d for d in upd.decisions if d.kind == "block"]
    assert len(block_decisions) >= 1
    assert any("bash" in d.blocked_tools for d in block_decisions)


def test_nano_banana_preflight_produces_hint_decision() -> None:
    """Nano Banana preflight 错误（直接执行 .py 脚本）产生 hint 决策，允许重试。"""
    state = ToolGuardState()
    tc = ToolCall(
        id="tc1",
        name="bash",
        arguments={"command": "/tmp/test_nano_banana_2.py --mode edit"},
    )
    result = ToolResult(
        success=False,
        output="[stderr]\nfrom: command not found\nimport: command not found\n[exit_code: 127]",
        error="from: command not found\nimport: command not found",
    )

    nb = _nb_hooks()

    upd = apply_tool_result_guards(
        state, tc, result, session_id="test", skill_hooks=nb,
    )
    hints = [d for d in upd.decisions if d.kind == "hint"]
    assert len(hints) >= 1
    assert hints[0].reason_code == "nano_banana_preflight_allow_retry"


def test_threshold_constants_are_consistent() -> None:
    """确保 warn < block 且 repeat warn < block < abort。"""
    assert BROWSER_WARN_LIMIT < BROWSER_BLOCK_LIMIT
    assert BASH_WARN_LIMIT < BASH_BLOCK_LIMIT
    assert REPEAT_WARN_ROUNDS < REPEAT_BLOCK_ROUNDS < REPEAT_ABORT_ROUNDS
