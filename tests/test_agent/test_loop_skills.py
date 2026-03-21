"""Tests for agent loop skill-lock, memory injection, and PPT/office interactions."""
# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import whaleclaw.agent.single_agent as loop_mod
from tests.test_agent.loop_helpers import (
    BashProbeTool,
    DummyMemoryManager,
    PptEditBusinessNoHitTool,
    PptEditNoopTool,
    make_router,
)
from whaleclaw.agent.single_agent import run_agent
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, Message, ToolCall
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.agent.helpers.skill_helpers import (
    build_skill_param_guard_reply as _build_skill_param_guard_reply,
    capture_param_value as _capture_param_value,
    extract_api_key_value as _extract_api_key_value,
    persist_param_api_key as _persist_param_api_key,
)
from whaleclaw.skills.parser import Skill, SkillParamGuard, SkillParamItem
from whaleclaw.tools.registry import ToolRegistry


def _load_hooks(skill_id: str) -> object:
    import importlib.util, sys as _sys
    from whaleclaw.config.paths import WORKSPACE_DIR
    candidates = [
        WORKSPACE_DIR / "skills" / skill_id / "hooks.py",
        Path(__file__).resolve().parents[2] / "whaleclaw" / "skills" / "bundled" / skill_id / "hooks.py",
    ]
    _hooks_path = next((p for p in candidates if p.is_file()), None)
    if _hooks_path is None:
        return None
    _mod_name = f"hooks_{skill_id.replace('-', '_')}_skills_test"
    _spec = importlib.util.spec_from_file_location(_mod_name, str(_hooks_path))
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    _sys.modules[_mod_name] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    return _mod.Hooks()


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
            for t in tools:  # type: ignore[union-attr]
                tool_item = cast(Any, t)
                if hasattr(tool_item, "name"):
                    name = str(getattr(tool_item, "name", "")).strip()
                    if name:
                        captured_tool_names.append(name)
                    continue
                if isinstance(tool_item, dict):
                    name = str(tool_item.get("name", "")).strip()
                    if not name and isinstance(tool_item.get("function"), dict):
                        name = str(tool_item["function"].get("name", "")).strip()
                    if name:
                        captured_tool_names.append(name)
        return AgentResponse(content="收到", model="test-model")

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(BashProbeTool())
    registry.register(PptEditNoopTool())

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

    router = make_router(chat_fn=fake_chat)
    registry = ToolRegistry()
    registry.register(PptEditBusinessNoHitTool())

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

    router = make_router(chat_fn=fake_chat)
    memory: Any = DummyMemoryManager(recalled="- 用户喜欢简洁回答")
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
    router = make_router(response=AgentResponse(content="记住了", model="test-model"))
    memory: Any = DummyMemoryManager()
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
    class _NoRecallMemory(DummyMemoryManager):
        def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
            self.policy_calls += 1
            return (False, False)

    router = make_router(response=AgentResponse(content="ok", model="test-model"))
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
    class _NoRecallMemory(DummyMemoryManager):
        def recall_policy(self, query: str) -> tuple[bool, bool]:  # noqa: ARG002
            self.policy_calls += 1
            return (False, False)

    router = make_router(response=AgentResponse(content="已开始制作", model="test-model"))
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

    class _StyleMemory(DummyMemoryManager):
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

    router = make_router(chat_fn=fake_chat)
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

    class _StyleAwareMemory(DummyMemoryManager):
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

    router = make_router(chat_fn=fake_chat)
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

    router = make_router(chat_fn=fake_chat)
    memory: Any = DummyMemoryManager()
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

    router = make_router(chat_fn=fake_chat)
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

    router = make_router(chat_fn=fake_chat)
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
    registry.register(BashProbeTool())
    router = make_router(chat_fn=fake_chat)
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

    router2 = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    result = await run_agent(
        message="本轮结束啦",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "还没有完成正式解锁" in result
    assert "请直接回复\u201c任务完成\u201d或\u201c任务结束\u201d" in result
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

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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

        router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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

        router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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
    router = make_router(chat_fn=fake_chat)
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "banana-ref.png"
    image_path.write_bytes(b"png-bytes")
    seen_messages: list[Any] = []

    nb_skill = Skill(
        id="nano-banana-image-t8",
        name="Nano Banana 生图联调",
        triggers=["香蕉生图"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(key="api_key", type="api_key", required=True, prompt="请提供 API Key"),
                SkillParamItem(key="prompt", type="text", required=True, prompt="请提供提示词"),
            ],
        ),
        source_path=Path("/tmp/nb_system_prompt.md"),
        hooks=_load_hooks("nano-banana-image-t8"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [nb_skill],  # noqa: ARG005
    )

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
    router = make_router(chat_fn=fake_chat)
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
    assert "`nano-banana-2`" in joined
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
        hooks=_load_hooks("nano-banana-image-t8"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = make_router(response=AgentResponse(content="收到", model="test-model"))
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
        hooks=_load_hooks("nano-banana-image-t8"),
    )
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
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

    router = make_router(response=AgentResponse(content="收到", model="test-model"))
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
        hooks=_load_hooks("nano-banana-image-t8"),
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

    router = make_router(response=AgentResponse(content="收到", model="test-model"))
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

    assert "请先回复\u201c任务完成\u201d" in result
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
        hooks=_load_hooks("nano-banana-image-t8"),
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

    router = make_router(response=AgentResponse(content="不应直接执行", model="test-model"))
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
    assert "请先回复\u201c任务完成\u201d" in result
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

    router = make_router(response=AgentResponse(content="继续生图处理", model="test-model"))
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
        hooks=_load_hooks("nano-banana-image-t8"),
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

    router = make_router(response=AgentResponse(content="收到", model="test-model"))
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

    assert "请先回复\u201c任务完成\u201d" in result
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
        hooks=_load_hooks("nano-banana-image-t8"),
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

    router = make_router(chat_fn=fake_chat)
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

    assert "请先回复\u201c任务完成\u201d" in first
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
        hooks=_load_hooks("nano-banana-image-t8"),
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

    router = make_router(response=AgentResponse(content="收到", model="test-model"))
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

    assert "请先回复\u201c任务完成\u201d" in first
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
        hooks=_load_hooks("nano-banana-image-t8"),
    )

    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [browser_skill] if "截图" in user_message else [banana_skill],  # noqa: ARG005,E501
    )

    router = make_router(response=AgentResponse(content="截图成功", model="test-model"))
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

    router = make_router(response=AgentResponse(content="浏览器技能执行", model="test-model"))
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


# ── 小红书守卫测试 ─────────────────────────────────────────────


def _make_xhs_skill() -> Skill:
    return Skill(
        id="xiaohongshu_publish",
        name="小红书笔记发布",
        triggers=["小红书", "发笔记", "发布笔记", "小红书发布"],
        instructions="x",
        lock_session=True,
        is_user_installed=True,
        param_guard=SkillParamGuard(
            enabled=True,
            params=[
                SkillParamItem(
                    key="content",
                    label="发布内容",
                    type="text",
                    required=True,
                    prompt="请告诉我你要发布的内容或主题",
                    aliases=["内容", "文案", "文章", "正文", "主题"],
                ),
                SkillParamItem(
                    key="images",
                    label="配图",
                    type="images",
                    required=False,
                    prompt="可以发送图片给我，或者我帮你用 AI 生成配图",
                    min_count=1,
                ),
            ],
        ),
        source_path=Path("/tmp/xhs.md"),
        hooks=_load_hooks("xiaohongshu_publish"),
    )


@pytest.mark.asyncio
async def test_run_agent_xiaohongshu_guard_blocks_without_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'使用小红书技能' 未带内容时，守卫阻塞并提示。"""
    skill = _make_xhs_skill()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-xhs-guard-no-content",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    result = await run_agent(
        message="使用小红书技能",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router,
        session=session,
    )

    assert "小红书笔记发布" in result
    assert "发布内容：未提供" in result
    assert "自动生成" in result
    assert session.metadata.get("locked_skill_ids") == ["xiaohongshu_publish"]
    router.chat.assert_not_called()


@pytest.mark.asyncio
async def test_run_agent_xiaohongshu_guard_shows_content_then_confirms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'使用小红书技能发布xxx' → 展示主题等确认 → 用户确认 → 放行。"""
    skill = _make_xhs_skill()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router1 = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-xhs-guard-confirm",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    # 第一步：带内容触发，守卫展示主题等确认
    first = await run_agent(
        message="使用小红书技能发布一篇上海旅游的笔记",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router1,
        session=session,
    )
    assert "上海旅游" in first
    assert "确认" in first
    router1.chat.assert_not_called()

    # 第二步：用户确认，守卫放行
    router2 = make_router(response=AgentResponse(content="开始准备小红书内容", model="test-model"))
    second = await run_agent(
        message="开始",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router2,
        session=session,
    )
    assert "开始准备小红书内容" in second


@pytest.mark.asyncio
async def test_run_agent_xiaohongshu_guard_quick_publish_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用户选「快速发布」→ 守卫放行，state 标记 __quick_publish__。"""
    skill = _make_xhs_skill()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router1 = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-xhs-guard-quick",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    first = await run_agent(
        message="使用小红书技能发布三花猫游上海的笔记",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router1,
        session=session,
    )
    assert "三花猫" in first
    assert "快速发布" in first
    router1.chat.assert_not_called()

    router2 = make_router(response=AgentResponse(content="快速出图发布中", model="test-model"))
    second = await run_agent(
        message="快速发布",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router2,
        session=session,
    )
    assert "快速出图发布中" in second
    xhs_state = session.metadata.get("skill_param_state", {}).get("xiaohongshu_publish", {})
    assert xhs_state.get("__quick_publish__") is True


# ── 小红书守卫展示单元测试 ──────────────────────────────────────


def test_xiaohongshu_guard_reply_no_content_no_images() -> None:
    state: dict[str, object] = {}
    reply = _build_skill_param_guard_reply("xiaohongshu_publish", [], state, hooks=_load_hooks("xiaohongshu_publish"))
    assert "小红书笔记发布" in reply
    assert "发布内容：未提供" in reply
    assert "已收到 0 张" in reply
    assert "自动生成" in reply
    assert "请告诉我你要发布的内容或主题" in reply


def test_xiaohongshu_guard_reply_with_content_shows_two_modes() -> None:
    state: dict[str, object] = {"content": "上海旅行攻略"}
    reply = _build_skill_param_guard_reply("xiaohongshu_publish", [], state, hooks=_load_hooks("xiaohongshu_publish"))
    assert "上海旅行攻略" in reply
    assert "开始" in reply
    assert "快速发布" in reply
    assert "自动生成" in reply


def test_xiaohongshu_guard_reply_with_content_and_images() -> None:
    state: dict[str, object] = {"content": "上海旅行攻略", "images": 2}
    reply = _build_skill_param_guard_reply("xiaohongshu_publish", [], state, hooks=_load_hooks("xiaohongshu_publish"))
    assert "上海旅行攻略" in reply
    assert "已收到 2 张" in reply
    assert "自动生成" not in reply
    assert "快速发布" in reply


@pytest.mark.asyncio
async def test_run_agent_xiaohongshu_confirm_does_not_overwrite_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用户回复'开始'时，content 不应被覆盖为'开始'。"""
    skill = _make_xhs_skill()
    monkeypatch.setattr(
        loop_mod._assembler,  # noqa: SLF001
        "route_skills",
        lambda user_message, forced_skill_ids=None: [skill],  # noqa: ARG005
    )

    router1 = make_router(response=AgentResponse(content="不应调用", model="test-model"))
    now = datetime.now(UTC)
    session = Session(
        id="s-xhs-no-overwrite",
        channel="webchat",
        peer_id="u1",
        messages=[],
        model="openai/gpt-5.2",
        created_at=now,
        updated_at=now,
        metadata={},
    )

    first = await run_agent(
        message="使用小红书技能发布三花猫的上海游",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router1,
        session=session,
    )
    assert "三花猫" in first
    router1.chat.assert_not_called()

    xhs_state = session.metadata.get("skill_param_state", {}).get("xiaohongshu_publish", {})
    captured_content = str(xhs_state.get("content", ""))
    assert "三花猫" in captured_content

    router2 = make_router(response=AgentResponse(content="好的开始执行", model="test-model"))
    second = await run_agent(
        message="开始",
        session_id=session.id,
        config=WhaleclawConfig(),
        router=router2,
        session=session,
    )
    assert "好的开始执行" in second

    xhs_state2 = session.metadata.get("skill_param_state", {}).get("xiaohongshu_publish", {})
    assert xhs_state2.get("content") == captured_content, "确认词不应覆盖原有 content"
    assert xhs_state2.get("__content_confirmed__") is True


@pytest.mark.parametrize(
    "confirm_word",
    ["开始", "确认", "ok", "好的", "好", "可以", "没问题", "就这样", "行", "发吧",
     "快速发布", "直接发", "发布", "任务完成", "取消"],
)
def test_capture_param_value_ignores_confirm_words_for_text(confirm_word: str) -> None:
    """确认/控制词不应被 capture_param_value 当作 text 内容。"""
    param = SkillParamItem(
        key="content", label="发布内容", type="text", required=True,
        prompt="请告诉我内容", aliases=["内容", "文案"],
    )
    result = _capture_param_value(param, confirm_word, None, "原有主题")
    assert result == "原有主题", f"'{confirm_word}' 不应覆盖已有 content"


def test_capture_param_value_strips_image_noise_from_text() -> None:
    """图片 markdown 和 '(用户发送了图片)' 不应污染 content 值。"""
    param = SkillParamItem(
        key="content", label="发布内容", type="text", required=True,
        prompt="请告诉我内容", aliases=["内容", "文案"],
    )
    noisy = "三花猫逛魔都\n\n(用户发送了图片)\n![飞书图片1](/tmp/img.png)"
    result = _capture_param_value(param, noisy, None, None)
    assert result == "三花猫逛魔都"


def test_xiaohongshu_guard_reply_strips_image_noise_from_display() -> None:
    """守卫展示 content 时，不应显示图片标记。"""
    state: dict[str, object] = {
        "content": "三花猫逛魔都\n(用户发送了图片)\n![飞书图片1](/tmp/img.png)",
        "images": 1,
    }
    reply = _build_skill_param_guard_reply("xiaohongshu_publish", [], state, hooks=_load_hooks("xiaohongshu_publish"))
    assert "三花猫逛魔都" in reply
    assert "用户发送了图片" not in reply
    assert "![" not in reply


@pytest.mark.parametrize("text", [
    "sk-vM87iLnMCjayIhl75dBa817706C545EeBd604e186d26022d",
    "更换APIkey sk-vM87iLnMCjayIhl75dBa817706C545EeBd604e186d26022d",
    "更换apikey sk-abc123def456ghi789",
    "API Key: sk-testkey1234567890abc",
])
def test_capture_param_value_text_ignores_api_key_messages(text: str) -> None:
    """包含 sk- API key 的消息不应被当作 prompt/content 捕获。"""
    param = SkillParamItem(
        key="prompt", label="提示词", type="text", required=True,
        prompt="请提供提示词", aliases=["提示词"],
    )
    result = _capture_param_value(param, text, None, "原有提示词")
    assert result == "原有提示词", f"消息 '{text}' 不应覆盖已有 prompt"


def test_capture_param_value_text_ignores_bare_api_key() -> None:
    """纯 sk- 字符串不应被当作 prompt 捕获（即使之前无值）。"""
    param = SkillParamItem(
        key="prompt", label="提示词", type="text", required=True,
        prompt="请提供提示词", aliases=["提示词"],
    )
    result = _capture_param_value(
        param, "sk-vM87iLnMCjayIhl75dBa817706C545EeBd604e186d26022d", None, None,
    )
    assert result is None


def test_capture_param_value_api_key_detects_bare_key() -> None:
    """纯 sk- 字符串应被 api_key 类型正确捕获。"""
    param = SkillParamItem(
        key="api_key", label="API Key", type="api_key", required=True,
        prompt="请提供 API Key", aliases=["apikey"],
    )
    result = _capture_param_value(
        param, "sk-vM87iLnMCjayIhl75dBa817706C545EeBd604e186d26022d", None, None,
    )
    assert result == "__present__"


@pytest.mark.parametrize("text,expected", [
    ("sk-abc123def456ghi789", "sk-abc123def456ghi789"),
    ("更换APIkey sk-newKey12345678", "sk-newKey12345678"),
    ("没有key的消息", ""),
    ("画一只猫", ""),
])
def test_extract_api_key_value(text: str, expected: str) -> None:
    assert _extract_api_key_value(text) == expected


def test_persist_param_api_key_writes_new_key(tmp_path: Path) -> None:
    """用户发送新 key 时应立即写入 saved_file。"""
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("sk-oldKeyOldKeyOldKey123", encoding="utf-8")
    params = [
        SkillParamItem(
            key="api_key", label="API Key", type="api_key", required=True,
            prompt="请提供 API Key", aliases=["apikey"],
            saved_file=str(key_file),
        ),
    ]
    new_key = "sk-brandNewKey9876543210abc"
    result = _persist_param_api_key(params, f"更换APIkey {new_key}")
    assert result is True
    assert key_file.read_text(encoding="utf-8") == new_key


def test_persist_param_api_key_skips_when_same(tmp_path: Path) -> None:
    """key 未变化时不应重复写入。"""
    key_file = tmp_path / "api_key.txt"
    existing = "sk-sameKeySameKeySameKey12"
    key_file.write_text(existing, encoding="utf-8")
    params = [
        SkillParamItem(
            key="api_key", label="API Key", type="api_key", required=True,
            prompt="请提供 API Key", aliases=["apikey"],
            saved_file=str(key_file),
        ),
    ]
    result = _persist_param_api_key(params, existing)
    assert result is False


def test_persist_param_api_key_no_key_in_message(tmp_path: Path) -> None:
    """消息中没有 key 时不应写入。"""
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("sk-originalKeyOriginal12", encoding="utf-8")
    params = [
        SkillParamItem(
            key="api_key", label="API Key", type="api_key", required=True,
            prompt="请提供 API Key", aliases=["apikey"],
            saved_file=str(key_file),
        ),
    ]
    result = _persist_param_api_key(params, "画一只猫")
    assert result is False
    assert key_file.read_text(encoding="utf-8") == "sk-originalKeyOriginal12"
