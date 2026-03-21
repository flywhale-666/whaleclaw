"""Tests for session group compressor behavior."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from whaleclaw.providers.base import ImageContent, Message
from whaleclaw.sessions.group_compressor import (
    SessionGroupCompressor,
    _compact_prev_group,  # pyright: ignore[reportPrivateUsage]
    _hash_group,  # pyright: ignore[reportPrivateUsage]
)
from whaleclaw.sessions.store import SessionStore


def _mk_group(i: int, text: str) -> list[Message]:
    return [
        Message(role="user", content=f"u{i}:{text}"),
        Message(role="assistant", content=f"a{i}:{text}"),
    ]


def _flatten(groups: list[list[Message]]) -> list[Message]:
    out: list[Message] = []
    for g in groups:
        out.extend(g)
    return out


class _NoopRouter:
    async def chat(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("chat should not be called when model_id is empty")


class _SlowRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        await asyncio.sleep(0.15)
        return SimpleNamespace(content="压缩摘要")


async def _mk_store(tmp_path: Path) -> SessionStore:
    store = SessionStore(db_path=tmp_path / "group_compressor.db")
    await store.open()
    return store


@pytest.mark.asyncio
async def test_window_plan_uses_absolute_group_index(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        groups = [_mk_group(i, "短消息") for i in range(1, 31)]
        plan = compressor._window_plan(_flatten(groups))  # pyright: ignore[reportPrivateUsage]
        assert len(plan) == 25
        assert plan[0].group_idx == 6
        assert plan[-1].group_idx == 30
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_schedules_background_generation(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    compressor = SessionGroupCompressor(store)
    try:
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s2",
            channel="webchat",
            peer_id="u2",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "需要压缩的历史消息 " + ("x" * 120)) for i in range(1, 13)]
        router = _SlowRouter()

        t0 = time.monotonic()
        output = await compressor.build_window_messages(
            session_id="s2",
            messages=_flatten(groups),
            router=router,  # type: ignore[arg-type]
            model_id="compress-model",
        )
        elapsed = time.monotonic() - t0

        assert elapsed < 0.2
        assert output

        plan = [item for item in compressor._window_plan(_flatten(groups)) if item.level != "L2"]  # pyright: ignore[reportPrivateUsage]
        found = False
        for _ in range(20):
            for item in plan:
                cached = await store.get_group_compression(
                    session_id="s2",
                    group_idx=item.group_idx,
                    level=item.level,
                    source_hash=_hash_group(item.group),
                )
                if cached:
                    found = True
                    break
            if found:
                break
            await asyncio.sleep(0.05)

        assert found
        assert router.calls > 0
    finally:
        await compressor.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_prewarm_session_resumes_only_pending_jobs(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    compressor = SessionGroupCompressor(store)
    try:
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-pending",
            channel="webchat",
            peer_id="u-pending",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
            metadata={},
        )
        groups = [_mk_group(i, "需要压缩的历史消息 " + ("x" * 80)) for i in range(1, 25 + 1)]
        plan = compressor._window_plan(_flatten(groups))  # pyright: ignore[reportPrivateUsage]
        compressed = [item for item in plan if item.level != "L2"]
        pending_items = compressed[:6]
        await store.update_session_field(
            "s-pending",
            metadata={
                "group_compression_plan_levels": {
                    str(item.group_idx): item.level for item in plan
                },
                "group_compression_pending": [
                    {"group_idx": item.group_idx, "level": item.level} for item in pending_items
                ],
            },
        )

        router = _SlowRouter()
        stats = await compressor.prewarm_session(
            session_id="s-pending",
            messages=_flatten(groups),
            router=router,  # type: ignore[arg-type]
            model_id="compress-model",
        )

        session = await store.get_session("s-pending")
        pending_after: object = session.metadata.get("group_compression_pending", []) if session else []
        assert stats["total_groups"] == 6
        assert stats["processed_groups"] == 6
        assert router.calls == 6
        assert pending_after == []
    finally:
        await compressor.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_enqueues_only_shifted_groups(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    compressor = SessionGroupCompressor(store)
    try:
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-shift",
            channel="webchat",
            peer_id="u-shift",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
            metadata={},
        )
        old_groups = [_mk_group(i, "历史消息 " + ("x" * 80)) for i in range(1, 25 + 1)]
        old_plan = compressor._window_plan(_flatten(old_groups))  # pyright: ignore[reportPrivateUsage]
        await store.update_session_field(
            "s-shift",
            metadata={
                "group_compression_plan_levels": {
                    str(item.group_idx): item.level for item in old_plan
                },
                "group_compression_pending": [],
            },
        )

        new_groups = [_mk_group(i, "历史消息 " + ("x" * 80)) for i in range(1, 29 + 1)]
        router = _SlowRouter()
        await compressor.build_window_messages(
            session_id="s-shift",
            messages=_flatten(new_groups),
            router=router,  # type: ignore[arg-type]
            model_id="compress-model",
        )

        session = await store.get_session("s-shift")
        assert session is not None
        raw_pending = session.metadata.get("group_compression_pending", [])
        assert isinstance(raw_pending, list)
        pending = cast(list[dict[str, object]], raw_pending)
        pending_pairs: set[tuple[int, str]] = set()
        for item in pending:
            pending_pairs.add((int(item["group_idx"]), str(item["level"])))  # type: ignore[arg-type]
        assert len(pending_pairs) == 8
        assert pending_pairs == {
            (14, "L0"),
            (15, "L0"),
            (16, "L0"),
            (17, "L0"),
            (21, "L1"),
            (22, "L1"),
            (23, "L1"),
            (24, "L1"),
        }
    finally:
        await compressor.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_prewarm_session_skips_already_cached_shifted_groups(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    compressor = SessionGroupCompressor(store)
    try:
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-cached-shift",
            channel="webchat",
            peer_id="u-cached-shift",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
            metadata={},
        )
        old_groups = [_mk_group(i, "历史消息 " + ("x" * 80)) for i in range(1, 25 + 1)]
        old_plan = compressor._window_plan(_flatten(old_groups))  # pyright: ignore[reportPrivateUsage]
        await store.update_session_field(
            "s-cached-shift",
            metadata={
                "group_compression_plan_levels": {
                    str(item.group_idx): item.level for item in old_plan
                },
                "group_compression_pending": [],
            },
        )

        new_groups = [_mk_group(i, "历史消息 " + ("x" * 80)) for i in range(1, 29 + 1)]
        new_plan = compressor._window_plan(_flatten(new_groups))  # pyright: ignore[reportPrivateUsage]
        shifted_items = [
            item
            for item in new_plan
            if item.level != "L2"
            and next(
                (old.level for old in old_plan if old.group_idx == item.group_idx),
                None,
            ) != item.level
        ]
        assert shifted_items
        for item in shifted_items:
            await store.upsert_group_compression(
                session_id="s-cached-shift",
                group_idx=item.group_idx,
                level=item.level,
                source_hash=_hash_group(item.group),
                content=f"cached-{item.group_idx}-{item.level}",
            )

        router = _SlowRouter()
        stats = await compressor.prewarm_session(
            session_id="s-cached-shift",
            messages=_flatten(new_groups),
            router=router,  # type: ignore[arg-type]
            model_id="compress-model",
        )

        session = await store.get_session("s-cached-shift")
        assert session is not None
        assert session.metadata.get("group_compression_pending", []) == []
        assert stats["total_groups"] == 0
        assert stats["processed_groups"] == 0
        assert router.calls == 0
    finally:
        await compressor.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_reschedules_existing_pending_jobs(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    compressor = SessionGroupCompressor(store)
    try:
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-retry-pending",
            channel="webchat",
            peer_id="u-retry-pending",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
            metadata={},
        )
        groups = [_mk_group(i, "需要压缩的历史消息 " + ("x" * 80)) for i in range(1, 25 + 1)]
        plan = compressor._window_plan(_flatten(groups))  # pyright: ignore[reportPrivateUsage]
        compressed = [item for item in plan if item.level != "L2"]
        pending_items = compressed[:2]
        await store.update_session_field(
            "s-retry-pending",
            metadata={
                "group_compression_plan_levels": {
                    str(item.group_idx): item.level for item in plan
                },
                "group_compression_pending": [
                    {"group_idx": item.group_idx, "level": item.level} for item in pending_items
                ],
            },
        )

        router = _SlowRouter()
        await compressor.build_window_messages(
            session_id="s-retry-pending",
            messages=_flatten(groups),
            router=router,  # type: ignore[arg-type]
            model_id="compress-model",
        )
        pending_after: object = [{"group_idx": -1, "level": "L0"}]
        for _ in range(20):
            session = await store.get_session("s-retry-pending")
            pending_after = session.metadata.get("group_compression_pending", []) if session else []
            if pending_after == []:
                break
            await asyncio.sleep(0.05)
        assert router.calls == 2
        assert pending_after == []
    finally:
        await compressor.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_recent_over_budget_keeps_latest_two_raw_and_downgrades_3_to_5_to_l0(
    tmp_path: Path,
) -> None:
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        groups = [
            _mk_group(1, "旧历史"),
            _mk_group(2, "最近第5组 " + ("超长内容 " * 600)),
            _mk_group(3, "最近第4组 " + ("超长内容 " * 600)),
            _mk_group(4, "最近第3组 " + ("超长内容 " * 600)),
            _mk_group(5, "最近第2组 " + ("超长内容 " * 600)),
            _mk_group(6, "最近第1组 " + ("超长内容 " * 600)),
        ]
        plan = compressor._window_plan(_flatten(groups))  # pyright: ignore[reportPrivateUsage]
        assert [x.level for x in plan][-5:] == ["L0", "L0", "L0", "L2", "L2"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_window_plan_compresses_20_groups_when_25_groups_present(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        groups = [_mk_group(i, "消息") for i in range(1, 25 + 1)]
        plan = compressor._window_plan(_flatten(groups))  # pyright: ignore[reportPrivateUsage]
        l2 = sum(1 for x in plan if x.level == "L2")
        l1 = sum(1 for x in plan if x.level == "L1")
        l0 = sum(1 for x in plan if x.level == "L0")
        assert l2 == 5
        assert l1 == 7
        assert l0 == 13
        assert l1 + l0 == 20
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_outputs_structured_blocks(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s1",
            channel="webchat",
            peer_id="u1",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "历史消息") for i in range(1, 6)]
        groups.append([Message(role="user", content="u6:当前轮用户请求")])
        output = await compressor.build_window_messages(
            session_id="s1",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        text = "\n".join(m.content for m in output)
        assert "【历史摘要" in text
        assert "【当前任务状态】" in text
        assert any(m.role == "user" and "u6:当前轮用户请求" in m.content for m in output)
        for i in range(2, 6):
            assert any(m.role == "user" and f"u{i}:历史消息" in m.content for m in output)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_build_window_messages_keeps_current_plus_previous_four_raw_groups(
    tmp_path: Path,
) -> None:
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s3",
            channel="webchat",
            peer_id="u3",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "历史消息") for i in range(1, 8)]
        groups.append([Message(role="user", content="u8:当前轮用户请求")])
        output = await compressor.build_window_messages(
            session_id="s3",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        history_blocks = [m for m in output if "【历史摘要" in m.content]
        assert len(history_blocks) == 1
        history_text = history_blocks[0].content
        assert "第3轮:" in history_text or "第2轮:" in history_text

        for i in range(4, 8):
            assert any(m.role == "user" and f"u{i}:历史消息" in m.content for m in output)
        assert any(m.role == "user" and "u8:当前轮用户请求" in m.content for m in output)
        assert not any(m.role == "user" and "u3:历史消息" in m.content for m in output)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_task_status_block_includes_current_progress_and_next_step(tmp_path: Path) -> None:
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s4",
            channel="webchat",
            peer_id="u4",
            model="qwen/qwen3.5-plus",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "历史消息") for i in range(1, 5)]
        groups.append(
            [
                Message(role="user", content="u5:帮我导出最终文件"),
                Message(role="assistant", content="已定位到目标目录"),
                Message(role="tool", content="导出成功：/tmp/demo.pptx"),
            ]
        )
        output = await compressor.build_window_messages(
            session_id="s4",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        status_block = next(m.content for m in output if "【当前任务状态】" in m.content)
        assert "待处理用户请求：u5:帮我导出最终文件" in status_block
        assert "本轮已知进展：" in status_block
        assert "- 本轮输出：已定位到目标目录" in status_block
        assert "- 工具结果：导出成功：/tmp/demo.pptx" in status_block
        assert "下一步：基于以上本轮进展继续完成当前请求" in status_block
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# 回归测试：多模态消息保真
# ---------------------------------------------------------------------------

_FAKE_IMAGE = ImageContent(mime="image/jpeg", data="ZmFrZS1pbWFnZQ==")


def _mk_image_group(i: int, text: str) -> list[Message]:
    return [
        Message(role="user", content=f"u{i}:{text}", images=[_FAKE_IMAGE]),
        Message(role="assistant", content=f"a{i}:{text}"),
    ]


@pytest.mark.asyncio
async def test_current_turn_images_preserved_in_output(tmp_path: Path) -> None:
    """当前轮含 images 时，输出消息中仍然存在带 images 的 user 消息。"""
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-img-cur",
            channel="webchat",
            peer_id="u-img",
            model="test",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, "历史消息") for i in range(1, 6)]
        groups.append([
            Message(
                role="user",
                content="描述这张图",
                images=[_FAKE_IMAGE],
            ),
        ])
        output = await compressor.build_window_messages(
            session_id="s-img-cur",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        image_msgs = [m for m in output if m.images]
        assert len(image_msgs) >= 1
        assert image_msgs[-1].role == "user"
        assert image_msgs[-1].images == [_FAKE_IMAGE]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_current_turn_tool_messages_unchanged(tmp_path: Path) -> None:
    """当前轮工具消息不变（不截断、不合并）。"""
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-cur-tool",
            channel="webchat",
            peer_id="u-tool",
            model="test",
            created_at=now,
            updated_at=now,
        )
        long_tool_result = "x" * 2000
        current_turn = [
            Message(role="user", content="帮我查询数据"),
            Message(role="assistant", content="正在查询..."),
            Message(role="tool", content=long_tool_result, tool_call_id="tc_1"),
            Message(role="assistant", content="查询完成"),
        ]
        groups = [_mk_group(i, "旧消息") for i in range(1, 4)]
        groups.append(current_turn)

        output = await compressor.build_window_messages(
            session_id="s-cur-tool",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        tool_msgs = [m for m in output if m.role == "tool"]
        assert len(tool_msgs) >= 1
        assert tool_msgs[-1].content == long_tool_result
        assert tool_msgs[-1].tool_call_id == "tc_1"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prev4_user_text_and_images_preserved(tmp_path: Path) -> None:
    """前 4 轮用户文本、图片 markdown、文件路径仍存在。"""
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-prev4-img",
            channel="webchat",
            peer_id="u-prev4",
            model="test",
            created_at=now,
            updated_at=now,
        )
        groups: list[list[Message]] = []
        for i in range(1, 8):
            groups.append(_mk_image_group(i, f"带图消息{i}"))
        groups.append([Message(role="user", content="当前轮")])

        output = await compressor.build_window_messages(
            session_id="s-prev4-img",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        for i in range(4, 8):
            assert any(
                m.role == "user" and f"u{i}:带图消息{i}" in m.content
                for m in output
            ), f"第{i}轮用户文本丢失"
            assert any(
                m.role == "user" and m.images and m.images == [_FAKE_IMAGE]
                for m in output
                if f"u{i}:带图消息{i}" in m.content
            ), f"第{i}轮图片丢失"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_prev4_tool_results_compacted_but_structure_kept(tmp_path: Path) -> None:
    """前 4 轮工具结果可以被缩短，但不能整轮坍缩成单个 assistant 文本块。"""
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-prev4-tool",
            channel="webchat",
            peer_id="u-prev4-t",
            model="test",
            created_at=now,
            updated_at=now,
        )
        long_tool_output = "output_line\n" * 200
        prev_turn = [
            Message(role="user", content="u-prev:帮我运行脚本"),
            Message(role="assistant", content="正在运行"),
            Message(role="tool", content=long_tool_output, tool_call_id="tc_prev"),
            Message(role="assistant", content="运行完毕"),
        ]
        groups = [_mk_group(i, "旧消息") for i in range(1, 4)]
        groups.append(prev_turn)
        groups.append([Message(role="user", content="当前轮请求")])

        output = await compressor.build_window_messages(
            session_id="s-prev4-tool",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        tool_msgs = [m for m in output if m.role == "tool" and m.tool_call_id == "tc_prev"]
        assert len(tool_msgs) == 1
        assert len(tool_msgs[0].content) < len(long_tool_output)
        assert "已截断" in tool_msgs[0].content

        assert any(m.role == "user" and "u-prev:帮我运行脚本" in m.content for m in output)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_older_history_still_uses_l1_l0_compression(tmp_path: Path) -> None:
    """第 6 轮及更早历史仍按 L1/L0 压缩进入窗口（不作为原始 Message 出现）。"""
    store = await _mk_store(tmp_path)
    try:
        compressor = SessionGroupCompressor(store)
        now = datetime.now(UTC).isoformat()
        await store.save_session(
            session_id="s-old-hist",
            channel="webchat",
            peer_id="u-old",
            model="test",
            created_at=now,
            updated_at=now,
        )
        groups = [_mk_group(i, f"旧历史{i}") for i in range(1, 10)]
        groups.append([Message(role="user", content="当前轮")])

        output = await compressor.build_window_messages(
            session_id="s-old-hist",
            messages=_flatten(groups),
            router=_NoopRouter(),  # type: ignore[arg-type]
            model_id="",
        )

        history_blocks = [m for m in output if "【历史摘要" in m.content]
        assert len(history_blocks) >= 1
        for i in range(6, 10):
            assert any(m.role == "user" and f"u{i}:旧历史{i}" in m.content for m in output)
        for i in range(1, 5):
            assert not any(m.role == "user" and f"u{i}:旧历史{i}" in m.content for m in output)
    finally:
        await store.close()


def test_compact_prev_group_preserves_images() -> None:
    """_compact_prev_group 保留 user 和 assistant 的 images 属性。"""
    group = [
        Message(role="user", content="看看图片", images=[_FAKE_IMAGE]),
        Message(
            role="assistant",
            content="图片内容分析",
            images=[_FAKE_IMAGE],
        ),
        Message(role="tool", content="y" * 1000, tool_call_id="tc_a"),
    ]
    compacted = _compact_prev_group(group)
    assert compacted[0].images == [_FAKE_IMAGE]
    assert compacted[1].images == [_FAKE_IMAGE]
    assert len(compacted[2].content) < 1000
    assert compacted[2].tool_call_id == "tc_a"
