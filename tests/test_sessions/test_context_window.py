"""Tests for context window management."""

from __future__ import annotations

from unittest.mock import MagicMock

from whaleclaw.providers.base import Message
from whaleclaw.sessions.context_window import TARGET_CONTENT_TOKENS, ContextWindow, _estimate_tokens


def _tokens(msgs: list[Message]) -> int:
    return sum(_estimate_tokens(m.content) for m in msgs)


def _content_tokens(msgs: list[Message]) -> int:
    return sum(_estimate_tokens(m.content) for m in msgs if m.role != "system")


class TestContextWindow:
    def test_get_max_context_known(self) -> None:
        cw = ContextWindow()
        assert cw.get_max_context("claude-sonnet-4-20250514") == 200_000

    def test_get_max_context_unknown(self) -> None:
        cw = ContextWindow()
        assert cw.get_max_context("unknown-model") == 128_000

    def test_no_compression_when_small_history(self) -> None:
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="system prompt"),
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ]
        trimmed = cw.trim(msgs, "glm-4.7")
        assert trimmed == msgs

    def test_recent_five_groups_are_protected(self) -> None:
        cw = ContextWindow()
        msgs = [Message(role="system", content="sys")]
        for i in range(10):
            msgs.append(Message(role="user", content=f"旧组用户{i} " + "旧历史内容 " * 120))
            msgs.append(Message(role="assistant", content=f"旧组助手{i} " + "旧历史回复 " * 120))
        recent_groups = [
            ("最近组1-用户", "最近组1-助手"),
            ("最近组2-用户", "最近组2-助手"),
            ("最近组3-用户", "最近组3-助手"),
            ("最近组4-用户", "最近组4-助手"),
            ("最近组5-用户", "最近组5-助手"),
        ]
        for u, a in recent_groups:
            msgs.append(Message(role="user", content=u))
            msgs.append(Message(role="assistant", content=a))
        trimmed = cw.trim(msgs, "qwen3.5-plus")
        tail = trimmed[-10:]
        assert [m.content for m in tail] == [x for g in recent_groups for x in g]

    def test_applies_l1_and_l0_compression(self) -> None:
        cw = ContextWindow()
        msgs = [Message(role="system", content="sys")]
        for i in range(14):
            msgs.append(Message(role="user", content=f"组{i}-用户 " + "历史 " * 220))
            msgs.append(Message(role="assistant", content=f"组{i}-助手 " + "历史回复 " * 220))
        msgs.extend([
            Message(role="user", content="recent-1-u"),
            Message(role="assistant", content="recent-1-a"),
            Message(role="user", content="recent-2-u"),
            Message(role="assistant", content="recent-2-a"),
            Message(role="user", content="recent-3-u"),
            Message(role="assistant", content="recent-3-a"),
            Message(role="user", content="recent-4-u"),
            Message(role="assistant", content="recent-4-a"),
            Message(role="user", content="recent-5-u"),
            Message(role="assistant", content="recent-5-a"),
        ])

        trimmed = cw.trim(msgs, "qwen3.5-plus")
        contents = [m.content for m in trimmed]
        assert any(c.startswith("[L1压缩]") for c in contents)
        # L0 压缩后的组可能因 budget 不足被丢弃，进入 compacted 摘要
        assert any(
            c.startswith("[L0压缩]")
            or c.startswith("[历史压缩摘要]")
            or c.startswith("[以下是本会话的历史摘要")
            for c in contents
        )

    def test_caps_content_tokens_close_to_1600(self) -> None:
        cw = ContextWindow()
        system = Message(role="system", content="S" * 120)
        msgs = [system]
        msgs.extend(Message(role="assistant", content="非常长的历史消息 " * 260) for _ in range(80))
        msgs.extend([
            Message(role="assistant", content="last-1"),
            Message(role="user", content="last-2"),
            Message(role="assistant", content="last-3"),
        ])

        trimmed = cw.trim(msgs, "qwen3.5-plus")
        content_total = _content_tokens(trimmed)
        assert content_total <= TARGET_CONTENT_TOKENS + 80


class TestTrimWithSummaries:
    def _make_summary_row(self, *, level: str, content: str, token_count: int) -> object:
        row = MagicMock()
        row.level = level
        row.content = content
        row.source_msg_start = 1
        row.source_msg_end = 10
        row.token_count = token_count
        return row

    def test_trim_with_summaries_generates_compact_block(self) -> None:
        cw = ContextWindow()
        msgs = [
            Message(role="system", content="系统提示 " * 1200),
            *[Message(role="assistant", content="超长历史 " * 600) for _ in range(220)],
            Message(role="user", content="最后用户消息"),
            Message(role="assistant", content="最后助手消息"),
            Message(role="user", content="最后第三条"),
        ]
        summaries = [
            self._make_summary_row(
                level="L0",
                content="L0总览：用户偏好简洁，近期在调试飞书附件",
                token_count=40,
            ),
            self._make_summary_row(
                level="L1",
                content="L1详述：历史中多次处理媒体文件与会话压缩问题",
                token_count=80,
            ),
        ]

        trimmed = cw.trim_with_summaries(msgs, "qwen3.5-plus", summaries)
        assert any(
            m.content.startswith("[历史压缩摘要]")
            or m.content.startswith("[以下是本会话的历史摘要")
            for m in trimmed
        )
        assert trimmed[-1].content == "最后第三条"
