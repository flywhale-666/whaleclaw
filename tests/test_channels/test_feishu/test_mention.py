"""Tests for Feishu mention utilities."""

from __future__ import annotations

from typing import Any

from whaleclaw.channels.feishu.mention import (
    format_mention_for_card,
    format_mention_for_text,
    is_bot_mentioned,
    strip_bot_mention,
)


class TestMention:
    def test_is_bot_mentioned_true(self) -> None:
        message = {
            "mentions": [
                {"key": "@bot", "id": {"open_id": "ou_bot123"}, "name": "Bot"},
            ],
        }
        assert is_bot_mentioned(message, "ou_bot123")

    def test_is_bot_mentioned_false(self) -> None:
        message: dict[str, Any] = {"mentions": []}
        assert not is_bot_mentioned(message, "ou_bot123")

    def test_strip_bot_mention(self) -> None:
        text = "@MyBot 你好世界"
        result = strip_bot_mention(text, "MyBot")
        assert result == "你好世界"

    def test_format_mention_text(self) -> None:
        result = format_mention_for_text("ou_123", "Alice")
        assert 'user_id="ou_123"' in result

    def test_format_mention_card(self) -> None:
        result = format_mention_for_card("ou_123", "Alice")
        assert result["tag"] == "at"
        assert result["user_id"] == "ou_123"
