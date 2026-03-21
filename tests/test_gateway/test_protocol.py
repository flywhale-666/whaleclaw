"""Tests for WebSocket protocol message types."""

from __future__ import annotations

from whaleclaw.gateway.protocol import (
    MessageType,
    WSMessage,
    make_error,
    make_message,
    make_pong,
    make_stream,
    make_tool_result,
)


class TestWSMessage:
    def test_roundtrip(self) -> None:
        msg = WSMessage(type=MessageType.MESSAGE, payload={"content": "hello"})
        raw = msg.model_dump_json()
        restored = WSMessage.model_validate_json(raw)
        assert restored.type == MessageType.MESSAGE
        assert restored.payload["content"] == "hello"
        assert restored.id == msg.id

    def test_auto_id(self) -> None:
        m1 = WSMessage(type=MessageType.PING)
        m2 = WSMessage(type=MessageType.PING)
        assert m1.id != m2.id

    def test_timestamp_set(self) -> None:
        msg = WSMessage(type=MessageType.PING)
        assert msg.timestamp is not None


class TestHelpers:
    def test_make_stream(self) -> None:
        msg = make_stream("sid-1", "chunk")
        assert msg.type == MessageType.STREAM
        assert msg.session_id == "sid-1"
        assert msg.payload["content"] == "chunk"

    def test_make_message(self) -> None:
        msg = make_message("sid-1", "full reply")
        assert msg.type == MessageType.MESSAGE
        assert msg.payload["content"] == "full reply"

    def test_make_error(self) -> None:
        msg = make_error("sid-1", "oops")
        assert msg.type == MessageType.ERROR
        assert msg.payload["error"] == "oops"

    def test_make_pong(self) -> None:
        msg = make_pong()
        assert msg.type == MessageType.PONG

    def test_make_tool_result_includes_error(self) -> None:
        msg = make_tool_result("sid-1", "ppt_edit", "", False, "删除失败")
        assert msg.type == MessageType.TOOL_RESULT
        assert msg.payload["name"] == "ppt_edit"
        assert msg.payload["success"] is False
        assert msg.payload["error"] == "删除失败"
