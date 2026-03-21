"""WebSocket message protocol definitions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageType(StrEnum):
    """WebSocket message types."""

    MESSAGE = "message"
    STREAM = "stream"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    STATUS = "status"
    CANVAS_PUSH = "canvas_push"
    CANVAS_RESET = "canvas_reset"
    CANVAS_EVENT = "canvas_event"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    AGENT_DONE = "agent_done"


class WSMessage(BaseModel):
    """A single WebSocket protocol message."""

    type: MessageType
    id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


def make_stream(session_id: str, content: str) -> WSMessage:
    """Create a stream chunk message."""
    return WSMessage(
        type=MessageType.STREAM,
        session_id=session_id,
        payload={"content": content},
    )


def make_message(session_id: str, content: str) -> WSMessage:
    """Create a final complete message."""
    return WSMessage(
        type=MessageType.MESSAGE,
        session_id=session_id,
        payload={"content": content},
    )


def make_error(session_id: str | None, error: str) -> WSMessage:
    """Create an error message."""
    return WSMessage(
        type=MessageType.ERROR,
        session_id=session_id,
        payload={"error": error},
    )


def make_pong() -> WSMessage:
    """Create a pong response."""
    return WSMessage(type=MessageType.PONG)


def make_tool_call(session_id: str, name: str, arguments: dict[str, Any]) -> WSMessage:
    """Create a tool_call notification."""
    return WSMessage(
        type=MessageType.TOOL_CALL,
        session_id=session_id,
        payload={"name": name, "arguments": arguments},
    )


def make_tool_result(
    session_id: str,
    name: str,
    output: str,
    success: bool,
    error: str | None = None,
) -> WSMessage:
    """Create a tool_result notification."""
    return WSMessage(
        type=MessageType.TOOL_RESULT,
        session_id=session_id,
        payload={"name": name, "output": output, "success": success, "error": error},
    )


def make_status(session_id: str, text: str) -> WSMessage:
    """Create a status update message."""
    return WSMessage(
        type=MessageType.STATUS,
        session_id=session_id,
        payload={"text": text},
    )


def make_canvas_push(
    session_id: str,
    html: str,
    css: str = "",
    js: str = "",
    title: str = "",
) -> WSMessage:
    """Create a canvas push message."""
    return WSMessage(
        type=MessageType.CANVAS_PUSH,
        session_id=session_id,
        payload={"html": html, "css": css, "js": js, "title": title},
    )


def make_canvas_reset(session_id: str) -> WSMessage:
    """Create a canvas reset message."""
    return WSMessage(
        type=MessageType.CANVAS_RESET,
        session_id=session_id,
        payload={},
    )


def make_agent_done(
    session_id: str,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    llm_rounds: int,
) -> WSMessage:
    """Create an agent_done message with execution metadata."""
    return WSMessage(
        type=MessageType.AGENT_DONE,
        session_id=session_id,
        payload={
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "llm_rounds": llm_rounds,
        },
    )


def make_approval_request(
    session_id: str,
    tool_name: str,
    command: str,
    reason: str,
    request_id: str | None = None,
) -> WSMessage:
    """Create an approval request for a high-risk tool call.

    payload:
        tool: 工具名
        command: 完整命令内容（不截断不篡改）
        reason: 需要审批的原因
        request_id: 用于关联 response 的唯一 ID
    """
    rid = request_id or uuid4().hex
    return WSMessage(
        type=MessageType.APPROVAL_REQUEST,
        session_id=session_id,
        payload={
            "tool": tool_name,
            "command": command,
            "reason": reason,
            "request_id": rid,
        },
    )
