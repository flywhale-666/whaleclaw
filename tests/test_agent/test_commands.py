"""Tests for chat commands."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from whaleclaw.agent.commands import ChatCommand
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore

_CmdFixture = tuple[ChatCommand, Session]


@pytest.fixture()
async def cmd(tmp_path: Path) -> AsyncIterator[_CmdFixture]:
    store = SessionStore(db_path=tmp_path / "test.db")
    await store.open()
    mgr = SessionManager(store, WhaleclawConfig())
    session = await mgr.create("webchat", "user1")
    yield ChatCommand(mgr), session
    await store.close()


@pytest.mark.asyncio
async def test_not_a_command(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    assert await chat_cmd.handle("hello", session) is None


@pytest.mark.asyncio
async def test_help(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/help", session)
    assert result is not None
    assert "/new" in result


@pytest.mark.asyncio
async def test_status(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/status", session)
    assert result is not None
    assert session.id in result


@pytest.mark.asyncio
async def test_model_switch(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/model openai/gpt-5.2", session)
    assert result is not None
    assert "gpt-5.2" in result
    assert session.model == "openai/gpt-5.2"


@pytest.mark.asyncio
async def test_model_no_arg(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/model", session)
    assert result is not None
    assert "当前模型" in result


@pytest.mark.asyncio
async def test_think(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/think high", session)
    assert result is not None
    assert "high" in result
    assert session.thinking_level == "high"


@pytest.mark.asyncio
async def test_think_invalid(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/think banana", session)
    assert result is not None
    assert "当前" in result


@pytest.mark.asyncio
async def test_reset(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/new", session)
    assert result is not None
    assert "重置" in result


@pytest.mark.asyncio
async def test_unknown_command(cmd: _CmdFixture) -> None:
    chat_cmd, session = cmd
    result = await chat_cmd.handle("/foobar", session)
    assert result is not None
    assert "未知命令" in result
