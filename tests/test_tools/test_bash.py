"""Tests for the bash tool."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from whaleclaw.tools import bash as bash_mod
from whaleclaw.tools.bash import BashTool


@pytest.fixture()
def tool() -> BashTool:
    return BashTool()


@pytest.mark.asyncio
async def test_echo(tool: BashTool) -> None:
    result = await tool.execute(command="echo hello")
    assert result.success
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_exit_code(tool: BashTool) -> None:
    result = await tool.execute(command="exit 1")
    assert not result.success
    assert "exit_code: 1" in result.output


@pytest.mark.asyncio
async def test_empty_command(tool: BashTool) -> None:
    result = await tool.execute(command="")
    assert not result.success
    assert result.error == "命令为空"


@pytest.mark.asyncio
async def test_dangerous_command(tool: BashTool) -> None:
    result = await tool.execute(command="rm -rf /")
    assert not result.success
    assert "危险命令" in (result.error or "")


@pytest.mark.asyncio
async def test_timeout(tool: BashTool) -> None:
    result = await tool.execute(command="sleep 10", timeout=1)
    assert not result.success
    assert "超时" in (result.error or "")


@pytest.mark.asyncio
async def test_background_returns_session_id(tool: BashTool) -> None:
    result = await tool.execute(command="sleep 1", background=True)
    assert result.success
    assert "session_id:" in result.output


@pytest.mark.asyncio
async def test_control_chars_are_stripped(tool: BashTool) -> None:
    result = await tool.execute(command="\x18echo hello\x18")
    assert result.success
    assert "hello" in result.output


@pytest.mark.asyncio
async def test_only_control_chars_becomes_empty(tool: BashTool) -> None:
    result = await tool.execute(command="\x10\x18\x00")
    assert not result.success
    assert result.error == "命令为空"


def test_prefer_project_python_rewrites_bare_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "python3.12"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(bash_mod, "_PROJECT_PYTHON", fake_python)

    rewritten = bash_mod._prefer_project_python("python3 /tmp/a.py && python -V")
    expected = f"{fake_python} /tmp/a.py && {fake_python} -V"
    assert rewritten == expected


def test_prefer_project_python_rewrites_direct_python_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "python3.12"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(bash_mod, "_PROJECT_PYTHON", fake_python)

    rewritten = bash_mod._prefer_project_python_for_direct_script(
        "/tmp/test_nano_banana_2.py --mode edit"
    )

    assert rewritten == f"{fake_python} /tmp/test_nano_banana_2.py --mode edit"


def test_prefer_project_python_rewrites_env_prefixed_direct_python_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "python3.12"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(bash_mod, "_PROJECT_PYTHON", fake_python)

    rewritten = bash_mod._prefer_project_python_for_direct_script(
        "NANO_BANANA_API_KEY='x' /tmp/test_nano_banana_2.py --mode edit"
    )

    assert rewritten == (
        f"NANO_BANANA_API_KEY='x' {fake_python} /tmp/test_nano_banana_2.py --mode edit"
    )


def test_rewrite_nano_banana_batches_normalizes_direct_script_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_python = tmp_path / "python3.12"
    fake_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(bash_mod, "_PROJECT_PYTHON", fake_python)

    command = "\n".join(
        [
            "echo before",
            "/tmp/test_nano_banana_2.py --mode text --prompt 'a'",
            "echo after",
        ]
    )

    rewritten = bash_mod._rewrite_long_running_batches(command)

    assert f"{fake_python} /tmp/test_nano_banana_2.py --mode text --prompt 'a'" in rewritten
    assert "echo before" in rewritten
    assert "echo after" in rewritten


def test_rewrite_nano_banana_batches_splits_into_parallel_chunks_of_five() -> None:
    lines = [
        f"./python/bin/python3.12 /tmp/test_nano_banana_{i}.py --mode text --prompt '{i}'"
        for i in range(7)
    ]
    rewritten = bash_mod._rewrite_long_running_batches("\n".join(lines))

    assert rewritten.count('__wc_nb_pids=""') == 2
    assert rewritten.count("( ./python/bin/python3.12 /tmp/test_nano_banana_") == 7
    assert "sleep 1.5" in rewritten


@pytest.mark.asyncio
async def test_nano_banana_script_uses_300_second_min_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return (b"ok", b"")

    async def _fake_create_subprocess_shell(*args: object, **kwargs: object) -> _FakeProc:
        return _FakeProc()

    async def _fake_wait_for(awaitable: object, timeout: float | None = None) -> tuple[bytes, bytes]:
        captured["timeout"] = timeout
        return await awaitable

    monkeypatch.setattr(bash_mod.asyncio, "create_subprocess_shell", _fake_create_subprocess_shell)
    monkeypatch.setattr(bash_mod.asyncio, "wait_for", _fake_wait_for)

    tool = BashTool()
    result = await tool.execute(
        command="./python/bin/python3.12 /tmp/test_nano_banana_2.py --mode text",
        timeout=250,
    )

    assert result.success is True
    assert captured["timeout"] == 300
