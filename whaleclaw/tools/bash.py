"""Bash command execution tool."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.tools.process_registry import register_background_process

_DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/\s*$"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if=/dev/zero\b"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"),
]

_MAX_OUTPUT = 50_000
_LONG_RUNNING_SCRIPT_RE = re.compile(r"test_nano_banana[_\w]*\.py")
_LONG_RUNNING_SCRIPT_TIMEOUT_SECONDS = 300
_NANO_BANANA_PARALLEL_LIMIT = 5
_NANO_BANANA_BATCH_DELAY_SECONDS = 1.5
_HEREDOC_START_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_PYTHON_BIN = _PROJECT_ROOT / "python" / "bin"
_PROJECT_PYTHON = _PROJECT_PYTHON_BIN / "python3.12"
_PROJECT_NODE_BIN = _PROJECT_ROOT / "node" / "bin"
_PYTHON_CMD_RE = re.compile(r"(?<![\w./-])(python3|python)(?=\s|$)")
_DIRECT_PY_SCRIPT_RE = re.compile(
    r"^"
    r"(?P<prefix>(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|[^\s]+)\s+)*)"
    r"(?P<script>(?:~|/|\./|\.\./)[^\s;&|]+\.py)"
    r"(?P<suffix>(?:\s+.*)?)$"
)


def _strip_control_chars(text: str) -> str:
    """Remove ASCII control characters except LF/TAB/CR."""
    return "".join(
        ch for ch in text
        if ch in ("\n", "\t", "\r") or (ord(ch) >= 32 and ord(ch) != 127)
    )


_BROKEN_PYTHON_PATH_RE = re.compile(
    r"(?:/[^\s;|&\"']+)?/\./python/bin/python3(?:\.12)?"
)


def _fix_broken_python_path(command: str) -> str:
    """修复 LLM 拼出的错误 Python 路径，如 /Users/x/./python/bin/python3.12"""
    if "/./python/bin/" not in command:
        return command
    return _BROKEN_PYTHON_PATH_RE.sub(str(_PROJECT_PYTHON), command)


def _prefer_project_python(command: str) -> str:
    """Rewrite bare python/python3 to project-embedded python when available."""
    if not _PROJECT_PYTHON.is_file():
        return command
    return _PYTHON_CMD_RE.sub(str(_PROJECT_PYTHON), command)


def _prefer_project_python_for_direct_script(command: str) -> str:
    """Rewrite a direct ``script.py`` invocation to use the embedded Python."""
    if not _PROJECT_PYTHON.is_file():
        return command
    match = _DIRECT_PY_SCRIPT_RE.match(command.strip())
    if match is None:
        return command
    prefix = match.group("prefix")
    script = match.group("script")
    suffix = match.group("suffix") or ""
    return f"{prefix}{_PROJECT_PYTHON} {script}{suffix}"


def _flush_parallel_nano_banana_lines(lines: list[str], out: list[str]) -> None:
    if not lines:
        return
    if len(lines) == 1:
        out.extend(lines)
        return
    for batch_idx in range(0, len(lines), _NANO_BANANA_PARALLEL_LIMIT):
        batch = lines[batch_idx : batch_idx + _NANO_BANANA_PARALLEL_LIMIT]
        if batch_idx > 0:
            out.append(f"sleep {_NANO_BANANA_BATCH_DELAY_SECONDS}")
        out.append('__wc_nb_pids=""')
        out.append("__wc_nb_fail=0")
        for line in batch:
            out.append(f"( {line} ) &")
            out.append('__wc_nb_pids="$__wc_nb_pids $!"')
        out.append("for __wc_nb_pid in $__wc_nb_pids; do")
        out.append('  wait "$__wc_nb_pid" || __wc_nb_fail=1')
        out.append("done")
        out.append('if [ "$__wc_nb_fail" -ne 0 ]; then')
        out.append('  exit "$__wc_nb_fail"')
        out.append("fi")


def _normalize_nano_banana_command_line(line: str) -> str:
    return _prefer_project_python_for_direct_script(
        _prefer_project_python(_fix_broken_python_path(line))
    )


def _rewrite_nano_banana_batches(command: str) -> str:
    """Normalize and batch parallelizable Nano Banana lines in shell scripts."""
    if _LONG_RUNNING_SCRIPT_RE.search(command) is None:
        return command

    lines = command.splitlines()
    rewritten: list[str] = []
    pending_nano_lines: list[str] = []
    heredoc_terminator: str | None = None

    for raw_line in lines:
        line = raw_line
        if heredoc_terminator is not None:
            _flush_parallel_nano_banana_lines(pending_nano_lines, rewritten)
            pending_nano_lines = []
            rewritten.append(line)
            if line.strip() == heredoc_terminator:
                heredoc_terminator = None
            continue

        if _LONG_RUNNING_SCRIPT_RE.search(line):
            pending_nano_lines.append(_normalize_nano_banana_command_line(line.strip()))
            continue

        _flush_parallel_nano_banana_lines(pending_nano_lines, rewritten)
        pending_nano_lines = []
        rewritten.append(line)

        heredoc_match = _HEREDOC_START_RE.search(line)
        if heredoc_match is not None:
            heredoc_terminator = heredoc_match.group(2)

    _flush_parallel_nano_banana_lines(pending_nano_lines, rewritten)
    return "\n".join(rewritten)


class BashTool(Tool):
    """Execute a bash command and return stdout/stderr/exit_code."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Execute a bash command. Returns stdout, stderr, and exit code.",
            parameters=[
                ToolParameter(
                    name="command", type="string", description="The bash command to execute."
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 30, max 300).",
                    required=False,
                ),
                ToolParameter(
                    name="background",
                    type="boolean",
                    description="Run command in background and return a session id.",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        raw_command: str = kwargs.get("command", "")
        command = _rewrite_nano_banana_batches(
            _prefer_project_python_for_direct_script(
                _prefer_project_python(_fix_broken_python_path(_strip_control_chars(raw_command)))
            )
        )
        timeout: int = int(kwargs.get("timeout", 30))
        if _LONG_RUNNING_SCRIPT_RE.search(command):
            timeout = max(timeout, _LONG_RUNNING_SCRIPT_TIMEOUT_SECONDS)
        background = bool(kwargs.get("background", False))

        if not command.strip():
            return ToolResult(success=False, output="", error="命令为空")

        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                return ToolResult(success=False, output="", error=f"危险命令被拦截: {command}")

        env = os.environ.copy()
        extra_paths: list[str] = []
        if _PROJECT_PYTHON_BIN.is_dir():
            extra_paths.append(str(_PROJECT_PYTHON_BIN))
        if _PROJECT_NODE_BIN.is_dir():
            extra_paths.append(str(_PROJECT_NODE_BIN))
        if extra_paths:
            env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            if background:
                session = register_background_process(
                    command=command,
                    cwd=os.getcwd(),
                    process=proc,
                )
                return ToolResult(
                    success=True,
                    output=(
                        f"后台命令已启动\n"
                        f"session_id: {session.id}\n"
                        f"pid: {proc.pid or 0}\n"
                        f"command: {command}"
                    ),
                )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return ToolResult(success=False, output="", error=f"命令超时 ({timeout}s)")
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        out = stdout.decode(errors="replace")[:_MAX_OUTPUT]
        err = stderr.decode(errors="replace")[:_MAX_OUTPUT]
        exit_code = proc.returncode or 0

        if exit_code == 0:
            _postprocess_delivery_files(out, command)

        output = out
        if err:
            output += f"\n[stderr]\n{err}"
        output += f"\n[exit_code: {exit_code}]"

        return ToolResult(
            success=exit_code == 0,
            output=output.strip(),
            error=err if exit_code != 0 else None,
        )


_DELIVERY_PATH_RE = re.compile(
    r"(/[^\s:\"'<>|]+\.(?:pptx|docx|html?))\b",
    re.IGNORECASE | re.UNICODE,
)

_POSTPROCESS_RECENCY_SEC = 30
_POSTPROCESS_SUFFIXES = {".pptx", ".docx", ".html", ".htm"}


def _postprocess_delivery_files(output: str, command: str = "") -> None:
    """Auto-fix generated delivery files after a successful bash run.

    Supported: .pptx (face crop + Z-order), .docx (face crop), .html (object-fit).

    Sources (deduplicated):
      1. Paths found in stdout
      2. Paths found in the command text itself
      3. Recently modified files in /tmp (within last 30s)
    """
    import time

    candidates: set[str] = set()

    for text in (output, command):
        for m in _DELIVERY_PATH_RE.finditer(text):
            candidates.add(m.group(1))

    cutoff = time.time() - _POSTPROCESS_RECENCY_SEC
    try:
        for p in Path("/tmp").iterdir():
            if p.suffix.lower() in _POSTPROCESS_SUFFIXES and p.stat().st_mtime >= cutoff:
                candidates.add(str(p))
    except Exception:
        pass

    for c in candidates:
        p = Path(c)
        if not p.exists():
            continue
        suffix = p.suffix.lower()
        try:
            if suffix == ".pptx":
                from whaleclaw.utils.pptx_postprocess import fix_pptx
                fix_pptx(p)
            elif suffix == ".docx":
                from whaleclaw.utils.docx_postprocess import fix_docx
                fix_docx(p)
            elif suffix in (".html", ".htm"):
                from whaleclaw.utils.html_postprocess import fix_html
                fix_html(p)
        except Exception:
            pass
