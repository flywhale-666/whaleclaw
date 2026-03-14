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

_PROJECT_PYTHON_BIN = Path(__file__).resolve().parents[2] / "python" / "bin"
_PROJECT_PYTHON = _PROJECT_PYTHON_BIN / "python3.12"
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
        command = _prefer_project_python_for_direct_script(
            _prefer_project_python(_fix_broken_python_path(_strip_control_chars(raw_command)))
        )
        timeout: int = int(kwargs.get("timeout", 30))
        background = bool(kwargs.get("background", False))

        if not command.strip():
            return ToolResult(success=False, output="", error="命令为空")

        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                return ToolResult(success=False, output="", error=f"危险命令被拦截: {command}")

        env = os.environ.copy()
        if _PROJECT_PYTHON_BIN.is_dir():
            env["PATH"] = f"{_PROJECT_PYTHON_BIN}:{env.get('PATH', '')}"

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
