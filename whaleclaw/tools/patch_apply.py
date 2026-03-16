"""Patch apply tool — apply unified diff patch to files."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

_MAX_OUTPUT = 50_000


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


class PatchApplyTool(Tool):
    """Apply or dry-run a unified diff patch."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="patch_apply",
            description=(
                "Apply unified diff patch text to files. "
                "Supports dry-run check mode and custom working directory."
            ),
            parameters=[
                ToolParameter(
                    name="patch",
                    type="string",
                    description="Unified diff text to apply.",
                ),
                ToolParameter(
                    name="strip",
                    type="integer",
                    description="Strip leading path components (default 1).",
                    required=False,
                ),
                ToolParameter(
                    name="check",
                    type="boolean",
                    description="Only validate patch applicability, do not write files.",
                    required=False,
                ),
                ToolParameter(
                    name="reverse",
                    type="boolean",
                    description="Apply patch in reverse.",
                    required=False,
                ),
                ToolParameter(
                    name="cwd",
                    type="string",
                    description="Working directory to resolve patch paths.",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        patch_text = str(kwargs.get("patch", ""))
        strip = int(kwargs.get("strip", 1))
        check = _as_bool(kwargs.get("check", False))
        reverse = _as_bool(kwargs.get("reverse", False))
        cwd_raw = str(kwargs.get("cwd", "")).strip()

        if not patch_text.strip():
            return ToolResult(success=False, output="", error="patch 为空")
        if strip < 0 or strip > 10:
            return ToolResult(success=False, output="", error="strip 必须在 0~10 之间")

        cwd = Path(cwd_raw).expanduser().resolve() if cwd_raw else Path.cwd().resolve()
        if not cwd.is_dir():
            return ToolResult(success=False, output="", error=f"cwd 目录不存在: {cwd}")

        patch_path: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".patch",
                delete=False,
            ) as tmp:
                tmp.write(patch_text)
                patch_path = Path(tmp.name)

            git_cmd = [
                "git",
                "apply",
                "--no-index",
                "--recount",
                f"-p{strip}",
            ]
            if reverse:
                git_cmd.append("--reverse")
            if check:
                git_cmd.append("--check")
            git_cmd.append(str(patch_path))

            git_out, git_err, git_code = await _run_cmd(git_cmd, cwd)
            if git_code == 0:
                mode = "校验通过" if check else "已应用"
                body = (git_out or git_err or "").strip()
                return ToolResult(
                    success=True,
                    output=f"patch_apply(git): {mode}\n{body}".strip(),
                )

            patch_cmd = [
                "patch",
                f"-p{strip}",
                "--batch",
                "--forward",
                "--input",
                str(patch_path),
            ]
            if reverse:
                patch_cmd.append("--reverse")
            if check:
                patch_cmd.append("--dry-run")

            patch_out, patch_err, patch_code = await _run_cmd(patch_cmd, cwd)
            if patch_code == 0:
                mode = "校验通过" if check else "已应用"
                body = (patch_out or patch_err or "").strip()
                return ToolResult(
                    success=True,
                    output=f"patch_apply(patch): {mode}\n{body}".strip(),
                )

            detail = (
                f"[git apply]\n{(git_out + '\n' + git_err).strip()}\n\n"
                f"[patch]\n{(patch_out + '\n' + patch_err).strip()}"
            ).strip()
            return ToolResult(
                success=False,
                output=detail[:_MAX_OUTPUT],
                error="补丁应用失败",
            )
        except (OSError, ValueError) as exc:
            return ToolResult(success=False, output="", error=str(exc))
        finally:
            if patch_path is not None:
                with suppress(OSError):
                    patch_path.unlink(missing_ok=True)


async def _run_cmd(cmd: list[str], cwd: Path, *, timeout: int = 30) -> tuple[str, str, int]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return ("", f"命令不存在: {cmd[0]}", 127)
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        return ("", f"命令超时 ({timeout}s): {cmd[0]}", 1)
    out = out_b.decode(errors="replace")[:_MAX_OUTPUT]
    err = err_b.decode(errors="replace")[:_MAX_OUTPUT]
    code = proc.returncode or 0
    return (out, err, code)
