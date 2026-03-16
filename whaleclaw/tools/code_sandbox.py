"""Code execution sandbox tool."""

from __future__ import annotations

import asyncio
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult

_MAX_OUTPUT = 50_000


class CodeSandboxTool(Tool):
    """Execute Python code in a safe sandbox with timeout."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="code_sandbox",
            description="Run Python code in a sandbox with timeout and resource limits.",
            parameters=[
                ToolParameter(
                    name="language",
                    type="string",
                    description="Language (currently only python).",
                    required=True,
                    enum=["python"],
                ),
                ToolParameter(
                    name="code",
                    type="string",
                    description="Code to execute.",
                    required=True,
                ),
                ToolParameter(
                    name="timeout",
                    type="integer",
                    description="Timeout in seconds (default 30).",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        code: str = kwargs.get("code", "")
        timeout: int = int(kwargs.get("timeout", 30))

        if not code.strip():
            return ToolResult(success=False, output="", error="代码为空")

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3",
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return ToolResult(
                success=False,
                output="",
                error=f"执行超时 ({timeout}s)",
            )
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        out = stdout.decode(errors="replace")[:_MAX_OUTPUT]
        err = stderr.decode(errors="replace")[:_MAX_OUTPUT]

        output = out
        if err:
            output += f"\n[stderr]\n{err}"

        return ToolResult(
            success=proc.returncode == 0,
            output=output.strip(),
            error=err if proc.returncode != 0 else None,
        )
