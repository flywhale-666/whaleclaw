"""MCP service management — Agent tool + gateway helper.

项目内嵌 Node.js + mcporter，路径固定为 ``./node/bin/mcporter``，
与 ``./python/bin/python3.12`` 同理，复制项目即可用。
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MCPORTER_BIN = _PROJECT_ROOT / "node" / "bin" / "mcporter"
_NODE_BIN_DIR = _PROJECT_ROOT / "node" / "bin"


def is_mcporter_available() -> bool:
    """项目内嵌的 mcporter 是否存在。"""
    return _MCPORTER_BIN.exists()


def _run_mcporter(*args: str, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """用项目内嵌的 node 运行 mcporter。"""
    import os

    env = os.environ.copy()
    env["PATH"] = str(_NODE_BIN_DIR) + os.pathsep + env.get("PATH", "")

    return subprocess.run(
        [str(_MCPORTER_BIN), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def list_mcporter_servers() -> list[dict[str, Any]]:
    """Query ``mcporter list --json`` and return parsed server list."""
    if not is_mcporter_available():
        return []
    try:
        result = _run_mcporter("list", "--json")
        if result.returncode != 0:
            log.debug("mcp.mcporter_list_failed", stderr=result.stderr.strip())
            return []
        raw: Any = json.loads(result.stdout)
        # mcporter list --json 返回 {"servers": [...], ...}
        servers: list[Any] = []
        if isinstance(raw, dict):
            servers = raw.get("servers", [])
        elif isinstance(raw, list):
            servers = raw
        return [
            _sanitize_server(s)
            for s in servers
            if isinstance(s, dict)
        ]
    except Exception as exc:
        log.debug("mcp.mcporter_list_error", error=str(exc))
        return []


def remove_mcporter_server(server_id: str, config_path: str | None = None) -> bool:
    """Remove a server managed by mcporter. Returns ``True`` on success.

    Args:
        server_id: 服务名称（mcporter config remove <name>）。
        config_path: 配置文件路径。mcporter list 返回的 source.path 指明服务
            定义在哪个 json 中，删除时需指向同一文件，否则默认只查本地配置。
    """
    if not is_mcporter_available():
        return False
    try:
        args: list[str] = []
        if config_path:
            args.extend(["--config", config_path])
        args.extend(["config", "remove", server_id])
        result = _run_mcporter(*args)
        return result.returncode == 0
    except Exception as exc:
        log.debug("mcp.mcporter_remove_error", error=str(exc))
        return False


def _sanitize_server(server: dict[str, Any]) -> dict[str, Any]:
    """脱敏 transport URL 中可能包含的密钥。"""
    sanitized = dict(server)
    transport = sanitized.get("transport", "")
    if isinstance(transport, str) and ("key=" in transport or "token=" in transport):
        sanitized["transport"] = re.sub(
            r"(key|token|secret|password)=[^&\s]+",
            r"\1=***",
            transport,
            flags=re.IGNORECASE,
        )
    return sanitized


def aggregate_mcp_servers(
    builtin_servers: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """聚合内置 MCP + mcporter 两个来源的服务列表。"""
    result: list[dict[str, Any]] = []

    for s in builtin_servers or []:
        entry = dict(s)
        entry.setdefault("source", "builtin")
        result.append(entry)

    for s in list_mcporter_servers():
        entry = dict(s)
        raw_source = entry.get("source")
        if isinstance(raw_source, dict):
            entry["config_path"] = raw_source.get("path", "")
        entry["source"] = "mcporter"
        result.append(entry)

    return result


class McpManageTool(Tool):
    """Agent 可调用的 MCP 管理工具。"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="mcp_manage",
            description=(
                "Manage MCP (Model Context Protocol) services: "
                "list configured MCP servers, or remove a server by name. "
                "To add a new server, use bash tool: "
                "./node/bin/mcporter config add <name> --url <url>"
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action: list or remove.",
                    enum=["list", "remove"],
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="Server name to remove (required for remove action).",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        action = str(kwargs.get("action", "")).lower()
        name = str(kwargs.get("name", "")).strip() if kwargs.get("name") else ""

        if action == "list":
            return self._list()
        if action == "remove":
            if not name:
                return ToolResult(success=False, output="", error="remove 需要 name")
            return self._remove(name)
        return ToolResult(success=False, output="", error=f"未知操作: {action}")

    def _list(self) -> ToolResult:
        servers = aggregate_mcp_servers()
        if not servers:
            hint = (
                "暂无 MCP 服务。\n"
                "可通过内嵌 mcporter CLI 配置:\n"
                "  ./node/bin/mcporter config add <name> --url <sse-url>\n"
                "配置后重新执行 mcp_manage(action=\"list\") 查看。"
            )
            return ToolResult(success=True, output=hint)
        lines: list[str] = []
        for s in servers:
            name = s.get("name") or s.get("id") or "unknown"
            source = s.get("source", "unknown")
            transport = s.get("transport", "")
            status = s.get("status", "unknown")
            lines.append(f"- {name} [{source}] transport={transport} status={status}")
        return ToolResult(success=True, output="\n".join(lines))

    def _remove(self, name: str) -> ToolResult:
        config_path = self._find_config_path(name)
        ok = remove_mcporter_server(name, config_path=config_path)
        if ok:
            return ToolResult(success=True, output=f"已删除 MCP 服务: {name}")
        return ToolResult(
            success=False,
            output="",
            error=f"删除失败: {name}（mcporter 不可用或服务不存在）",
        )

    @staticmethod
    def _find_config_path(name: str) -> str | None:
        """从当前服务列表中查找指定服务所在的配置文件路径。"""
        for s in aggregate_mcp_servers():
            sname = s.get("name") or s.get("id") or ""
            if sname == name and s.get("config_path"):
                return str(s["config_path"])
        return None
