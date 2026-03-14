"""Permission model — tool whitelist/blacklist, path restrictions."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field


class ToolPermission(BaseModel):
    """Tool permission configuration."""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)


class SecurityPolicy(BaseModel):
    """Per-session security policy."""

    sandbox: bool = False
    tools: ToolPermission = Field(default_factory=ToolPermission)
    max_tool_rounds: int = 50
    allow_file_write: bool = True
    allow_network: bool = True
    allowed_paths: list[str] = Field(default_factory=list)
    denied_paths: list[str] = Field(
        default_factory=lambda: [
            "/etc/",
            "/var/",
            "/usr/",
            "/sys/",
            "/proc/",
            "~/.ssh/",
            "~/.gnupg/",
        ]
    )


_DANGEROUS_CMD_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"rm\s+-fr\s+/", re.IGNORECASE),
    re.compile(r"rm\s+-r\s+-f\s+/", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"dd\s+if=/dev/zero", re.IGNORECASE),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
]


class PermissionChecker:
    """Checks tool, path, and command permissions against a SecurityPolicy."""

    @staticmethod
    def check_tool(tool_name: str, policy: SecurityPolicy) -> bool:
        """Return True if tool is allowed, False if denied."""
        tools = policy.tools
        if tool_name in tools.deny:
            return False
        if "*" in tools.allow:
            return True
        return tool_name in tools.allow

    @staticmethod
    def check_path(path: str, policy: SecurityPolicy, write: bool = False) -> bool:
        """Return True if path access is allowed."""
        if write and not policy.allow_file_write:
            return False
        expanded = str(Path(path).expanduser())
        for denied in policy.denied_paths:
            if expanded.startswith(str(Path(denied).expanduser())):
                return False
        if policy.allowed_paths:
            for allowed in policy.allowed_paths:
                if expanded.startswith(str(Path(allowed).expanduser())):
                    return True
            return False
        return True

    @staticmethod
    def check_command(command: str, policy: SecurityPolicy) -> bool:
        """Return False if command matches dangerous patterns."""
        return all(not pat.search(command) for pat in _DANGEROUS_CMD_PATTERNS)
