"""Permission model — tool whitelist/blacklist, path restrictions, approval policy.

三层安全模型中的第 2 层（工具调用安全）：
- 危险命令拦截（绝对禁止）
- 高风险命令审批（需用户确认）
- 不可撤回操作检测
- safe-delete 策略（优先 trash 而非 rm）
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class ToolPermission(BaseModel):
    """Tool permission configuration."""

    allow: list[str] = Field(default_factory=lambda: ["*"])
    deny: list[str] = Field(default_factory=list)


class ApprovalDecision(StrEnum):
    """用户对高风险操作的审批决策。"""

    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    DENY = "deny"


class ToolApprovalPolicy(BaseModel):
    """高风险工具调用的审批策略。

    - always_allowed_commands: 用户已永久授权的命令模式（正则）
    - require_approval_patterns: 需要审批的命令模式（正则）
    """

    require_approval_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\bcurl\b.*\b-X\s*(POST|PUT|DELETE|PATCH)\b",
            r"\bgit\s+push\b",
            r"\bgit\s+push\s+.*--force\b",
            r"\bnpm\s+publish\b",
            r"\bpip\s+install\b.*--user",
            r"\bsudo\b",
            r"\bchmod\s+777\b",
            r"\bchown\b",
            r"\bsystemctl\s+(start|stop|restart|enable|disable)\b",
            r"\blaunchctl\b",
        ]
    )
    always_allowed_commands: list[str] = Field(default_factory=list)


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
    approval: ToolApprovalPolicy = Field(default_factory=ToolApprovalPolicy)
    safe_delete: bool = True


# ── 绝对禁止的命令（第 2 层底线，无法审批通过） ──

_DANGEROUS_CMD_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/", re.IGNORECASE),
    re.compile(r"rm\s+-fr\s+/", re.IGNORECASE),
    re.compile(r"rm\s+-r\s+-f\s+/", re.IGNORECASE),
    re.compile(r"\bmkfs\b", re.IGNORECASE),
    re.compile(r"dd\s+if=/dev/zero", re.IGNORECASE),
    re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", re.IGNORECASE),
]

# ── 不可撤回操作检测 ──

_IRREVOCABLE_PATTERNS = [
    re.compile(r"\bgit\s+push\s+.*--force\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\s+-f\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+publish\b", re.IGNORECASE),
    re.compile(r"\bcurl\b.*\b-X\s*(DELETE|POST)\b.*\bapi\b", re.IGNORECASE),
    re.compile(r"\bsendmail\b|\bmail\s+-s\b", re.IGNORECASE),
    re.compile(r"\bdropdb\b|\bDROP\s+DATABASE\b", re.IGNORECASE),
    re.compile(r"\bdrop\s+table\b", re.IGNORECASE),
]

# ── safe-delete：rm 命令改写为 trash（macOS）或 gio trash（Linux） ──

_RM_PATTERN = re.compile(
    r"^(?P<pre>\s*)"
    r"rm\s+(?P<flags>-[a-zA-Z]*\s+)*"
    r"(?P<targets>.+)$",
)


def rewrite_rm_to_trash(command: str) -> str | None:
    """尝试将 rm 命令改写为 trash 命令。成功返回新命令，不适用返回 None。

    仅改写工作空间内的文件删除；系统路径的 rm 不改写（由 check_command 拦截）。
    """
    match = _RM_PATTERN.match(command.strip())
    if match is None:
        return None
    targets = match.group("targets").strip()
    if not targets or targets.startswith("/etc") or targets.startswith("/usr"):
        return None
    # macOS 使用系统 trash 命令（需 brew install trash 或 macos-trash）
    # 回退到 mv 到 ~/.Trash
    return f"mv {targets} ~/.Trash/ 2>/dev/null || rm {command.strip()[3:]}"


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
        """Return False if command matches dangerous patterns (absolute block)."""
        return all(not pat.search(command) for pat in _DANGEROUS_CMD_PATTERNS)

    @staticmethod
    def requires_approval(command: str, policy: SecurityPolicy) -> bool:
        """Return True if command needs user approval before execution."""
        approval = policy.approval
        for pattern in approval.always_allowed_commands:
            if re.search(pattern, command):
                return False
        for pattern in approval.require_approval_patterns:
            if re.search(pattern, command):
                return True
        return False

    @staticmethod
    def is_irrevocable(command: str) -> bool:
        """Return True if command performs an irrevocable external action."""
        return any(pat.search(command) for pat in _IRREVOCABLE_PATTERNS)

    @staticmethod
    def apply_safe_delete(command: str, policy: SecurityPolicy) -> str:
        """如果启用 safe_delete 且命令是 rm，尝试改写为 trash。"""
        if not policy.safe_delete:
            return command
        rewritten = rewrite_rm_to_trash(command)
        return rewritten if rewritten is not None else command
