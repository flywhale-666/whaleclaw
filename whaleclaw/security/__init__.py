"""Security subsystem — permissions, pairing, sandbox, audit."""

from whaleclaw.security.audit import AuditEvent, AuditLogger
from whaleclaw.security.pairing import AllowListStore, PairingRequest, PairingService
from whaleclaw.security.permissions import (
    ApprovalDecision,
    PermissionChecker,
    SecurityPolicy,
    ToolApprovalPolicy,
    ToolPermission,
    rewrite_rm_to_trash,
)
from whaleclaw.security.sandbox import DockerSandbox, SandboxConfig, SandboxMode

__all__ = [
    "AllowListStore",
    "ApprovalDecision",
    "AuditEvent",
    "AuditLogger",
    "DockerSandbox",
    "PairingRequest",
    "PairingService",
    "PermissionChecker",
    "SandboxConfig",
    "SandboxMode",
    "SecurityPolicy",
    "ToolApprovalPolicy",
    "ToolPermission",
    "rewrite_rm_to_trash",
]
