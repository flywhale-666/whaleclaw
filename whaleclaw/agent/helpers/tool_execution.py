"""Tool registry and execution helpers for the single-agent runtime."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from whaleclaw.agent.context import OnToolCall, OnToolResult
from whaleclaw.agent.helpers.office_rules import (
    is_office_path_probe_command,
    looks_like_ppt_generation_command,
    looks_like_ppt_generation_script,
)
from whaleclaw.providers.base import ToolCall
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.tools.base import ToolResult
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.cron.scheduler import CronScheduler
    from whaleclaw.memory.base import MemoryStore
    from whaleclaw.memory.manager import MemoryManager

log = get_logger(__name__)

_PROJECT_PYTHON = Path(__file__).resolve().parents[3] / "python" / "bin" / "python3.12"
_DIRECT_PY_SCRIPT_RE = re.compile(
    r"^"
    r"(?P<prefix>(?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|[^\s]+)\s+)*)"
    r"(?P<script>(?:~|/|\./|\.\./)[^\s;&|]+\.py)"
    r"(?P<suffix>(?:\s+.*)?)$"
)
_PY_SHELL_MISMATCH_HINTS = (
    "from: command not found",
    "import: command not found",
)
_NANO_BANANA_SCRIPT_HINT = "test_nano_banana"
_NANO_BANANA_MODE_REWRITES = {
    "text2image": "text",
    "txt2img": "text",
    "image2image": "edit",
    "img2img": "edit",
}


def create_default_registry(
    session_manager: SessionManager | None = None,
    cron_scheduler: CronScheduler | None = None,
    *,
    memory_manager: MemoryManager | None = None,
    memory_store: MemoryStore | None = None,
) -> ToolRegistry:
    from whaleclaw.tools.bash import BashTool
    from whaleclaw.tools.browser import BrowserTool
    from whaleclaw.tools.desktop_capture import DesktopCaptureTool
    from whaleclaw.tools.docx_edit import DocxEditTool
    from whaleclaw.tools.file_edit import FileEditTool
    from whaleclaw.tools.file_read import FileReadTool
    from whaleclaw.tools.file_write import FileWriteTool
    from whaleclaw.tools.patch_apply import PatchApplyTool
    from whaleclaw.tools.ppt_edit import PptEditTool
    from whaleclaw.tools.process import ProcessTool
    from whaleclaw.tools.web_fetch import WebFetchTool
    from whaleclaw.tools.xlsx_edit import XlsxEditTool

    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(ProcessTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(PatchApplyTool())
    registry.register(PptEditTool())
    registry.register(DocxEditTool())
    registry.register(XlsxEditTool())
    registry.register(WebFetchTool())
    registry.register(BrowserTool())
    registry.register(DesktopCaptureTool())

    if session_manager is not None:
        from whaleclaw.tools.sessions import (
            SessionsHistoryTool,
            SessionsListTool,
            SessionsSendTool,
        )

        registry.register(SessionsListTool(session_manager))
        registry.register(SessionsHistoryTool(session_manager))
        registry.register(SessionsSendTool(session_manager))

    if cron_scheduler is not None:
        from whaleclaw.tools.cron_tool import CronManageTool
        from whaleclaw.tools.reminder import ReminderTool

        registry.register(CronManageTool(cron_scheduler))
        registry.register(ReminderTool(cron_scheduler))

    from whaleclaw.skills.manager import SkillManager
    from whaleclaw.tools.mcp_manage import McpManageTool
    from whaleclaw.tools.skill_tool import SkillManageTool

    registry.register(SkillManageTool(SkillManager()))
    registry.register(McpManageTool())

    if memory_manager is not None:
        from whaleclaw.tools.memory_tool import MemoryAddTool, MemorySearchTool

        registry.register(MemorySearchTool(memory_manager))
        registry.register(MemoryAddTool(memory_manager))
    if memory_store is not None:
        from whaleclaw.tools.memory_tool import MemoryListTool

        registry.register(MemoryListTool(memory_store))

    return registry


def parse_fallback_tool_calls(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []

    fenced = re.findall(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
    candidates: list[str] = list(fenced)

    if not candidates:
        for match in re.finditer(r"\{[^{}]*\"tool\"[^{}]*\{[^}]*\}[^}]*\}", text):
            candidates.append(match.group(0))
        for match in re.finditer(r"\{[^{}]*\"tool\"[^{}]*\}", text):
            candidates.append(match.group(0))

    for raw in candidates:
        raw = raw.strip()
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        data = cast(dict[str, object], obj)
        raw_name = data.get("tool", "")
        raw_args = data.get("arguments", {})
        if isinstance(raw_name, str) and raw_name and isinstance(raw_args, dict):
            calls.append(
                ToolCall(
                    id=f"fallback_{len(calls)}",
                    name=raw_name,
                    arguments=raw_args,
                )
            )

    return calls


def strip_tool_json(text: str) -> str:
    cleaned = re.sub(
        r'```(?:json)?\s*\n?\s*\{[^`]*"tool"\s*:[^`]*\}\s*\n?\s*```',
        "",
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r'\{\s*"tool"\s*:\s*"[^"]*"[^}]*\}', "", cleaned)
    return cleaned.strip()


async def persist_message(
    manager: SessionManager,
    session: Session,
    role: str,
    content: str,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> None:
    try:
        await manager.add_message(
            session,
            role,
            content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
    except Exception as exc:
        log.debug("agent.persist_failed", error=str(exc))


async def execute_tool(
    registry: ToolRegistry,
    tc: ToolCall,
    *,
    evomap_enabled: bool,
    browser_allowed: bool,
    office_block_bash_probe: bool,
    office_block_message: str,
    office_edit_only: bool,
    office_edit_path: str,
    on_tool_call: OnToolCall | None,
    on_tool_result: OnToolResult | None,
) -> tuple[str, ToolResult]:
    if on_tool_call:
        await on_tool_call(tc.name, tc.arguments)

    t0 = time.monotonic()

    if tc.name == "browser" and not browser_allowed:
        result = ToolResult(
            success=False,
            output="",
            error="请先执行 evomap_fetch；若无命中会自动切换到 browser",
        )
    elif tc.name == "file_write" and office_edit_only:
        content = str(tc.arguments.get("content", ""))
        if looks_like_ppt_generation_script(content):
            result = ToolResult(
                success=False,
                output="",
                error=(
                    "检测到这是修改已有PPT的请求，禁止重新生成新PPT。\n"
                    f"请直接使用 ppt_edit 修改：{office_edit_path}"
                ),
            )
        else:
            result = await _execute_registered_tool(registry, tc)
    elif tc.name == "bash" and office_block_bash_probe:
        raw_command = str(tc.arguments.get("command", ""))
        if is_office_path_probe_command(raw_command):
            result = ToolResult(success=False, output="", error=office_block_message)
        elif office_edit_only and looks_like_ppt_generation_command(raw_command):
            result = ToolResult(
                success=False,
                output="",
                error=(
                    "检测到这是修改已有PPT的请求，禁止重新生成新PPT。\n"
                    f"请直接使用 ppt_edit 修改：{office_edit_path}"
                ),
            )
        else:
            result = await _execute_registered_tool(registry, tc)
            result = await _maybe_retry_after_mkdir(registry, tc, result)
            result = await _maybe_retry_python_script_invocation(registry, tc, result)
    elif tc.name.startswith("evomap_") and not evomap_enabled:
        result = ToolResult(
            success=False,
            output="",
            error="EvoMap 已关闭，请先在设置中开启",
        )
    else:
        result = await _execute_registered_tool(registry, tc)
        if tc.name == "bash":
            result = await _maybe_retry_after_mkdir(registry, tc, result)
            result = await _maybe_retry_python_script_invocation(registry, tc, result)

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "agent.tool_exec",
        tool=tc.name,
        success=result.success,
        elapsed_ms=elapsed_ms,
        args_preview=str(tc.arguments)[:200],
    )

    if on_tool_result:
        await on_tool_result(tc.name, result)

    return tc.id, result


async def _execute_registered_tool(registry: ToolRegistry, tc: ToolCall) -> ToolResult:
    tool = registry.get(tc.name)
    if tool is None:
        return ToolResult(success=False, output="", error=f"未知工具: {tc.name}")
    try:
        return await tool.execute(**tc.arguments)
    except Exception as exc:
        return ToolResult(success=False, output="", error=str(exc))


async def _maybe_retry_after_mkdir(
    registry: ToolRegistry,
    tc: ToolCall,
    result: ToolResult,
) -> ToolResult:
    tool = registry.get(tc.name)
    if result.success or tool is None:
        return result
    missing_target = can_auto_create_parent_for_failure(result)
    if not missing_target:
        return result
    parent = str(Path(missing_target).expanduser().resolve().parent)
    try:
        mkdir_result = await tool.execute(command=f"mkdir -p '{parent}'", timeout=30)
        if mkdir_result.success:
            return await tool.execute(**tc.arguments)
    except Exception:
        pass
    return result


def _rewrite_direct_python_script_command(command: str) -> str:
    if not _PROJECT_PYTHON.is_file():
        return command
    match = _DIRECT_PY_SCRIPT_RE.match(command.strip())
    if match is None:
        return command
    prefix = match.group("prefix")
    script = match.group("script")
    suffix = match.group("suffix") or ""
    return f"{prefix}{_PROJECT_PYTHON} {script}{suffix}"


def _is_python_shell_mismatch(result: ToolResult) -> bool:
    if result.success:
        return False
    text = f"{result.error or ''}\n{result.output or ''}".lower()
    return any(hint in text for hint in _PY_SHELL_MISMATCH_HINTS)


async def _maybe_retry_python_script_invocation(
    registry: ToolRegistry,
    tc: ToolCall,
    result: ToolResult,
) -> ToolResult:
    tool = registry.get(tc.name)
    if result.success or tool is None or tc.name != "bash":
        return result
    if not _is_python_shell_mismatch(result):
        return result
    raw_command = str(tc.arguments.get("command", "")).strip()
    if not raw_command:
        return result
    rewritten = _rewrite_direct_python_script_command(raw_command)
    if rewritten == raw_command:
        return result
    log.warning(
        "agent.bash_python_script_retry",
        original=raw_command[:200],
        rewritten=rewritten[:200],
    )
    retry_args = dict(tc.arguments)
    retry_args["command"] = rewritten
    try:
        return await tool.execute(**retry_args)
    except Exception:
        return result


_ERROR_SUMMARY_MAX_CHARS = 300

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):")
_FINAL_ERROR_RE = re.compile(
    r"^(\w+(?:\.\w+)*(?:Error|Exception|Warning|Fault|Interrupt|Exit).*)",
    re.MULTILINE,
)
_USER_FRAME_RE = re.compile(
    r'^\s+File "(/tmp/[^"]+)", line (\d+)',
    re.MULTILINE,
)
_NON_TB_TAIL_LINES = 3


def _summarize_error(text: str) -> str:
    """从错误输出中提取摘要：错误类型 + 用户脚本位置 + 非堆栈关键行。"""
    lines = text.splitlines()
    if len(lines) <= 5:
        return text

    parts: list[str] = []

    # 提取 traceback 之前的内容（如 "生图失败: HTTP 422" / API 响应）
    tb_match = _TRACEBACK_RE.search(text)
    if tb_match:
        pre_tb = text[:tb_match.start()].strip()
        if pre_tb:
            pre_lines = pre_tb.splitlines()
            for ln in pre_lines[:3]:
                ln = ln.strip()
                if ln and ln != "[stderr]":
                    parts.append(ln)

    # 提取最终错误行（如 "ModuleNotFoundError: No module named pptx"）
    final_errors: list[str] = []
    for m in _FINAL_ERROR_RE.finditer(text):
        final_errors.append(m.group(1).strip())
    if final_errors:
        parts.append(final_errors[-1])

    # 提取用户脚本触发位置（/tmp/ 下的文件，非 site-packages）
    user_frame = _USER_FRAME_RE.search(text)
    if user_frame:
        parts.append(f"触发位置: {user_frame.group(1)} 第{user_frame.group(2)}行")

    # 无 traceback 的长输出：取首尾各几行
    if not tb_match and not final_errors:
        head = lines[:2]
        tail = lines[-_NON_TB_TAIL_LINES:]
        omitted = len(lines) - len(head) - len(tail)
        if omitted > 0:
            return "\n".join(head + [f"  ... (省略 {omitted} 行) ..."] + tail)
        return text

    if not parts:
        return text if len(text) <= _ERROR_SUMMARY_MAX_CHARS else text[:_ERROR_SUMMARY_MAX_CHARS]
    return "\n".join(parts)


def format_tool_output(result: ToolResult) -> str:
    """格式化工具输出，当轮保持完整不压缩。

    压缩/截断统一由 ContextWindow 在消息进入历史时处理。
    """
    if result.success:
        return result.output or "(empty output)"
    error_part = result.error or "unknown error"
    output_part = result.output or ""
    raw = f"[ERROR] {error_part}\n{output_part}".strip()
    diagnosis = diagnose_failure_hint(result)
    if diagnosis:
        raw += f"\n[DIAGNOSIS] {diagnosis}"
    return raw


def is_transient_cli_usage_error(result: ToolResult) -> bool:
    """Return whether a failed tool result is just a CLI usage banner."""
    if result.success:
        return False
    text = f"{result.error or ''}\n{result.output or ''}".lower()
    return "usage:" in text and "error:" in text and "--help" not in text


def is_nano_banana_cli_param_error(result: ToolResult) -> bool:
    """Return whether Nano Banana failed before execution due to bad CLI args."""
    if not is_transient_cli_usage_error(result):
        return False
    text = f"{result.error or ''}\n{result.output or ''}".lower()
    if "usage: test_nano_banana_2.py" not in text:
        return False
    param_error_markers = (
        "invalid choice",
        "unrecognized arguments",
        "the following arguments are required",
        "argument --",
    )
    return any(marker in text for marker in param_error_markers)


def is_nano_banana_preflight_error(result: ToolResult) -> bool:
    """Return whether Nano Banana failed before the real image flow started."""
    return is_nano_banana_cli_param_error(result) or _is_python_shell_mismatch(result)


def diagnose_failure_hint(result: ToolResult) -> str:
    text = f"{result.error or ''}\n{result.output or ''}"
    if "No such file or directory" in text and "FileNotFoundError" in text:
        return "更可能是目标路径或上级目录不存在，请先创建目录再写文件，不是依赖缺失。"
    if "ModuleNotFoundError" in text:
        return "这是依赖缺失，请安装缺失模块后重试。"
    if "Permission denied" in text:
        return "这是权限问题，请检查目标路径可写权限。"
    return ""


def extract_missing_target_path(text: str) -> str:
    match = re.search(r"No such file or directory:\s*['\"](/[^'\"]+)['\"]", text)
    if not match:
        return ""
    path = match.group(1).strip()
    return path if path else ""


def can_auto_create_parent_for_failure(result: ToolResult) -> str:
    text = f"{result.error or ''}\n{result.output or ''}"
    if "FileNotFoundError" not in text or "No such file or directory" not in text:
        return ""
    target = extract_missing_target_path(text)
    if not target:
        return ""
    suffix = Path(target).suffix.lower()
    if suffix not in {".pptx", ".docx", ".xlsx", ".pdf", ".html", ".md", ".txt", ".py"}:
        return ""
    return target


def validate_tool_call_args(tc: ToolCall, registry: ToolRegistry) -> str | None:
    tool = registry.get(tc.name)
    if tool is None:
        return None
    for param in tool.definition.parameters:
        if not param.required:
            continue
        if param.name not in tc.arguments:
            return f"{tc.name}.{param.name} 缺失"
        value = tc.arguments.get(param.name)
        if isinstance(value, str) and not value.strip():
            return f"{tc.name}.{param.name} 为空"
    if tc.name == "browser":
        action = str(tc.arguments.get("action", "")).strip()
        if not action:
            return "browser.action 为空"
        action_reqs: dict[str, tuple[str, ...]] = {
            "navigate": ("url",),
            "click": ("selector",),
            "type": ("selector", "text"),
            "evaluate": ("script",),
            "search_images": ("text",),
        }
        for field in action_reqs.get(action, ()):
            value = tc.arguments.get(field)
            if isinstance(value, str):
                if not value.strip():
                    return f"browser.{field} 为空"
            elif value is None:
                return f"browser.{field} 缺失"
    if tc.name == "file_edit":
        old_string = tc.arguments.get("old_string")
        new_string = tc.arguments.get("new_string")
        for field_name, field_val in (("old_string", old_string), ("new_string", new_string)):
            if not isinstance(field_val, str):
                continue
            if field_val.count("\\n") >= 3 and "\n" not in field_val:
                return f"file_edit.{field_name} 疑似转义块文本，改用 file_write 重写文件"
    return None


def is_non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def first_non_empty_arg(arguments: dict[str, object], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = arguments.get(name)
        if is_non_empty_str(value):
            return str(value).strip()
    return None


def looks_like_image_request(text: str) -> bool:
    lower = text.lower()
    keywords = (
        "图",
        "图片",
        "照片",
        "近照",
        "头像",
        "搜图",
        "找图",
        "壁纸",
        "海报",
        "背景",
        "写真",
        "image",
        "photo",
        "picture",
        "wallpaper",
        "poster",
    )
    return any(keyword.lower() in lower for keyword in keywords)


def is_garbled_query(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    escaped_noise = (
        stripped.count("\\n")
        + stripped.count("\\t")
        + stripped.count("\\r")
        + stripped.count("\\x")
    )
    if escaped_noise >= 2:
        return True
    if (
        re.search(r"(?:\\+[nrt]\d*){2,}", stripped)
        or re.search(r"(?:\n\d*){2,}", stripped)
        or stripped.count("\\x") >= 2
    ):
        return True
    return bool(len(stripped) > 40 and len(set(stripped)) < 10)


def _repair_nano_banana_bash_command(command: str) -> tuple[str, bool]:
    """Normalize common LLM mistakes for Nano Banana CLI invocations."""
    if _NANO_BANANA_SCRIPT_HINT not in command:
        return command, False

    repaired = command

    for wrong_mode, right_mode in _NANO_BANANA_MODE_REWRITES.items():
        repaired = re.sub(
            rf"(--mode\s+)(['\"]?){re.escape(wrong_mode)}\2",
            rf"\1\2{right_mode}\2",
            repaired,
            flags=re.IGNORECASE,
        )

    repaired = re.sub(r"(?<!\S)--api-base(?!\S)", "--base-url", repaired)

    if "--aspect-ratio" not in repaired:
        repaired = re.sub(
            r"(--size\s+)(['\"]?)(\d{1,2}:\d{1,2})\2",
            r"--aspect-ratio \2\3\2",
            repaired,
            flags=re.IGNORECASE,
        )

    return repaired, repaired != command


def repair_tool_call(tc: ToolCall, user_message: str) -> tuple[ToolCall, str | None]:
    args: dict[str, object] = dict(tc.arguments)
    changed = False
    reasons: list[str] = []

    alias_by_tool: dict[str, dict[str, tuple[str, ...]]] = {
        "bash": {"command": ("cmd", "script", "shell")},
        "browser": {"text": ("query", "keyword", "keywords", "q"), "selector": ("css",)},
        "file_read": {"path": ("file", "file_path")},
        "file_write": {"path": ("file", "file_path"), "content": ("text", "code")},
        "file_edit": {
            "path": ("file", "file_path"),
            "old_string": ("old", "old_text"),
            "new_string": ("new", "new_text"),
        },
    }
    alias_map = alias_by_tool.get(tc.name, {})
    for canonical, aliases in alias_map.items():
        if is_non_empty_str(args.get(canonical)):
            continue
        value = first_non_empty_arg(args, aliases)
        if value is None:
            continue
        args[canonical] = value
        changed = True
        reasons.append(f"{canonical}<-alias")

    if tc.name == "bash" and is_non_empty_str(args.get("command")):
        repaired_command, command_changed = _repair_nano_banana_bash_command(
            str(args.get("command", ""))
        )
        if command_changed:
            args["command"] = repaired_command
            changed = True
            reasons.append("command<-nano-banana-cli")

    if tc.name == "browser":
        action = str(args.get("action", "")).strip().lower()
        inferred_action: str | None = None

        if not action:
            if is_non_empty_str(args.get("url")):
                inferred_action = "navigate"
            elif is_non_empty_str(args.get("script")):
                inferred_action = "evaluate"
            elif is_non_empty_str(args.get("selector")) and is_non_empty_str(args.get("text")):
                inferred_action = "type"
            elif is_non_empty_str(args.get("selector")):
                if any(token in user_message for token in ("读取", "提取", "文本", "内容", "text")):
                    inferred_action = "get_text"
                else:
                    inferred_action = "click"
            else:
                query = first_non_empty_arg(args, ("text", "query", "keyword", "keywords", "q"))
                if query is not None and looks_like_image_request(user_message):
                    args["text"] = query
                    inferred_action = "search_images"

            if inferred_action is not None:
                args["action"] = inferred_action
                action = inferred_action
                changed = True
                reasons.append("action<-inferred")

        if action == "navigate" and not is_non_empty_str(args.get("url")):
            candidate = first_non_empty_arg(args, ("text",))
            if candidate and candidate.startswith(("http://", "https://")):
                args["url"] = candidate
                changed = True
                reasons.append("url<-text")
        if action == "search_images":
            query = str(args.get("text", "")).strip()
            if is_garbled_query(query) and looks_like_image_request(user_message):
                args["text"] = user_message.strip()
                changed = True
                reasons.append("text<-user_message")

    if not changed:
        return tc, None
    return ToolCall(id=tc.id, name=tc.name, arguments=args), ",".join(reasons)


__all__ = [
    "can_auto_create_parent_for_failure",
    "create_default_registry",
    "execute_tool",
    "first_non_empty_arg",
    "format_tool_output",
    "is_nano_banana_cli_param_error",
    "is_nano_banana_preflight_error",
    "is_transient_cli_usage_error",
    "is_non_empty_str",
    "parse_fallback_tool_calls",
    "persist_message",
    "repair_tool_call",
    "strip_tool_json",
    "validate_tool_call_args",
]
