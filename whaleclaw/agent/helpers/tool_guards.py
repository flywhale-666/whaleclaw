"""Tool guard helpers for the single-agent runtime.

两套独立熔断机制并行运行：
- 搜图专用熔断：只管 search_images，不参与通用签名
- 通用模糊熔断：管所有其他工具，双层签名（精确 + 模糊）取较大值
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from whaleclaw.agent.helpers.tool_execution import (
    is_nano_banana_preflight_error as _is_nano_banana_preflight_error,
)
from whaleclaw.agent.helpers.image_search import (
    extract_planned_image_count,
    is_search_images_call,
    normalize_search_images_query,
)
from whaleclaw.providers.base import ToolCall
from whaleclaw.tools.base import ToolResult

_MCPORTER_CALL_RE = re.compile(
    r"^(mcporter\s+call\s+\S+\s+\S+)",
)
_BASH_NOISE_RE = re.compile(
    r"""(?x)
    \s+--output\s+\S+
    | \s+2>\s*(?:/dev/null|\$null|&1)
    | \s+\d+>\s*(?:/dev/null|\$null)
    | \s+--?\w+=\S+
    | \s+--\w+(?:\s+(?!-)\S+)?
    """,
)


@dataclass(slots=True)
class GuardLogEvent:
    level: Literal["info", "warning"]
    event: str
    fields: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ToolGuardUpdate:
    conversation_messages: list[str] = field(default_factory=list)
    final_texts: list[str] = field(default_factory=list)
    log_events: list[GuardLogEvent] = field(default_factory=list)
    stop_for_probe_loop: bool = False
    stop_for_repeat_loop: bool = False


@dataclass(slots=True)
class ToolGuardState:
    recent_exact_signatures: list[str] = field(default_factory=list)
    recent_fuzzy_signatures: list[str] = field(default_factory=list)
    loop_detect_window: int = 3
    loop_warning_signature: str = ""
    loop_block_signature: str = ""
    browser_fail_streak: int = 0
    search_images_count: int = 0
    planned_image_count: int | None = None
    search_images_limit: int | None = None
    search_images_blocked_reason: str = ""
    search_query_repeat_streak: int = 0
    last_search_query: str = ""
    bash_fail_streak: int = 0
    same_failed_bash_streak: int = 0
    last_failed_bash_signature: str = ""
    blocked_tools: set[str] = field(default_factory=set)
    low_value_bash_probe_streak: int = 0
    # 兼容旧字段名（测试 / 外部引用）
    @property
    def recent_signatures(self) -> list[str]:
        return self.recent_exact_signatures


def is_low_value_bash_probe(tc: ToolCall) -> bool:
    if tc.name != "bash":
        return False
    raw = str(tc.arguments.get("command", "")).strip().lower()
    if not raw:
        return False
    probe_hints = ("ls ", "ls\t", "stat ", "test -f ", "test -e ", "echo ")
    risky_hints = (
        "python ",
        "python3 ",
        "cp ",
        "mv ",
        "rm ",
        "sed ",
        "awk ",
        "perl ",
        "open ",
        "soffice ",
    )
    if any(h in raw for h in risky_hints):
        return False
    return any(h in raw for h in probe_hints)


def normalize_bash_command_signature(command: str) -> str:
    """Normalize bash command text for repeated-failure detection."""
    return re.sub(r"\s+", " ", command.strip())


def _fuzzy_signature_for_tool(tc: ToolCall) -> str:
    """生成工具调用的模糊签名，忽略参数值的差异。

    规则：
    - bash (mcporter call): 只保留 命令 + 服务名 + 工具名
    - bash (普通命令): 去掉 key=value、--flag、重定向等噪音
    - browser: action + url/text（不同目标 = 不同签名）
    - search_images: 不参与通用熔断，返回空
    - 其他工具: 工具名 + 参数键名（忽略参数值）
    """
    if tc.name == "browser":
        action = str(tc.arguments.get("action", "")).strip().lower()
        if action == "search_images":
            return ""
        target = str(
            tc.arguments.get("url", "") or tc.arguments.get("text", "")
        ).strip()
        return f"browser:{action}:{target}"

    if tc.name == "bash":
        raw = str(tc.arguments.get("command", "")).strip()
        m = _MCPORTER_CALL_RE.match(raw)
        if m:
            return f"bash:{re.sub(r'\\s+', ' ', m.group(1))}"
        skeleton = _BASH_NOISE_RE.sub("", raw)
        skeleton = re.sub(r"\s+", " ", skeleton).strip()
        return f"bash:{skeleton}"

    keys = sorted(tc.arguments.keys())
    return f"{tc.name}:{','.join(keys)}"


def tail_repeat_count(items: list[str]) -> int:
    """Count how many identical signatures repeat at the end of a sequence."""
    if not items:
        return 0
    last = items[-1]
    count = 0
    for item in reversed(items):
        if item != last:
            break
        count += 1
    return count


def is_progress_stage_tool_call(tc: ToolCall) -> bool:
    if tc.name in {
        "file_write",
        "file_edit",
        "patch_apply",
        "ppt_edit",
        "docx_edit",
        "xlsx_edit",
    }:
        return True
    return tc.name == "bash" and not is_low_value_bash_probe(tc)


def update_planned_image_count(
    state: ToolGuardState,
    content: str,
) -> None:
    if state.planned_image_count is not None:
        return
    detected_count = extract_planned_image_count(content)
    if detected_count is None:
        return
    state.planned_image_count = detected_count
    # max(计划配图数 × 1.5 + 1, 计划配图数 + 1)
    state.search_images_limit = max(int(detected_count * 1.5 + 1), detected_count + 1)


def blocked_tool_reasons(
    tool_calls: list[ToolCall],
    state: ToolGuardState,
) -> list[str]:
    reasons = [
        f"{tc.name} 已熔断，禁止继续调用"
        for tc in tool_calls
        if tc.name in state.blocked_tools
    ]
    if state.search_images_blocked_reason:
        reasons.extend(
            state.search_images_blocked_reason
            for tc in tool_calls
            if is_search_images_call(tc)
        )
    return reasons


def apply_tool_result_guards(
    state: ToolGuardState,
    tc: ToolCall,
    result: ToolResult,
    *,
    office_loop_guard_enabled: bool,
    image_api_probe_guard_enabled: bool,
    session_id: str | None,
) -> ToolGuardUpdate:
    update = ToolGuardUpdate()

    if tc.name == "browser":
        action = str(tc.arguments.get("action", "")).strip().lower()
        if action == "search_images" and result.success:
            state.search_images_count += 1
            query_sig = normalize_search_images_query(tc)
            if query_sig and query_sig == state.last_search_query:
                state.search_query_repeat_streak += 1
            elif query_sig:
                state.last_search_query = query_sig
                state.search_query_repeat_streak = 1
            else:
                state.last_search_query = ""
                state.search_query_repeat_streak = 0
        if result.success:
            state.browser_fail_streak = 0
        else:
            state.browser_fail_streak += 1
            if state.browser_fail_streak >= 2 and "browser" not in state.blocked_tools:
                state.blocked_tools.add("browser")
                update.log_events.append(
                    GuardLogEvent(
                        level="warning",
                        event="agent.tool_circuit_open",
                        fields={
                            "tool": "browser",
                            "fail_streak": state.browser_fail_streak,
                            "session_id": session_id or "",
                        },
                    )
                )
                update.conversation_messages.append(
                    "[系统降级] browser 工具连续失败，已自动熔断。"
                    "后续请不要再调用 browser。"
                    "请改用 bash 工具执行可复现的命令行方案完成任务。"
                )

    if tc.name == "bash":
        is_timeout = bool(result.error and "命令超时" in result.error)
        if result.success:
            state.bash_fail_streak = 0
            state.same_failed_bash_streak = 0
            state.last_failed_bash_signature = ""
        else:
            cmd_text = str(tc.arguments.get("command", ""))
            is_image_gen = "test_nano_banana" in cmd_text
            if is_image_gen:
                # 参数/命令行校验错误尚未真正进入生图流程，允许修正参数后立即重试。
                if _is_nano_banana_preflight_error(result):
                    update.conversation_messages.append(
                        "[系统提示] 上一次 Nano Banana 命令在进入实际生图流程前就失败了。"
                        "如果是参数错误或脚本调用方式错误，允许你立刻修正后重试一次；"
                        "但必须修改错误点，禁止原样重试。"
                    )
                    return update

                # 真正进入生图流程后的失败独立处理，不参与通用 bash 熔断计数。
                update.conversation_messages.append(
                    "[系统提示] 本次生图失败，禁止自动重试。"
                    "继续执行剩余任务，最终将成功和失败的结果一并回复用户，"
                    "由用户决定是否对失败的图重新操作。"
                )
                return update
            state.bash_fail_streak += 1
            failed_sig = normalize_bash_command_signature(cmd_text)
            if failed_sig and failed_sig == state.last_failed_bash_signature:
                state.same_failed_bash_streak += 1
            elif failed_sig:
                state.same_failed_bash_streak = 1
                state.last_failed_bash_signature = failed_sig
            else:
                state.same_failed_bash_streak = 0
                state.last_failed_bash_signature = ""
            if is_timeout and state.same_failed_bash_streak < 3:
                update.conversation_messages.append(
                    "[系统提示] bash 命令超时，子进程已被终止。"
                    "如需重试，请检查参数并增大 timeout。"
                    "禁止不改参数直接重试。"
                )
            if state.same_failed_bash_streak >= 3 and "bash" not in state.blocked_tools:
                state.blocked_tools.add("bash")
                update.log_events.append(
                    GuardLogEvent(
                        level="warning",
                        event="agent.tool_circuit_open",
                        fields={
                            "tool": "bash",
                            "fail_streak": state.bash_fail_streak,
                            "same_failed_streak": state.same_failed_bash_streak,
                            "command_signature": state.last_failed_bash_signature[:200],
                            "session_id": session_id or "",
                        },
                    )
                )
                update.conversation_messages.append(
                    "[系统降级] 同一 bash 命令模板已连续失败 3 次，"
                    "已自动熔断并切换策略。"
                    "后续请不要再调用 bash。"
                    "请改用结构化编辑工具（ppt_edit/docx_edit/xlsx_edit）"
                    "或文件工具（file_read/file_write/file_edit）继续。"
                )

    if (office_loop_guard_enabled or image_api_probe_guard_enabled) and is_low_value_bash_probe(tc):
        state.low_value_bash_probe_streak += 1
    else:
        state.low_value_bash_probe_streak = 0
    if image_api_probe_guard_enabled and state.low_value_bash_probe_streak >= 2:
        update.final_texts.append(
            "检测到连续探测环境但未进入实测，已停止循环以避免卡住。"
            "下一步我将直接执行最小生图脚本（/tmp）并返回状态码与图片路径。"
        )
        update.stop_for_probe_loop = True
        return update
    if office_loop_guard_enabled and state.low_value_bash_probe_streak >= 3:
        update.final_texts.append(
            "检测到连续的文件探测命令（如 ls/stat/test）且无实质修改，"
            "已停止循环以避免卡住。"
            "我将改用文档局部编辑工具继续，请直接告诉我要改哪一页/哪一段/哪个单元格。"
        )
        update.stop_for_probe_loop = True
        return update

    if result.success and is_progress_stage_tool_call(tc):
        state.search_query_repeat_streak = 0
        state.last_search_query = ""
    return update


def apply_post_round_guards(
    state: ToolGuardState,
    tool_calls: list[ToolCall],
    *,
    round_idx: int,
    session_id: str | None,
) -> ToolGuardUpdate:
    update = ToolGuardUpdate()

    if (
        not state.search_images_blocked_reason
        and state.search_images_limit is not None
        and state.search_images_count > state.search_images_limit
    ):
        state.search_images_blocked_reason = (
            "search_images 已超过计划配图数限制，禁止继续调用"
        )
        update.log_events.append(
            GuardLogEvent(
                level="warning",
                event="agent.search_images_over_plan",
                fields={
                    "session_id": session_id or "",
                    "planned_image_count": state.planned_image_count or 0,
                    "search_images_limit": state.search_images_limit,
                    "count": state.search_images_count,
                },
            )
        )
        update.conversation_messages.append(
            "[系统提示] 你已超过计划配图数的搜索额度。"
            f"计划配图数={state.planned_image_count}，"
            f"允许搜索上限={state.search_images_limit}，"
            f"当前已搜索={state.search_images_count}。"
            "禁止再调用 search_images。请立即进入生成或编辑步骤。"
        )
    if not state.search_images_blocked_reason and state.search_query_repeat_streak >= 3:
        state.search_images_blocked_reason = (
            "search_images 在重复相同搜索词且未取得新进展，禁止继续调用"
        )
        update.log_events.append(
            GuardLogEvent(
                level="warning",
                event="agent.search_images_repeat_query_blocked",
                fields={
                    "session_id": session_id or "",
                    "query": state.last_search_query[:200],
                    "repeat_streak": state.search_query_repeat_streak,
                },
            )
        )
        update.conversation_messages.append(
            "[系统提示] 检测到你在重复使用相同搜索词搜图且未取得新进展。"
            f"重复搜索词：{state.last_search_query or '(空)'}。"
            "禁止继续调用 search_images。请立即进入生成或编辑步骤。"
        )

    # ── 双层签名：精确 + 模糊（搜图不参与通用签名） ──
    non_search_calls = [tc for tc in tool_calls if not is_search_images_call(tc)]
    if not non_search_calls:
        return update

    exact_parts: list[str] = []
    fuzzy_parts: list[str] = []
    for tc in non_search_calls:
        arg_str = json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)[:200]
        exact_parts.append(f"{tc.name}:{arg_str}")
        fsig = _fuzzy_signature_for_tool(tc)
        if fsig:
            fuzzy_parts.append(fsig)

    exact_sig = hashlib.md5("|".join(exact_parts).encode()).hexdigest()  # noqa: S324
    fuzzy_sig = hashlib.md5("|".join(fuzzy_parts).encode()).hexdigest() if fuzzy_parts else exact_sig  # noqa: S324

    state.recent_exact_signatures.append(exact_sig)
    state.recent_fuzzy_signatures.append(fuzzy_sig)
    exact_repeat = tail_repeat_count(state.recent_exact_signatures)
    fuzzy_repeat = tail_repeat_count(state.recent_fuzzy_signatures)
    repeat_count = max(exact_repeat, fuzzy_repeat)

    repeated_tools = sorted({tc.name for tc in non_search_calls})
    repeated_tools_text = "、".join(repeated_tools) or "当前工具"

    # 合并签名用于去重（避免同一轮多次触发）
    combined_sig = f"{exact_sig}:{fuzzy_sig}"

    # ── 三级递进：3 轮警告 → 4 轮熔断 → 5 轮中止 ──
    warn_threshold = state.loop_detect_window          # 3
    block_threshold = state.loop_detect_window + 1     # 4
    abort_threshold = state.loop_detect_window + 2     # 5

    if (
        repeat_count >= warn_threshold
        and combined_sig != state.loop_warning_signature
    ):
        state.loop_warning_signature = combined_sig
        update.log_events.append(
            GuardLogEvent(
                level="warning",
                event="agent.loop_detected",
                fields={
                    "session_id": session_id or "",
                    "rounds": round_idx + 1,
                    "repeated_tools": repeated_tools,
                    "exact_repeat": exact_repeat,
                    "fuzzy_repeat": fuzzy_repeat,
                },
            )
        )
        update.conversation_messages.append(
            "[系统提示] 检测到你在重复执行相似操作且未取得进展。"
            f"重复工具：{repeated_tools_text}。"
            "请跳过前置检查，直接执行核心操作。"
        )
    if (
        repeat_count >= block_threshold
        and combined_sig != state.loop_block_signature
    ):
        state.loop_block_signature = combined_sig
        newly_blocked = [name for name in repeated_tools if name not in state.blocked_tools]
        if newly_blocked:
            state.blocked_tools.update(newly_blocked)
            update.log_events.append(
                GuardLogEvent(
                    level="warning",
                    event="agent.loop_circuit_open",
                    fields={
                        "session_id": session_id or "",
                        "rounds": round_idx + 1,
                        "blocked_tools": newly_blocked,
                        "exact_repeat": exact_repeat,
                        "fuzzy_repeat": fuzzy_repeat,
                    },
                )
            )
            update.conversation_messages.append(
                "[系统降级] 检测到重复路径仍在继续，已自动熔断这些工具："
                f"{'、'.join(newly_blocked)}。"
                "后续禁止继续调用它们。"
                "请改用其他工具或更直接的方案完成任务。"
            )
    if repeat_count >= abort_threshold:
        update.final_texts.append(
            "检测到任务陷入重复循环，已自动中止当前路径。"
            "请让我改用更直接的方式继续完成任务。"
        )
        update.stop_for_repeat_loop = True
    return update


__all__ = [
    "GuardLogEvent",
    "ToolGuardState",
    "ToolGuardUpdate",
    "apply_post_round_guards",
    "apply_tool_result_guards",
    "blocked_tool_reasons",
    "is_low_value_bash_probe",
    "is_progress_stage_tool_call",
    "normalize_bash_command_signature",
    "tail_repeat_count",
    "update_planned_image_count",
    "_fuzzy_signature_for_tool",
]
