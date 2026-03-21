"""Tool guard helpers for the single-agent runtime.

三类独立守卫并行运行，输出统一 ``GuardDecision``:

- FailureGuard: 工具执行连续失败 → block_tool
- LoopGuard: 重复路径未推进 → warn / block / abort
- DomainGuard: 业务预算约束 (search_images / 探测循环)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Literal

from whaleclaw.skills.hooks import SkillHooks
from whaleclaw.agent.helpers.image_search import (
    extract_planned_image_count,
    is_search_images_call,
    normalize_search_images_query,
)
from whaleclaw.providers.base import ToolCall
from whaleclaw.tools.base import ToolResult

# ── 集中阈值常量 ──────────────────────────────────────────────
BROWSER_WARN_LIMIT = 2
BROWSER_BLOCK_LIMIT = 3
BASH_WARN_LIMIT = 2
BASH_BLOCK_LIMIT = 3
REPEAT_WARN_ROUNDS = 3
REPEAT_BLOCK_ROUNDS = 4
REPEAT_ABORT_ROUNDS = 5
SEARCH_IMAGES_REPEAT_QUERY_LIMIT = 3
MODEL_REPAIR_RETRY_LIMIT = 2

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


# ── 统一决策模型 ──────────────────────────────────────────────

@dataclass(slots=True)
class GuardDecision:
    """单条守卫决策。上层只消费这个结构，不再理解多种零散字段。"""

    kind: Literal["warn", "block", "abort", "hint"]
    scope: str
    reason_code: str
    message: str
    blocked_tools: list[str] = field(default_factory=list)
    stop_current_path: bool = False


@dataclass(slots=True)
class GuardLogEvent:
    level: Literal["info", "warning"]
    event: str
    fields: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ToolGuardUpdate:
    """单轮 guard 的聚合输出。

    ``decisions`` 是新增的统一结构；``conversation_messages`` /
    ``final_texts`` / ``stop_for_*`` 保留作兼容层，由内部从 decisions 回填。
    """

    conversation_messages: list[str] = field(default_factory=list)
    final_texts: list[str] = field(default_factory=list)
    log_events: list[GuardLogEvent] = field(default_factory=list)
    stop_for_probe_loop: bool = False
    stop_for_repeat_loop: bool = False
    decisions: list[GuardDecision] = field(default_factory=list)


def _backfill_from_decisions(update: ToolGuardUpdate) -> None:
    """从 ``decisions`` 回填旧兼容字段，保证上层不改也能正常工作。"""
    for d in update.decisions:
        if d.kind in ("warn", "hint"):
            update.conversation_messages.append(d.message)
        elif d.kind == "block":
            update.conversation_messages.append(d.message)
        elif d.kind == "abort":
            update.final_texts.append(d.message)
            if d.scope.startswith("probe:"):
                update.stop_for_probe_loop = True
            else:
                update.stop_for_repeat_loop = True


# ── 状态模型（按守卫职责分组） ──────────────────────────────

@dataclass(slots=True)
class ToolGuardState:
    # -- FailureGuard --
    browser_fail_streak: int = 0
    bash_fail_streak: int = 0
    same_failed_bash_streak: int = 0
    last_failed_bash_signature: str = ""

    # -- LoopGuard --
    recent_exact_signatures: list[str] = field(default_factory=list)
    recent_fuzzy_signatures: list[str] = field(default_factory=list)
    loop_detect_window: int = REPEAT_WARN_ROUNDS
    loop_warning_signature: str = ""
    loop_block_signature: str = ""

    # -- DomainGuard (search_images) --
    search_images_count: int = 0
    planned_image_count: int | None = None
    search_images_limit: int | None = None
    search_images_blocked_reason: str = ""
    search_query_repeat_streak: int = 0
    last_search_query: str = ""

    # -- 统一封禁记录 --
    blocked_tools: set[str] = field(default_factory=set)

    @property
    def recent_signatures(self) -> list[str]:
        return self.recent_exact_signatures


# ── 工具函数 ──────────────────────────────────────────────────

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


# ── FailureGuard ──────────────────────────────────────────────

def _apply_failure_guard(
    state: ToolGuardState,
    tc: ToolCall,
    result: ToolResult,
    *,
    session_id: str | None,
    skill_hooks: list[SkillHooks] | None = None,
) -> ToolGuardUpdate:
    """处理工具执行连续失败的熔断逻辑。"""
    update = ToolGuardUpdate()

    if tc.name == "browser":
        if result.success:
            state.browser_fail_streak = 0
        else:
            state.browser_fail_streak += 1
            if (
                state.browser_fail_streak >= BROWSER_WARN_LIMIT
                and state.browser_fail_streak < BROWSER_BLOCK_LIMIT
                and "browser" not in state.blocked_tools
            ):
                update.decisions.append(GuardDecision(
                    kind="warn",
                    scope="tool:browser",
                    reason_code="browser_failure_warn",
                    message=(
                        "[系统提示] browser 工具已连续失败，请检查调用方式。"
                        "如果继续失败将自动熔断。"
                    ),
                ))
                update.log_events.append(GuardLogEvent(
                    level="warning",
                    event="agent.tool_failure_warn",
                    fields={
                        "tool": "browser",
                        "fail_streak": state.browser_fail_streak,
                        "session_id": session_id or "",
                    },
                ))
            if (
                state.browser_fail_streak >= BROWSER_BLOCK_LIMIT
                and "browser" not in state.blocked_tools
            ):
                state.blocked_tools.add("browser")
                msg = (
                    "[系统降级] browser 工具连续失败，已自动熔断。"
                    "后续请不要再调用 browser。"
                    "请改用 bash 工具执行可复现的命令行方案完成任务。"
                )
                update.decisions.append(GuardDecision(
                    kind="block",
                    scope="tool:browser",
                    reason_code="browser_consecutive_failure",
                    message=msg,
                    blocked_tools=["browser"],
                ))
                update.log_events.append(GuardLogEvent(
                    level="warning",
                    event="agent.tool_circuit_open",
                    fields={
                        "tool": "browser",
                        "fail_streak": state.browser_fail_streak,
                        "session_id": session_id or "",
                    },
                ))

    if tc.name == "bash":
        is_timeout = bool(result.error and "命令超时" in result.error)
        if result.success:
            state.bash_fail_streak = 0
            state.same_failed_bash_streak = 0
            state.last_failed_bash_signature = ""
        else:
            cmd_text = str(tc.arguments.get("command", ""))
            if skill_hooks:
                for _sh in skill_hooks:
                    _decision = _sh.on_tool_failure(tc, result)
                    if _decision is not None:
                        update.decisions.append(_decision)
                        _backfill_from_decisions(update)
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

            if is_timeout and state.same_failed_bash_streak < BASH_BLOCK_LIMIT:
                update.decisions.append(GuardDecision(
                    kind="hint",
                    scope="tool:bash",
                    reason_code="bash_timeout_hint",
                    message=(
                        "[系统提示] bash 命令超时，子进程已被终止。"
                        "如需重试，请检查参数并增大 timeout。"
                        "禁止不改参数直接重试。"
                    ),
                ))

            if (
                state.same_failed_bash_streak >= BASH_WARN_LIMIT
                and state.same_failed_bash_streak < BASH_BLOCK_LIMIT
                and "bash" not in state.blocked_tools
            ):
                update.decisions.append(GuardDecision(
                    kind="warn",
                    scope="tool:bash",
                    reason_code="bash_failure_warn",
                    message=(
                        "[系统提示] 同一 bash 命令模板已连续失败，请检查命令或换用其他方案。"
                        "如果继续失败将自动熔断。"
                    ),
                ))
                update.log_events.append(GuardLogEvent(
                    level="warning",
                    event="agent.tool_failure_warn",
                    fields={
                        "tool": "bash",
                        "same_failed_streak": state.same_failed_bash_streak,
                        "command_signature": state.last_failed_bash_signature[:200],
                        "session_id": session_id or "",
                    },
                ))

            if (
                state.same_failed_bash_streak >= BASH_BLOCK_LIMIT
                and "bash" not in state.blocked_tools
            ):
                state.blocked_tools.add("bash")
                msg = (
                    f"[系统降级] 同一 bash 命令模板已连续失败 {BASH_BLOCK_LIMIT} 次，"
                    "已自动熔断并切换策略。"
                    "后续请不要再调用 bash。"
                    "请改用结构化编辑工具（ppt_edit/docx_edit/xlsx_edit）"
                    "或文件工具（file_read/file_write/file_edit）继续。"
                )
                update.decisions.append(GuardDecision(
                    kind="block",
                    scope="tool:bash",
                    reason_code="bash_same_template_failure",
                    message=msg,
                    blocked_tools=["bash"],
                ))
                update.log_events.append(GuardLogEvent(
                    level="warning",
                    event="agent.tool_circuit_open",
                    fields={
                        "tool": "bash",
                        "fail_streak": state.bash_fail_streak,
                        "same_failed_streak": state.same_failed_bash_streak,
                        "command_signature": state.last_failed_bash_signature[:200],
                        "session_id": session_id or "",
                    },
                ))

    _backfill_from_decisions(update)
    return update


# ── DomainGuard (search_images) ──────────────────────────────

def _apply_search_images_domain_guard(
    state: ToolGuardState,
    tc: ToolCall,
    result: ToolResult,
) -> None:
    """更新 search_images 计数状态（不直接产出决策，轮末由 post_round 检查）。"""
    if tc.name != "browser":
        return
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


def _check_search_images_post_round(
    state: ToolGuardState,
    *,
    session_id: str | None,
) -> ToolGuardUpdate:
    """轮末检查 search_images 是否超额或重复。"""
    update = ToolGuardUpdate()

    if (
        not state.search_images_blocked_reason
        and state.search_images_limit is not None
        and state.search_images_count > state.search_images_limit
    ):
        state.search_images_blocked_reason = (
            "search_images 已超过计划配图数限制，禁止继续调用"
        )
        msg = (
            "[系统提示] 你已超过计划配图数的搜索额度。"
            f"计划配图数={state.planned_image_count}，"
            f"允许搜索上限={state.search_images_limit}，"
            f"当前已搜索={state.search_images_count}。"
            "禁止再调用 search_images。请立即进入生成或编辑步骤。"
        )
        update.decisions.append(GuardDecision(
            kind="block",
            scope="domain:search_images",
            reason_code="search_images_over_plan",
            message=msg,
        ))
        update.log_events.append(GuardLogEvent(
            level="warning",
            event="agent.search_images_over_plan",
            fields={
                "session_id": session_id or "",
                "planned_image_count": state.planned_image_count or 0,
                "search_images_limit": state.search_images_limit,
                "count": state.search_images_count,
            },
        ))

    if not state.search_images_blocked_reason and state.search_query_repeat_streak >= SEARCH_IMAGES_REPEAT_QUERY_LIMIT:
        state.search_images_blocked_reason = (
            "search_images 在重复相同搜索词且未取得新进展，禁止继续调用"
        )
        msg = (
            "[系统提示] 检测到你在重复使用相同搜索词搜图且未取得新进展。"
            f"重复搜索词：{state.last_search_query or '(空)'}。"
            "禁止继续调用 search_images。请立即进入生成或编辑步骤。"
        )
        update.decisions.append(GuardDecision(
            kind="block",
            scope="domain:search_images",
            reason_code="search_images_repeat_query",
            message=msg,
        ))
        update.log_events.append(GuardLogEvent(
            level="warning",
            event="agent.search_images_repeat_query_blocked",
            fields={
                "session_id": session_id or "",
                "query": state.last_search_query[:200],
                "repeat_streak": state.search_query_repeat_streak,
            },
        ))

    _backfill_from_decisions(update)
    return update


# ── LoopGuard ─────────────────────────────────────────────────

def _apply_loop_guard(
    state: ToolGuardState,
    tool_calls: list[ToolCall],
    *,
    round_idx: int,
    session_id: str | None,
) -> ToolGuardUpdate:
    """轮末检查重复路径（搜图不参与通用签名）。"""
    update = ToolGuardUpdate()

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

    combined_sig = f"{exact_sig}:{fuzzy_sig}"

    warn_threshold = state.loop_detect_window                              # 3
    block_threshold = state.loop_detect_window + 1                         # 4
    abort_threshold = state.loop_detect_window + 2                         # 5

    if (
        repeat_count >= warn_threshold
        and combined_sig != state.loop_warning_signature
    ):
        state.loop_warning_signature = combined_sig
        msg = (
            "[系统提示] 检测到你在重复执行相似操作且未取得进展。"
            f"重复工具：{repeated_tools_text}。"
            "请跳过前置检查，直接执行核心操作。"
        )
        update.decisions.append(GuardDecision(
            kind="warn",
            scope="loop:repeat_path",
            reason_code="loop_detected",
            message=msg,
        ))
        update.log_events.append(GuardLogEvent(
            level="warning",
            event="agent.loop_detected",
            fields={
                "session_id": session_id or "",
                "rounds": round_idx + 1,
                "repeated_tools": repeated_tools,
                "exact_repeat": exact_repeat,
                "fuzzy_repeat": fuzzy_repeat,
            },
        ))

    if (
        repeat_count >= block_threshold
        and combined_sig != state.loop_block_signature
    ):
        state.loop_block_signature = combined_sig
        newly_blocked = [name for name in repeated_tools if name not in state.blocked_tools]
        if newly_blocked:
            state.blocked_tools.update(newly_blocked)
            msg = (
                "[系统降级] 检测到重复路径仍在继续，已自动熔断这些工具："
                f"{'、'.join(newly_blocked)}。"
                "后续禁止继续调用它们。"
                "请改用其他工具或更直接的方案完成任务。"
            )
            update.decisions.append(GuardDecision(
                kind="block",
                scope="loop:repeat_path",
                reason_code="loop_circuit_open",
                message=msg,
                blocked_tools=newly_blocked,
            ))
            update.log_events.append(GuardLogEvent(
                level="warning",
                event="agent.loop_circuit_open",
                fields={
                    "session_id": session_id or "",
                    "rounds": round_idx + 1,
                    "blocked_tools": newly_blocked,
                    "exact_repeat": exact_repeat,
                    "fuzzy_repeat": fuzzy_repeat,
                },
            ))

    if repeat_count >= abort_threshold:
        update.decisions.append(GuardDecision(
            kind="abort",
            scope="loop:repeat_path",
            reason_code="loop_abort",
            message=(
                "检测到任务陷入重复循环，已自动中止当前路径。"
                "请让我改用更直接的方式继续完成任务。"
            ),
            stop_current_path=True,
        ))

    _backfill_from_decisions(update)
    return update


# ── 公开组合入口（保持对外签名不变） ─────────────────────────

def apply_tool_result_guards(
    state: ToolGuardState,
    tc: ToolCall,
    result: ToolResult,
    *,
    session_id: str | None,
    skill_hooks: list[SkillHooks] | None = None,
) -> ToolGuardUpdate:
    """每个工具执行完后调用。组合 FailureGuard + DomainGuard(搜图计数)。"""
    _apply_search_images_domain_guard(state, tc, result)

    failure_update = _apply_failure_guard(
        state, tc, result, session_id=session_id, skill_hooks=skill_hooks,
    )
    if failure_update.decisions:
        last = failure_update.decisions[-1]
        if last.scope.startswith("tool:bash:skill_hook"):
            return failure_update

    if result.success and is_progress_stage_tool_call(tc):
        state.search_query_repeat_streak = 0
        state.last_search_query = ""

    return failure_update


def apply_post_round_guards(
    state: ToolGuardState,
    tool_calls: list[ToolCall],
    *,
    round_idx: int,
    session_id: str | None,
) -> ToolGuardUpdate:
    """每轮工具全部执行完后调用。组合 DomainGuard(搜图轮末检查) + LoopGuard。"""
    search_update = _check_search_images_post_round(state, session_id=session_id)
    loop_update = _apply_loop_guard(
        state,
        tool_calls,
        round_idx=round_idx,
        session_id=session_id,
    )
    _merge_updates(search_update, loop_update)
    return search_update


def _merge_updates(target: ToolGuardUpdate, source: ToolGuardUpdate) -> None:
    """把 source 的所有内容合并到 target。"""
    target.conversation_messages.extend(source.conversation_messages)
    target.final_texts.extend(source.final_texts)
    target.log_events.extend(source.log_events)
    target.decisions.extend(source.decisions)
    if source.stop_for_probe_loop:
        target.stop_for_probe_loop = True
    if source.stop_for_repeat_loop:
        target.stop_for_repeat_loop = True


__all__ = [
    "BASH_BLOCK_LIMIT",
    "BASH_WARN_LIMIT",
    "BROWSER_BLOCK_LIMIT",
    "BROWSER_WARN_LIMIT",
    "GuardDecision",
    "GuardLogEvent",
    "MODEL_REPAIR_RETRY_LIMIT",
    "REPEAT_ABORT_ROUNDS",
    "REPEAT_BLOCK_ROUNDS",
    "REPEAT_WARN_ROUNDS",
    "SEARCH_IMAGES_REPEAT_QUERY_LIMIT",
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
