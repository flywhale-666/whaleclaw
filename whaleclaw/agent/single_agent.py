# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAssignmentType=false, reportRedeclaration=false, reportUnusedFunction=false, reportUnusedImport=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false, reportUnusedVariable=false, reportGeneralTypeIssues=false, reportInvalidTypeForm=false, reportUnusedVariable=false, reportGeneralTypeIssues=false
"""Agent main loop — message -> LLM -> tool -> reply (multi-turn).

The loop is provider-agnostic.  Tool invocation follows a single code
path regardless of whether the provider supports native ``tools`` API:

* **Native mode** — tool schemas are passed via ``tools=`` parameter;
  the provider returns structured ``ToolCall`` objects in the response.
* **Fallback mode** — tool descriptions are injected into the system
  prompt; the LLM outputs a JSON block which the loop parses.
"""

import asyncio
import base64
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from whaleclaw.agent.context import OnToolCall, OnToolResult
from whaleclaw.agent.helpers.evomap_helpers import (
    build_evomap_choice_prompt as _build_evomap_choice_prompt,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    build_evomap_first_system_message as _build_evomap_first_system_message,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    extract_evomap_choice_index as _extract_evomap_choice_index,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    extract_topic_terms as _extract_topic_terms,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    extra_memory_has_evomap_hint as _extra_memory_has_evomap_hint,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    infer_task_kind as _infer_task_kind,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    is_evomap_enabled as _is_evomap_enabled,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    is_evomap_status_question as _is_evomap_status_question,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    is_no_match_evomap_output as _is_no_match_evomap_output,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    is_tasky_message_for_evomap as _is_tasky_message_for_evomap,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    parse_evomap_fetch_candidates as _parse_evomap_fetch_candidates,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    pick_top_evomap_candidates as _pick_top_evomap_candidates,
)
from whaleclaw.agent.helpers.evomap_helpers import (
    recommended_evomap_signals as _recommended_evomap_signals,
)
from whaleclaw.agent.helpers.office_rules import (
    ABS_FILE_PATH_RE as _ABS_FILE_PATH_RE,
)
from whaleclaw.agent.helpers.office_rules import (
    NON_DELIVERY_EXTS as _NON_DELIVERY_EXTS,
)
from whaleclaw.agent.helpers.office_rules import (
    OFFICE_PATH_RE as _OFFICE_PATH_RE,
)
from whaleclaw.agent.helpers.office_rules import (
    append_office_system_hints as _append_office_system_hints,
)
from whaleclaw.agent.helpers.office_rules import (
    build_image_generation_system_message as _build_image_generation_system_message,
)
from whaleclaw.agent.helpers.office_rules import (
    build_office_path_block_message as _build_office_path_block_message,
)
from whaleclaw.agent.helpers.office_rules import (
    capture_latest_pptx as _capture_latest_pptx,
)
from whaleclaw.agent.helpers.office_rules import (
    extract_artifact_baseline as _extract_artifact_baseline,
)
from whaleclaw.agent.helpers.office_rules import (
    extract_delivery_artifact_paths as _extract_delivery_artifact_paths,
)
from whaleclaw.agent.helpers.office_rules import (
    extract_office_paths as _extract_office_paths,
)
from whaleclaw.agent.helpers.office_rules import (
    extract_round_delivery_section as _extract_round_delivery_section,
)
from whaleclaw.agent.helpers.office_rules import (
    fix_version_suffix as _fix_version_suffix,
)
from whaleclaw.agent.helpers.office_rules import (
    force_include_office_edit_tools as _force_include_office_edit_tools,
)
from whaleclaw.agent.helpers.office_rules import (
    get_default_office_edit_path as _get_default_office_edit_path,
)
from whaleclaw.agent.helpers.office_rules import (
    has_any_last_office_path as _has_any_last_office_path,
)
from whaleclaw.agent.helpers.office_rules import (
    is_followup_edit_message as _is_followup_edit_message,
)
from whaleclaw.agent.helpers.office_rules import (
    is_image_generation_request as _is_image_generation_request,
)
from whaleclaw.agent.helpers.office_rules import (
    is_office_edit_request as _is_office_edit_request,
)
from whaleclaw.agent.helpers.office_rules import (
    mentions_specific_dark_bar_target as _mentions_specific_dark_bar_target,
)
from whaleclaw.agent.helpers.office_rules import (
    remember_office_path as _remember_office_path,
)
from whaleclaw.agent.helpers.office_rules import (
    snapshot_round_artifacts as _snapshot_round_artifacts,
)
from whaleclaw.agent.helpers.office_rules import with_round_version_suffix
from whaleclaw.agent.helpers.skill_helpers import (
    build_skill_lock_system_message as _build_skill_lock_system_message,
)
from whaleclaw.agent.helpers.skill_helpers import (
    build_skill_param_guard_reply as _build_skill_param_guard_reply,
)
from whaleclaw.agent.helpers.skill_helpers import (
    detect_assistant_name_update as _detect_assistant_name_update,
)
from whaleclaw.agent.helpers.skill_helpers import guarded_skills as _guarded_skills
from whaleclaw.agent.helpers.skill_helpers import (
    is_task_done_confirmation as _is_task_done_confirmation,
)
from whaleclaw.agent.helpers.skill_helpers import (
    looks_like_skill_activation_message as _looks_like_skill_activation_message,
)
from whaleclaw.agent.helpers.skill_helpers import normalize_for_match as _normalize_for_match
from whaleclaw.agent.helpers.skill_helpers import normalize_skill_ids as _normalize_skill_ids
from whaleclaw.agent.helpers.skill_helpers import parse_use_command as _parse_use_command
from whaleclaw.agent.helpers.skill_helpers import preview_text as _preview_text
from whaleclaw.agent.helpers.skill_helpers import (
    select_native_tool_names as _select_native_tool_names,
)
from whaleclaw.agent.helpers.skill_helpers import skill_announcement as _skill_announcement
from whaleclaw.agent.helpers.skill_helpers import (
    skill_explicitly_mentioned as _skill_explicitly_mentioned,
)
from whaleclaw.agent.helpers.skill_helpers import (
    skill_trigger_mentioned as _skill_trigger_mentioned,
)
from whaleclaw.agent.helpers.skill_helpers import (
    is_api_key_only_message as _is_api_key_only_message,
)
from whaleclaw.agent.helpers.skill_helpers import (
    persist_param_api_key as _persist_param_api_key,
)
from whaleclaw.agent.helpers.skill_helpers import update_guard_state as _update_guard_state
from whaleclaw.agent.helpers.skill_helpers import (
    build_skill_queue_advance_message as _build_skill_queue_advance_message,
)
from whaleclaw.agent.helpers.skill_helpers import (
    build_skill_queue_plan_messages as _build_skill_queue_plan_messages,
)
from whaleclaw.agent.helpers.skill_helpers import (
    build_skill_queue_status_message as _build_skill_queue_status_message,
)
from whaleclaw.agent.helpers.skill_helpers import (
    is_queue_advance_confirmation as _is_queue_advance_confirmation,
)
from whaleclaw.agent.helpers.skill_helpers import (
    parse_skill_queue_plan as _parse_skill_queue_plan,
)
from whaleclaw.agent.helpers.skill_helpers import (
    skill_queue_has_next as _skill_queue_has_next,
)
from whaleclaw.agent.helpers.image_refs import (
    append_image_reference_history as _append_image_reference_history,
)
from whaleclaw.agent.helpers.image_refs import (
    extract_input_image_paths_from_text as _extract_input_image_paths_from_text,
)
from whaleclaw.agent.helpers.image_refs import (
    fix_image_paths as _fix_image_paths,
)
from whaleclaw.agent.helpers.image_refs import (
    is_image_reference_lookup_message as _is_image_reference_lookup_message,
)
from whaleclaw.agent.helpers.image_refs import (
    load_images_from_paths as _load_images_from_paths,
)
from whaleclaw.agent.helpers.image_refs import (
    is_numbered_image_reference_edit_message as _is_numbered_image_reference_edit_message,
)
from whaleclaw.agent.helpers.image_refs import (
    message_may_need_prior_images as _message_may_need_prior_images,
)
from whaleclaw.agent.helpers.image_refs import (
    message_requests_image_edit as _message_requests_image_edit,
)
from whaleclaw.agent.helpers.image_refs import (
    message_requests_image_regenerate as _message_requests_image_regenerate,
)
from whaleclaw.agent.helpers.image_refs import (
    recover_last_input_images as _recover_last_input_images,
)
from whaleclaw.agent.helpers.image_refs import (
    recover_latest_generated_image as _recover_latest_generated_image,
)
from whaleclaw.agent.helpers.image_refs import (
    resolve_numbered_input_image_paths as _resolve_numbered_input_image_paths,
)
from whaleclaw.agent.helpers.image_refs import (
    resolve_relative_image_reference_path as _resolve_relative_image_reference_path,
)
from whaleclaw.agent.helpers.image_refs import (
    recover_recent_session_image_paths as _recover_recent_session_image_paths,
)
from whaleclaw.agent.helpers.image_refs import (
    skill_requires_images as _skill_requires_images,
)
from whaleclaw.agent.helpers.image_refs import (
    strip_inline_image_markdown as _strip_inline_image_markdown,
)
from whaleclaw.skills.hooks import SkillHooks, get_all_hooks, get_skill_hooks
from whaleclaw.agent.helpers.tool_execution import (
    can_auto_create_parent_for_failure,
    create_default_registry,
)
from whaleclaw.agent.helpers.tool_execution import (
    execute_tool as _execute_tool,
)
from whaleclaw.agent.helpers.tool_execution import (
    format_tool_output as _format_tool_output,
)
from whaleclaw.agent.helpers.tool_execution import (
    is_transient_cli_usage_error as _is_transient_cli_usage_error,
)
from whaleclaw.agent.helpers.tool_execution import (
    parse_fallback_tool_calls as _parse_fallback_tool_calls,
)
from whaleclaw.agent.helpers.tool_execution import (
    persist_message as _persist_message,
)
from whaleclaw.agent.helpers.tool_execution import repair_tool_call as _repair_tool_call
from whaleclaw.agent.helpers.tool_execution import strip_tool_json as _strip_tool_json
from whaleclaw.agent.helpers.tool_execution import (
    validate_tool_call_args as _validate_tool_call_args,
)
from whaleclaw.agent.helpers.tool_guards import (
    MODEL_REPAIR_RETRY_LIMIT,
    ToolGuardState,
    apply_post_round_guards,
    apply_tool_result_guards,
    blocked_tool_reasons,
    update_planned_image_count,
)
from whaleclaw.agent.prompt import PromptAssembler
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import AgentResponse, ImageContent, Message, ToolCall
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.compressor import ContextCompressor
from whaleclaw.sessions.context_window import RECENT_PROTECTED, ContextWindow, TrimResult
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore, SummaryRow
from whaleclaw.skills.parser import Skill
from whaleclaw.tools.base import ToolResult
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.types import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    StreamCallback,
)
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.memory.manager import MemoryManager
    from whaleclaw.sessions.group_compressor import SessionGroupCompressor

log = get_logger(__name__)

OnRoundResult = Callable[[int, str], Awaitable[None]]


class AgentDoneInfo:
    """Metadata emitted when the agent loop completes."""

    __slots__ = ("model", "input_tokens", "output_tokens", "llm_rounds")

    def __init__(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        llm_rounds: int,
    ) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.llm_rounds = llm_rounds


OnAgentDone = Callable[[AgentDoneInfo], Awaitable[None]]

_assembler = PromptAssembler()
_context_window = ContextWindow()
_compressor = ContextCompressor()
_memory_organizer_tasks: dict[str, asyncio.Task[None]] = {}

_MAX_OUTPUT_TOKENS = 200_000
_EVOMAP_MAX_TOKENS = 1000
_EXTRA_MEMORY_COMPRESS_TIMEOUT_SECONDS = 8
_DEFAULT_ASSISTANT_NAME = "WhaleClaw"

_NON_SKILL_STEP_RE = re.compile(
    r"(?:做|做个|做一个|创建|新建|生成|制作|写)"
    r"(?:一个|个|一份|份)?"
    r"(?:.{0,8})"
    r"(?:PPT|ppt|pptx|PPTX|幻灯片|Word|word|docx|DOCX|文档|Excel|excel|xlsx|XLSX|表格)"
    r"|(?:插入|放入|放到|加入|加到|添加到|嵌入|写入)"
    r"(?:.{0,6})"
    r"(?:PPT|ppt|pptx|Word|word|docx|文档|表格)",
    re.IGNORECASE,
)

_VERSION_SUFFIX_RE = re.compile(r"_V\d+$", re.IGNORECASE)
_COORDINATOR_ASK_RE = re.compile(
    r"(?:你要(?:我|什么)|需要你(?:提供|告诉|回复|回答|确认|选择)|"
    r"请(?:告诉|选择|提供|告知)我|"
    r"(?:按|用)(?:下面|以下)(?:模板|格式)(?:回|填|答))",
)
_ASSISTANT_NAME_RESET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"恢复默认名字"),
    re.compile(r"改回\s*whaleclaw", re.IGNORECASE),
    re.compile(r"还是叫\s*whaleclaw", re.IGNORECASE),
)
_ASSISTANT_NAME_SET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:以后|从现在起|今后|之后|开始)\s*(?:你|机器人|助手)?\s*(?:就)?(?:叫|改叫|改名叫)\s*([^\s，。！？!?、,]{1,24})"
    ),
    re.compile(r"(?:把你|你)\s*(?:改名|改名为|名字改成|名字改为)\s*([^\s，。！？!?、,]{1,24})"),
    re.compile(r"^\s*(?:你|助手|机器人)\s*(?:就)?叫\s*([^\s，。！？!?、,]{1,24})\s*$"),
)
_USE_CMD_RE = re.compile(r"^\s*/use\s+([^\s]+)\s*(.*)$", re.IGNORECASE | re.DOTALL)
_USE_CLEAR_IDS = {"clear", "none", "off", "default", "reset"}
_TASK_DONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*任务完成\s*$"),
    re.compile(r"^\s*完成任务\s*$"),
    re.compile(r"^\s*任务结束\s*$"),
    re.compile(r"^\s*结束任务\s*$"),
    re.compile(r"^\s*完成了?\s*$"),
    re.compile(r"^\s*结束了?\s*$"),
    re.compile(r"^\s*可以了?\s*$"),
    re.compile(r"^\s*ok\s*$", re.IGNORECASE),
)
_TASK_DONE_NEAR_MISS_RE = re.compile(
    r"(?:本轮|这轮|这次|这一轮|本次).{0,4}(?:结束|完成|搞定|好了)"
    r"|(?:结束|完成|搞定).{0,4}(?:啦|了|嘞|咯)",
)


def _is_task_done_intent_near_miss(text: str) -> bool:
    """用户话语接近「任务完成」但不精确匹配 _TASK_DONE_PATTERNS 时返回 True。"""
    stripped = text.strip()
    if _is_task_done_confirmation(stripped, task_done_patterns=_TASK_DONE_PATTERNS):
        return False
    return bool(_TASK_DONE_NEAR_MISS_RE.search(stripped))


_SKILL_LOCK_STATUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:现在|当前).{0,8}(?:被)?锁定在.{0,8}(?:哪个|什么).{0,6}(?:技能|skill)"),
    re.compile(r"(?:现在|当前).{0,6}(?:技能|skill).{0,6}(?:锁定|状态)"),
    re.compile(r"(?:锁定在.{0,8}(?:哪个|什么).{0,6}(?:技能|skill))"),
)
_SKILL_SWITCH_CONSENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:同意|可以|确认|允许).{0,8}(?:切换|换).{0,8}(?:技能|skill)?", re.IGNORECASE),
    re.compile(r"(?:切换|换).{0,8}(?:技能|skill).{0,8}(?:吧|可以|行|好的|ok)", re.IGNORECASE),
    re.compile(r"(?:换成|改用|切到).{0,24}(?:技能|skill)", re.IGNORECASE),
)
_SKILL_SWITCH_KEEP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:继续|仍然|还是).{0,8}(?:沿用|使用|走).{0,8}(?:原技能|原来的技能|当前技能)"),
    re.compile(r"(?:不|别).{0,4}(?:切换|换).{0,8}(?:技能|skill)?", re.IGNORECASE),
    re.compile(r"(?:保持|沿用).{0,8}(?:原技能|当前技能)", re.IGNORECASE),
)
_SKILL_ACTIVATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:使用|调用|启动|启用|走|用).{0,24}(?:技能|skill)", re.IGNORECASE),
    re.compile(r"(?:技能|skill).{0,16}(?:文生图|图生图|处理|执行|联调)", re.IGNORECASE),
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    MULTI_AGENT_CANCEL_PATTERNS as _MULTI_AGENT_CANCEL_PATTERNS,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    MULTI_AGENT_CONFIRM_PATTERNS as _MULTI_AGENT_CONFIRM_PATTERNS,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    MULTI_AGENT_DISCUSS_DONE_PATTERNS as _MULTI_AGENT_DISCUSS_DONE_PATTERNS,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    MULTI_AGENT_ROUNDS_PATTERNS as _MULTI_AGENT_ROUNDS_PATTERNS,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    MULTI_AGENT_SCENARIO_LABELS,
)

_MULTI_AGENT_SCENARIO_LABELS = MULTI_AGENT_SCENARIO_LABELS

# Public aliases for cross-module reuse.
ABS_FILE_PATH_RE = _ABS_FILE_PATH_RE
NON_DELIVERY_EXTS = _NON_DELIVERY_EXTS
OFFICE_PATH_RE = _OFFICE_PATH_RE
COORDINATOR_ASK_RE = _COORDINATOR_ASK_RE
_with_round_version_suffix = with_round_version_suffix
_can_auto_create_parent_for_failure = can_auto_create_parent_for_failure

_TOOL_HINTS: dict[str, str] = {
    "browser": "搜索相关资料",
    "web_fetch": "抓取网页正文",
    "desktop_capture": "点亮并截图桌面",
    "bash": "执行命令",
    "process": "查看或结束进程",
    "file_write": "生成文件",
    "file_read": "读取文件",
    "file_edit": "编辑文件",
    "patch_apply": "应用补丁",
    "ppt_edit": "修改现有PPT",
    "docx_edit": "修改现有Word",
    "xlsx_edit": "修改现有Excel",
    "memory_search": "检索长期记忆",
    "memory_add": "写入长期记忆",
    "memory_list": "查看长期记忆",
    "skill": "查找技能",
}

def _build_memory_system_message(recalled: str) -> Message:
    """Wrap recalled memory as durable preference/fact context."""
    return Message(
        role="system",
        content=(
            "以下是从长期记忆召回的历史信息，包含用户长期偏好、稳定约束与历史事实。\n"
            "执行规则：\n"
            "1) 若内容属于长期偏好/写作与产出规则，且不与本轮用户要求冲突，请默认执行；\n"
            "2) 若内容属于历史事实且你不确定当前是否仍然有效，可先向用户确认。\n"
            f"{recalled}"
        ),
    )


def _build_global_style_system_message(style_directive: str) -> Message:
    return Message(
        role="system",
        content=(
            "以下是用户长期稳定的全局回复风格偏好，请默认遵守：\n"
            f"{style_directive.strip()}\n"
            "若用户在本轮消息中明确提出不同风格/长度要求，以本轮用户要求为准。"
        ),
    )


def _build_external_memory_system_message(extra_memory: str) -> Message:
    return Message(
        role="system",
        content=(
            "以下是来自协作网络的外部经验候选，仅作为补充参考：\n"
            f"{extra_memory.strip()}\n"
            "若与用户本轮明确要求冲突，以用户本轮要求为准；"
            "若与本地长期记忆冲突，以本地长期记忆为准。"
        ),
    )


def _est_tokens(text: str) -> int:
    return max(0, len(text) // 3)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    char_cap = max_tokens * 3
    if len(text) <= char_cap:
        return text
    return text[:char_cap]


async def _compress_external_memory_with_llm(
    *,
    router: ModelRouter,
    model_id: str,
    text: str,
    max_tokens: int,
) -> str:
    sys_prompt = (
        "你是外部经验压缩器。"
        "请将输入经验压缩到给定 token 上限内，保留可执行做法与约束。"
        "禁止新增事实。输出纯文本。"
    )
    user_prompt = f"目标上限约 {max_tokens} tokens。\n输入如下：\n{text}\n\n请输出压缩结果。"
    try:
        resp = await router.chat(
            model_id,
            [
                Message(role="system", content=sys_prompt),
                Message(role="user", content=user_prompt),
            ],
        )
    except Exception:
        return ""
    out = resp.content.strip()
    if not out:
        return ""
    if _est_tokens(out) <= max_tokens:
        return out
    return _truncate_to_tokens(out, max_tokens)


async def _load_session_summaries_typed(
    store: SessionStore,
    session_id: str,
) -> list[SummaryRow]:
    return await store.get_summaries(session_id)


def _merge_recall_blocks(profile: str, raw: str) -> str:
    blocks = [x.strip() for x in (profile, raw) if x.strip()]
    return "\n".join(blocks)


def _is_creation_task_message(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return False
    keys = (
        "ppt",
        "幻灯片",
        "演示文稿",
        "文档",
        "报告",
        "方案",
        "写",
        "生成",
        "制作",
        "整理",
        "设计",
        "润色",
        "总结",
        "改写",
        "脚本",
        "代码",
        "html",
        "页面",
        "海报",
        "计划",
        "create",
        "generate",
        "draft",
        "design",
        "write",
        "build",
        "compose",
    )
    return any(k in low for k in keys)


async def _llm_judge_task_phase(
    router: "ModelRouter",
    model_id: str,
    *,
    session: Session | None,
    message: str,
) -> str:
    """Use the main model to classify task phase: NEW_TASK or EDITING."""
    system = Message(
        role="system",
        content=(
            "你是任务阶段分类器，只输出一个标签：NEW_TASK 或 EDITING。\n"
            "NEW_TASK=开始一个全新主要任务/新主题/新产物。\n"
            "EDITING=在已有任务上修改/补充/继续/讨论细节。\n"
            "只输出标签，不要解释。"
        ),
    )
    context: list[Message] = [system]
    if session is not None and session.messages:
        recent: list[Message] = []
        for msg in session.messages[-6:]:
            if msg.role not in {"user", "assistant"}:
                continue
            recent.append(
                Message(role=msg.role, content=_preview_text(msg.content or "", limit=400))
            )
        context.extend(recent)
    context.append(
        Message(role="user", content=f"当前用户消息：{_preview_text(message, limit=600)}")
    )
    try:
        resp = await router.chat(model_id, context, tools=None, on_stream=None)
    except Exception:
        return "EDITING"
    raw = (resp.content or "").strip().upper()
    if "NEW_TASK" in raw:
        return "NEW_TASK"
    if "EDITING" in raw:
        return "EDITING"
    return "EDITING"


def _schedule_memory_organizer_task(
    session_id: str,
    *,
    memory_manager: "MemoryManager",
    router: ModelRouter,
    model_id: str,
    organizer_min_new_entries: int,
    organizer_interval_seconds: int,
    organizer_max_raw_window: int,
    keep_profile_versions: int,
    max_raw_entries: int,
) -> None:
    running = _memory_organizer_tasks.get(session_id)
    if running is not None and not running.done():
        return

    async def _run() -> None:
        try:
            organized = await memory_manager.organize_if_needed(
                router=router,
                model_id=model_id,
                organizer_min_new_entries=organizer_min_new_entries,
                organizer_interval_seconds=organizer_interval_seconds,
                organizer_max_raw_window=organizer_max_raw_window,
                keep_profile_versions=keep_profile_versions,
                max_raw_entries=max_raw_entries,
            )
            if organized:
                log.info("agent.memory_organized", session_id=session_id)
        except Exception as exc:
            log.debug("agent.memory_organize_failed", error=str(exc), session_id=session_id)

    task = asyncio.create_task(_run(), name=f"memory-organizer:{session_id}")
    _memory_organizer_tasks[session_id] = task

    def _cleanup(_task: asyncio.Task[None]) -> None:
        current = _memory_organizer_tasks.get(session_id)
        if current is _task:
            _memory_organizer_tasks.pop(session_id, None)

    task.add_done_callback(_cleanup)


def _make_plan_hint(tool_names: list[str], user_msg: str) -> str:
    """Generate a brief plan message when LLM jumps straight to tool calls."""
    steps: list[str] = []
    seen: set[str] = set()
    for name in tool_names:
        if name in seen:
            continue
        seen.add(name)
        steps.append(_TOOL_HINTS.get(name, f"调用 {name}"))
    plan = "、".join(steps)
    return f"好的，我来处理。正在{plan}…\n\n"


_CN_DIGIT_MAP: dict[str, str] = {
    "零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
    "十": "10", "两": "2",
}
_CN_COMPOUND_TEN_RE = re.compile(r"十([一二三四五六七八九])")
_CN_TENS_RE = re.compile(r"([二三四五六七八九])十")


def _normalize_cn_time(text: str) -> str:
    """把中文数字时间表达归一化为阿拉伯数字，方便正则匹配。"""
    text = _CN_COMPOUND_TEN_RE.sub(lambda m: "1" + _CN_DIGIT_MAP[m.group(1)], text)
    text = _CN_TENS_RE.sub(lambda m: _CN_DIGIT_MAP[m.group(1)] + "0", text)
    text = text.replace("十", "10")
    for cn, ar in _CN_DIGIT_MAP.items():
        text = text.replace(cn, ar)
    return text


_DEFERRED_TASK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\d+\s*(?:分钟|小时|秒|min|hour|h)\s*(?:后|之后|以后)"),
    re.compile(r"(?:每天|每日|每周|每月|每隔)\s*\d"),
    re.compile(
        r"(?:(?:今天|明天|后天)?\s*(?:今晚|明早|早上|上午|中午|下午|晚上|凌晨|傍晚)\s*)?"
        r"\d{1,2}\s*[点时:：]\s*(?:半|\d{0,2})\s*(?:分钟?)?"
        r"(?:\s*的时候|\s*时候)?"
        r"\s*(?:去|做|执行|运行|发布|发送|提醒|推送|搜索|搜|找|画|写|打开|关闭|启动|停止|用|使用|叫)"
    ),
    re.compile(
        r"(?:每天|每日|每周|每月)\s*(?:早上|上午|中午|下午|晚上|凌晨|傍晚)?"
        r"\s*\d{1,2}\s*[点时:：]"
    ),
)
_DEFERRED_SCHEDULE_KW = re.compile(r"(?:定时|定期)\s*(?:执行|运行|发布|发送|提醒|推送)")
_QUESTION_TAIL_RE = re.compile(r"[吗呢？\?]\s*$")
_QUESTION_PREFIX_RE = re.compile(r"(?:有没有|是否|是不是|有做|查看|查询|查下)")


def _is_deferred_task_intent(message: str) -> bool:
    """检测用户消息是否是定时/周期任务意图，而非立即执行。"""
    normalized = _normalize_cn_time(message)
    if any(p.search(normalized) for p in _DEFERRED_TASK_PATTERNS):
        return True
    if not _DEFERRED_SCHEDULE_KW.search(normalized):
        return False
    if _QUESTION_TAIL_RE.search(normalized) or _QUESTION_PREFIX_RE.search(normalized):
        return False
    return True


def _is_skill_lock_status_question(message: str) -> bool:
    """Detect direct questions asking which skill the session is locked to."""
    text = message.strip()
    if not text:
        return False
    return any(pattern.search(text) for pattern in _SKILL_LOCK_STATUS_PATTERNS)


def _extract_locked_skill_ids_from_metadata(metadata: object) -> list[str]:
    """Normalize locked skill ids from arbitrary session metadata."""
    if not isinstance(metadata, dict):
        return []
    metadata_dict = cast(dict[object, object], metadata)
    raw_locked = metadata_dict.get("locked_skill_ids")
    if isinstance(raw_locked, list):
        locked_items = cast(list[object], raw_locked)
        return [
            item.strip().lower()
            for item in locked_items
            if isinstance(item, str) and item.strip()
        ]
    raw_forced = metadata_dict.get("forced_skill_id")
    if isinstance(raw_forced, str) and raw_forced.strip():
        return [raw_forced.strip().lower()]
    return []


def _build_skill_lock_status_reply(session: Session | None) -> str:
    """Render the current skill-lock status from persisted metadata."""
    locked_skill_ids = _extract_locked_skill_ids_from_metadata(
        session.metadata if session is not None else None
    )
    if not locked_skill_ids:
        return "当前没有技能锁定。"
    current_skills = "、".join(locked_skill_ids)
    extra_parts: list[str] = []
    locked_skills = _assembler.route_skills("", forced_skill_ids=locked_skill_ids)
    for skill in locked_skills:
        hooks = get_skill_hooks(skill)
        if hooks is not None:
            extra = hooks.build_lock_status_extra(session)
            if extra:
                extra_parts.append(extra)
    base = f"现在锁定在技能：{current_skills}。"
    if extra_parts:
        base += "\n" + "\n".join(extra_parts)
    base += "\n要解除锁定，回复“任务完成”即可。"
    return base


def _raw_skill_trigger_mentioned(skill: Skill, text: str) -> bool:
    lower = text.lower()
    normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lower)
    for raw in skill.triggers:
        trigger = raw.strip().lower()
        if not trigger:
            continue
        if trigger in lower:
            return True
        compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", trigger)
        if compact and compact in normalized:
            return True
    return False


def _lockable_skill_ids(skills: list[Skill]) -> list[str]:
    return _normalize_skill_ids([
        skill for skill in skills
        if skill.lock_session and skill.is_user_installed
    ])


def _snapshot_image_history_at_activation(session: Session) -> None:
    """记录技能激活时 image_reference_history 的长度，用于后续过滤旧图。"""
    raw = session.metadata.get("image_reference_history", [])
    length = len(raw) if isinstance(raw, list) else 0
    session.metadata["__skill_activation_image_snapshot__"] = length


def _contains_lock_confirm_tip(text: str) -> bool:
    """Check if the reply already contains the lock-confirm tip."""
    return "任务完成" in text and "解除技能锁定" in text


def _cleanup_final_reply_text(text: str) -> str:
    """Trim excessive blank lines and whitespace from the final reply."""
    text = text.strip()
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text


def _canonicalize_lock_confirm_tip(text: str, tip: str) -> str:
    """Ensure the lock-confirm tip appears at most once, at the end."""
    if not tip or tip not in text:
        return text
    cleaned = text.replace(tip, "").strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return f"{cleaned}\n{tip}" if cleaned else tip


from whaleclaw.agent.helpers.multi_agent_helpers import (
    attach_rounds_marker as _attach_rounds_marker,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    build_multi_agent_requirement_baseline,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    compact_role_output,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    extract_multi_agent_rounds as _extract_multi_agent_rounds,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    extract_requested_deliverables,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    extract_rounds_marker as _extract_rounds_marker,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    format_multi_agent_preflight_text as _format_multi_agent_preflight_text,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    is_multi_agent_cancel as _is_multi_agent_cancel,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    is_multi_agent_confirm,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    is_multi_agent_discuss_done,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    looks_like_bad_coordinator_output,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    looks_like_role_stall_output,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    multi_agent_cfg as _multi_agent_cfg,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    multi_agent_system_prompt,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    need_image_output,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    persist_session_metadata as _persist_session_metadata,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    resolve_multi_agent_cfg as _resolve_multi_agent_cfg,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    run_multi_agent_controller_discussion as _run_multi_agent_controller_discussion,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    run_multi_agent_executor as _run_multi_agent_executor,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    scenario_delivery_focus as _scenario_delivery_focus,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    scenario_discuss_focus as _scenario_discuss_focus,
)
from whaleclaw.agent.helpers.multi_agent_helpers import (
    sync_multi_agent_compression_boundary as _sync_multi_agent_compression_boundary,
)

_multi_agent_system_prompt = multi_agent_system_prompt
_compact_role_output = compact_role_output
_looks_like_bad_coordinator_output = looks_like_bad_coordinator_output
_looks_like_role_stall_output = looks_like_role_stall_output
_need_image_output = need_image_output
_extract_requested_deliverables = extract_requested_deliverables
_build_multi_agent_requirement_baseline = build_multi_agent_requirement_baseline
_is_multi_agent_confirm = is_multi_agent_confirm
_is_multi_agent_discuss_done = is_multi_agent_discuss_done

scenario_discuss_focus = _scenario_discuss_focus
truncate_to_tokens = _truncate_to_tokens
resolve_multi_agent_cfg = _resolve_multi_agent_cfg
is_multi_agent_confirm = _is_multi_agent_confirm  # pyright: ignore[reportRedeclaration]
extract_multi_agent_rounds = _extract_multi_agent_rounds
is_multi_agent_discuss_done = _is_multi_agent_discuss_done  # pyright: ignore[reportRedeclaration]
select_native_tool_names = _select_native_tool_names
extract_round_delivery_section = _extract_round_delivery_section
extract_delivery_artifact_paths = _extract_delivery_artifact_paths
fix_version_suffix = _fix_version_suffix
snapshot_round_artifacts = _snapshot_round_artifacts
extract_artifact_baseline = _extract_artifact_baseline

async def run_agent(
    message: str,
    session_id: str,
    config: WhaleclawConfig,
    on_stream: StreamCallback | None = None,
    *,
    session: Session | None = None,
    router: ModelRouter | None = None,
    registry: ToolRegistry | None = None,
    on_tool_call: OnToolCall | None = None,
    on_tool_result: OnToolResult | None = None,
    on_round_result: OnRoundResult | None = None,
    on_done: OnAgentDone | None = None,
    images: list[ImageContent] | None = None,
    session_manager: SessionManager | None = None,
    session_store: SessionStore | None = None,
    memory_manager: "MemoryManager | None" = None,
    extra_memory: str = "",
    trigger_event_id: str = "",
    trigger_text_preview: str = "",
    group_compressor: "SessionGroupCompressor | None" = None,
    multi_agent_internal: bool = False,
) -> str:
    """Run the Agent loop with tool support and multi-turn context.

    The loop is provider-agnostic:
    1. Check if provider supports native tools API
    2. If yes  -> pass schemas via ``tools=``; parse structured tool_calls
    3. If no   -> inject tool descriptions into system prompt; parse JSON text
    4. Execute tools, append results, loop (until no tool calls or token budget exhausted)
    5. Return final text reply
    """
    agent_cfg = config.agent
    models_cfg = config.models
    summarizer_cfg = agent_cfg.summarizer

    model_id: str = session.model if session else agent_cfg.model
    if router is None:
        router = ModelRouter(models_cfg)
    if registry is None:
        registry = create_default_registry()

    if not multi_agent_internal:
        ma_cfg = _resolve_multi_agent_cfg(config, session)
        await _sync_multi_agent_compression_boundary(
            session,
            session_manager,
            group_compressor,
            ma_enabled=bool(ma_cfg.get("enabled", False)),
        )
        if bool(ma_cfg.get("enabled", False)):
            if session is not None:
                state = str(session.metadata.get("multi_agent_state", "")).strip().lower()
                waiting = state == "confirm" or (state == "" and bool(
                    session.metadata.get("multi_agent_waiting_confirm", False)
                ))
                intro_done = bool(session.metadata.get("multi_agent_intro_done", False))
                pending_topic = str(session.metadata.get("multi_agent_pending_topic", "")).strip()

                if waiting and _is_multi_agent_cancel(message):
                    session.metadata.pop("multi_agent_state", None)
                    session.metadata.pop("multi_agent_waiting_confirm", None)
                    session.metadata.pop("multi_agent_intro_done", None)
                    session.metadata.pop("multi_agent_pending_topic", None)
                    session.metadata.pop("multi_agent_pending_rounds", None)
                    await _persist_session_metadata(session, session_manager)
                    return "已取消本次多Agent执行。你可以继续普通对话，或发新需求后再确认启动。"

                rounds_override = _extract_multi_agent_rounds(message)
                if rounds_override is not None and waiting:
                    session.metadata["multi_agent_pending_rounds"] = rounds_override
                    topic = _attach_rounds_marker(pending_topic or message, rounds_override)
                    session.metadata["multi_agent_pending_topic"] = topic
                    await _persist_session_metadata(session, session_manager)
                    ma_cfg["max_rounds"] = rounds_override
                    return await _run_multi_agent_controller_discussion(
                        user_message=message,
                        pending_topic=topic,
                        cfg=ma_cfg,
                        session_id=session_id,
                        config=config,
                        router=router,
                        registry=registry,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        include_intro=not intro_done,
                    )

                if state == "discuss":
                    if _is_multi_agent_cancel(message):
                        session.metadata.pop("multi_agent_state", None)
                        session.metadata.pop("multi_agent_intro_done", None)
                        session.metadata.pop("multi_agent_pending_topic", None)
                        session.metadata.pop("multi_agent_pending_rounds", None)
                        await _persist_session_metadata(session, session_manager)
                        return "已取消本次多Agent讨论。你可以继续普通对话。"

                    topic = pending_topic

                    if _is_multi_agent_discuss_done(message):
                        rounds_raw = session.metadata.get("multi_agent_pending_rounds")
                        if isinstance(rounds_raw, int):
                            ma_cfg["max_rounds"] = max(1, min(rounds_raw, 10))
                        cleaned_topic, marker_rounds = _extract_rounds_marker(topic or "")
                        if not isinstance(rounds_raw, int) and marker_rounds is not None:
                            ma_cfg["max_rounds"] = marker_rounds
                        session.metadata.pop("multi_agent_state", None)
                        session.metadata.pop("multi_agent_waiting_confirm", None)
                        session.metadata.pop("multi_agent_intro_done", None)
                        session.metadata.pop("multi_agent_pending_topic", None)
                        session.metadata.pop("multi_agent_pending_rounds", None)
                        await _persist_session_metadata(session, session_manager)
                        return await _run_multi_agent_executor(
                            message=cleaned_topic or "（请补充你的任务目标）",
                            session_id=session_id,
                            config=config,
                            on_stream=on_stream,
                            router=router,
                            registry=registry,
                            images=images,
                            extra_memory=extra_memory,
                            trigger_event_id=trigger_event_id,
                            trigger_text_preview=trigger_text_preview,
                            ma_cfg=ma_cfg,
                            on_round_result=on_round_result,
                        )

                    if message.strip():
                        if topic:
                            topic = f"{topic}\n补充要求: {message.strip()}".strip()
                        else:
                            topic = message.strip()
                        session.metadata["multi_agent_pending_topic"] = topic
                    if rounds_override is not None:
                        session.metadata["multi_agent_pending_rounds"] = rounds_override
                        session.metadata["multi_agent_pending_topic"] = _attach_rounds_marker(
                            topic or message,
                            rounds_override,
                        )
                        topic = str(session.metadata["multi_agent_pending_topic"])
                        ma_cfg["max_rounds"] = rounds_override

                    include_intro = not bool(session.metadata.get("multi_agent_intro_done", False))
                    if include_intro:
                        session.metadata["multi_agent_intro_done"] = True
                    await _persist_session_metadata(session, session_manager)
                    return await _run_multi_agent_controller_discussion(
                        user_message=message,
                        pending_topic=topic or message,
                        cfg=ma_cfg,
                        session_id=session_id,
                        config=config,
                        router=router,
                        registry=registry,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        include_intro=include_intro,
                    )

                if waiting and _is_multi_agent_confirm(message):
                    topic = pending_topic or "（请补充你的任务目标）"
                    rounds_raw = session.metadata.get("multi_agent_pending_rounds")
                    if isinstance(rounds_raw, int):
                        ma_cfg["max_rounds"] = max(1, min(rounds_raw, 10))
                    cleaned_topic, marker_rounds = _extract_rounds_marker(topic or "")
                    if not isinstance(rounds_raw, int) and marker_rounds is not None:
                        ma_cfg["max_rounds"] = marker_rounds
                    session.metadata.pop("multi_agent_state", None)
                    session.metadata.pop("multi_agent_waiting_confirm", None)
                    session.metadata.pop("multi_agent_intro_done", None)
                    session.metadata.pop("multi_agent_pending_topic", None)
                    session.metadata.pop("multi_agent_pending_rounds", None)
                    await _persist_session_metadata(session, session_manager)
                    return await _run_multi_agent_executor(
                        message=cleaned_topic or topic,
                        session_id=session_id,
                        config=config,
                        on_stream=on_stream,
                        router=router,
                        registry=registry,
                        images=images,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        ma_cfg=ma_cfg,
                        on_round_result=on_round_result,
                    )

                if waiting and not _is_multi_agent_confirm(message):
                    topic = pending_topic
                    if message.strip() and not _is_multi_agent_cancel(message):
                        topic = f"{topic}\n补充要求: {message.strip()}".strip()
                        session.metadata["multi_agent_pending_topic"] = topic
                        await _persist_session_metadata(session, session_manager)
                    rounds_raw = session.metadata.get("multi_agent_pending_rounds")
                    if isinstance(rounds_raw, int):
                        ma_cfg["max_rounds"] = max(1, min(rounds_raw, 10))
                    else:
                        _, marker_rounds = _extract_rounds_marker(topic or "")
                        if marker_rounds is not None:
                            ma_cfg["max_rounds"] = marker_rounds
                    topic = topic or "（请补充你的任务目标）"
                    return await _run_multi_agent_controller_discussion(
                        user_message=message,
                        pending_topic=topic,
                        cfg=ma_cfg,
                        session_id=session_id,
                        config=config,
                        router=router,
                        registry=registry,
                        extra_memory=extra_memory,
                        trigger_event_id=trigger_event_id,
                        trigger_text_preview=trigger_text_preview,
                        include_intro=False,
                    )

                session.metadata["multi_agent_state"] = "discuss"
                session.metadata["multi_agent_waiting_confirm"] = False
                session.metadata["multi_agent_intro_done"] = True
                session.metadata["multi_agent_pending_topic"] = message.strip() or message
                session.metadata.pop("multi_agent_pending_rounds", None)
                await _persist_session_metadata(session, session_manager)
                return await _run_multi_agent_controller_discussion(
                    user_message=message,
                    pending_topic=message.strip() or message,
                    cfg=ma_cfg,
                    session_id=session_id,
                    config=config,
                    router=router,
                    registry=registry,
                    extra_memory=extra_memory,
                    trigger_event_id=trigger_event_id,
                    trigger_text_preview=trigger_text_preview,
                    include_intro=True,
                )

            return await _run_multi_agent_executor(
                message=message,
                session_id=session_id,
                config=config,
                on_stream=on_stream,
                router=router,
                registry=registry,
                images=images,
                extra_memory=extra_memory,
                trigger_event_id=trigger_event_id,
                trigger_text_preview=trigger_text_preview,
                ma_cfg=ma_cfg,
                on_round_result=on_round_result,
            )

    if _is_evomap_status_question(message):
        enabled = _is_evomap_enabled(config)
        switch_text = "已开启" if enabled else "已关闭"
        return (
            f"当前 EvoMap 开关{switch_text}（本地配置状态）。"
            "如果你要我检查远端服务连通性，我可以再单独做一次连通检测。"
        )

    metadata_dirty = False
    assistant_name = _DEFAULT_ASSISTANT_NAME

    llm_message = message
    locked_skill_ids: list[str] = []
    previous_locked_skill_ids: list[str] = []
    lock_is_explicit = False
    pending_lock_skill_ids: list[str] = []
    lock_waiting_done = False
    skill_lock_resumed_from_waiting = False
    skill_announce_pending = False
    skill_queue: list[dict[str, str]] = []
    skill_queue_index: int = 0
    has_skill_queue = False
    _skill_recommended_commands: dict[str, str] = {}
    routed_skills: list[Skill] = []
    routed_skill_ids: list[str] = []
    if not multi_agent_internal:
        if session is not None:
            raw_queue = session.metadata.get("skill_queue")
            if isinstance(raw_queue, list) and raw_queue:
                queue_items = cast(list[object], raw_queue)
                skill_queue = [
                    {
                        key: str(value)
                        for key, value in cast(dict[object, object], item).items()
                        if isinstance(key, str)
                    }
                    for item in queue_items
                    if isinstance(item, dict)
                ]
                raw_idx = session.metadata.get("skill_queue_index")
                skill_queue_index = int(raw_idx) if isinstance(raw_idx, int) else 0
                has_skill_queue = bool(skill_queue)
            raw_locked = session.metadata.get("locked_skill_ids")
            if isinstance(raw_locked, list):
                locked_skill_ids = [
                    str(x).strip().lower()  # pyright: ignore[reportUnknownArgumentType]
                    for x in raw_locked  # pyright: ignore[reportUnknownVariableType]
                    if isinstance(x, str) and str(x).strip()
                ]
                if locked_skill_ids:
                    lock_is_explicit = True
            elif isinstance(session.metadata.get("forced_skill_id"), str):
                legacy_forced = str(session.metadata.get("forced_skill_id", "")).strip().lower()
                if legacy_forced:
                    locked_skill_ids = [legacy_forced]
                    lock_is_explicit = True
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata.pop("forced_skill_id", None)
                    metadata_dirty = True
            raw_waiting = session.metadata.get("skill_lock_waiting_done")
            lock_waiting_done = bool(raw_waiting)
            skill_announce_pending = bool(session.metadata.get("skill_lock_announce_pending"))
        if _is_skill_lock_status_question(message):
            status_session = session
            if session is not None and session_manager is not None:
                refreshed_session = await session_manager.get(session.id)
                if refreshed_session is not None:
                    session.metadata = refreshed_session.metadata
                    status_session = refreshed_session
            return _build_skill_lock_status_reply(status_session)
        previous_locked_skill_ids = list(locked_skill_ids)
        if session is not None:
            session.metadata.pop("pending_skill_switch_ids", None)
            session.metadata.pop("pending_skill_switch_message", None)

        if lock_is_explicit and locked_skill_ids:
            _activation_skills = _assembler.route_skills(message, forced_skill_ids=locked_skill_ids)
            for _act_skill in _activation_skills:
                _act_hooks = get_skill_hooks(_act_skill)
                if _act_hooks is not None and _act_hooks.is_activation_message(message):
                    _already_reply = _act_hooks.build_already_locked_reply(session)
                    if _already_reply is not None:
                        return _already_reply
        _locked_hooks_for_ref = [
            get_skill_hooks(s)
            for s in _assembler.route_skills(llm_message, forced_skill_ids=locked_skill_ids)
            if get_skill_hooks(s) is not None
        ] if lock_is_explicit and locked_skill_ids else []
        if lock_is_explicit and locked_skill_ids and _is_image_reference_lookup_message(
            llm_message, locked_hooks=_locked_hooks_for_ref, session=session,  # type: ignore[arg-type]
        ):
            ref_path = _resolve_relative_image_reference_path(session, llm_message)
            if not ref_path:
                # 用户没有指定"这张"/"上一张"，兜底取最新生成的图片
                recent = _recover_recent_session_image_paths(session, limit=1)
                if recent:
                    ref_path = recent[0]
            if ref_path:
                return f"你说的这张是：\n\n![历史图片]({ref_path})"

        use_cmd = _parse_use_command(message, use_cmd_re=_USE_CMD_RE)
        if use_cmd is not None:
            use_skill_ids, remainder = use_cmd
            if len(use_skill_ids) == 1 and use_skill_ids[0] in _USE_CLEAR_IDS:
                locked_skill_ids = []
                lock_is_explicit = False
                pending_lock_skill_ids = []
                lock_waiting_done = False
                skill_announce_pending = False
                if session is not None:
                    session.metadata.pop("locked_skill_ids", None)
                    session.metadata.pop("skill_lock_waiting_done", None)
                    session.metadata.pop("skill_lock_announce_pending", None)
                    session.metadata.pop("pending_skill_switch_ids", None)
                    session.metadata.pop("pending_skill_switch_message", None)
                    session.metadata.pop("skill_param_state", None)
                    session.metadata.pop("skill_queue", None)
                    session.metadata.pop("skill_queue_index", None)
                    session.metadata.pop("skill_queue_original_message", None)
                    metadata_dirty = True
                skill_queue = []
                has_skill_queue = False
                if remainder:
                    llm_message = remainder
            else:
                forced_skills = _assembler.route_skills(
                    remainder or message,
                    forced_skill_ids=use_skill_ids,
                )
                lockable_use_skill_ids = _lockable_skill_ids(forced_skills)
                locked_skill_ids = lockable_use_skill_ids
                lock_is_explicit = bool(lockable_use_skill_ids)
                lock_waiting_done = False
                skill_announce_pending = bool(lockable_use_skill_ids)
                if session is not None:
                    if lockable_use_skill_ids:
                        session.metadata["locked_skill_ids"] = locked_skill_ids
                        session.metadata["skill_lock_waiting_done"] = False
                        session.metadata["skill_lock_announce_pending"] = True
                        session.metadata.pop("skill_param_state", None)
                        # 切换技能时清理上一轮遗留的图片/草稿数据
                        session.metadata.pop("last_generated_image_path", None)
                        session.metadata.pop("last_input_image_paths", None)
                        session.metadata.pop("image_reference_history", None)
                        _snapshot_image_history_at_activation(session)
                    else:
                        session.metadata.pop("locked_skill_ids", None)
                        session.metadata.pop("skill_lock_waiting_done", None)
                        session.metadata.pop("skill_lock_announce_pending", None)
                        session.metadata.pop("skill_param_state", None)
                        session.metadata.pop("skill_queue", None)
                        session.metadata.pop("skill_queue_index", None)
                        session.metadata.pop("skill_queue_original_message", None)
                        session.metadata.pop("__skill_activation_image_snapshot__", None)
                        session.metadata.pop("last_generated_image_path", None)
                        session.metadata.pop("last_input_image_paths", None)
                        session.metadata.pop("image_reference_history", None)
                    metadata_dirty = True
                if not lockable_use_skill_ids:
                    skill_queue = []
                    has_skill_queue = False
                llm_message = remainder or f"使用技能 {', '.join(use_skill_ids)} 处理当前请求。"
        elif lock_is_explicit and locked_skill_ids and _is_task_done_confirmation(
            message,
            task_done_patterns=_TASK_DONE_PATTERNS,
        ):
            locked_skill_ids = []
            lock_is_explicit = False
            lock_waiting_done = False
            has_skill_queue = False
            skill_queue = []
            if session is not None:
                session.metadata.pop("locked_skill_ids", None)
                session.metadata.pop("skill_lock_waiting_done", None)
                session.metadata.pop("skill_lock_announce_pending", None)
                session.metadata.pop("pending_skill_switch_ids", None)
                session.metadata.pop("pending_skill_switch_message", None)
                session.metadata.pop("skill_param_state", None)
                session.metadata.pop("skill_queue", None)
                session.metadata.pop("skill_queue_index", None)
                session.metadata.pop("skill_queue_original_message", None)
                session.metadata.pop("__skill_activation_image_snapshot__", None)
                # 清理上一轮技能遗留的图片/草稿数据，防止下次使用技能时互相污染
                session.metadata.pop("last_generated_image_path", None)
                session.metadata.pop("last_input_image_paths", None)
                session.metadata.pop("image_reference_history", None)
                if session_manager is not None:
                    await session_manager.update_metadata(session, session.metadata)
            return "已确认任务完成，已解除本轮技能锁定。"
        elif (
            has_skill_queue
            and lock_waiting_done
            and _is_queue_advance_confirmation(message)
        ):
            # 队列推进：用户确认"继续"，推进到下一个技能
            next_idx = skill_queue_index + 1
            if next_idx < len(skill_queue):
                if skill_queue_index < len(skill_queue):
                    skill_queue[skill_queue_index]["status"] = "done"
                skill_queue[next_idx]["status"] = "active"
                skill_queue_index = next_idx
                next_skill_id = skill_queue[next_idx]["skill_id"]
                locked_skill_ids = [next_skill_id]
                lock_is_explicit = True
                lock_waiting_done = False
                skill_announce_pending = False
                if session is not None:
                    session.metadata["skill_queue"] = skill_queue
                    session.metadata["skill_queue_index"] = skill_queue_index
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata.pop("skill_param_state", None)
                    _snapshot_image_history_at_activation(session)
                    metadata_dirty = True
                next_task = skill_queue[next_idx].get("task", "")
                original_msg = ""
                if session is not None:
                    original_msg = str(
                        session.metadata.get("skill_queue_original_message", "")
                    )
                llm_message = (
                    f"继续执行第 {next_idx + 1} 步: {next_task}"
                    + (f"\n原始任务: {original_msg}" if original_msg else "")
                )
                log.info(
                    "agent.skill_queue_advanced",
                    session_id=session_id,
                    new_index=next_idx,
                    skill_id=next_skill_id,
                )
            else:
                # 已是最后一步，走正常解锁
                locked_skill_ids = []
                lock_is_explicit = False
                lock_waiting_done = False
                has_skill_queue = False
                if session is not None:
                    session.metadata.pop("skill_queue", None)
                    session.metadata.pop("skill_queue_index", None)
                    session.metadata.pop("skill_queue_original_message", None)
                    session.metadata.pop("locked_skill_ids", None)
                    session.metadata.pop("skill_lock_waiting_done", None)
                    session.metadata.pop("skill_lock_announce_pending", None)
                    session.metadata.pop("skill_param_state", None)
                    if session_manager is not None:
                        await session_manager.update_metadata(session, session.metadata)
                return "全部子任务已完成，已解除技能锁定。"
        elif lock_waiting_done and _is_task_done_intent_near_miss(message):
            return (
                "还没有完成正式解锁。"
                "请直接回复\u201c任务完成\u201d或\u201c任务结束\u201d以解除技能锁定。"
            )
        elif lock_waiting_done:
            lock_waiting_done = False
            skill_lock_resumed_from_waiting = True
            if session is not None:
                session.metadata["skill_lock_waiting_done"] = False
                metadata_dirty = True

        routed_skills = _assembler.route_skills(llm_message)
        routed_skill_ids = _normalize_skill_ids(routed_skills)
        lockable_routed_skill_ids = _lockable_skill_ids(routed_skills)

        is_deferred = _is_deferred_task_intent(message) and not lock_is_explicit
        if is_deferred:
            routed_skills = []
            routed_skill_ids = []
            lockable_routed_skill_ids = []

        _is_compound_with_other_steps = (
            routed_skill_ids
            and any(_skill_trigger_mentioned(skill, message) for skill in routed_skills)
            and not _looks_like_skill_activation_message(
                message, skill_activation_patterns=_SKILL_ACTIVATION_PATTERNS,
            )
            and _NON_SKILL_STEP_RE.search(message) is not None
        )

        _is_natural_activation = (
            not locked_skill_ids
            and routed_skill_ids
            and not _is_compound_with_other_steps
            and use_cmd is None
            and (
                _looks_like_skill_activation_message(
                    message,
                    skill_activation_patterns=_SKILL_ACTIVATION_PATTERNS,
                )
                or any(_skill_trigger_mentioned(skill, message) for skill in routed_skills)
            )
        )
        if _is_natural_activation:
            if len(lockable_routed_skill_ids) >= 2 and session is not None:
                # 多 lockable 技能 → 调 LLM 做任务拆解，建立串行队列
                plan_messages = _build_skill_queue_plan_messages(
                    lockable_routed_skill_ids, llm_message,
                )
                try:
                    plan_resp = await router.chat(
                        model_id, plan_messages, tools=None, on_stream=None,
                    )
                    parsed_queue = _parse_skill_queue_plan(
                        plan_resp.content or "", lockable_routed_skill_ids,
                    )
                except Exception:
                    parsed_queue = []
                if len(parsed_queue) >= 2:
                    parsed_queue[0]["status"] = "active"
                    skill_queue = parsed_queue
                    skill_queue_index = 0
                    has_skill_queue = True
                    first_skill_id = parsed_queue[0]["skill_id"]
                    locked_skill_ids = [first_skill_id]
                    lock_is_explicit = True
                    lock_waiting_done = False
                    skill_announce_pending = False
                    session.metadata["skill_queue"] = skill_queue
                    session.metadata["skill_queue_index"] = 0
                    session.metadata["skill_queue_original_message"] = llm_message
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata["skill_lock_announce_pending"] = False
                    _snapshot_image_history_at_activation(session)
                    metadata_dirty = True
                    status_text = _build_skill_queue_status_message(
                        skill_queue, skill_queue_index,
                    )
                    if session_manager is not None:
                        await _persist_message(
                            session_manager, session, "assistant", status_text,
                        )
                    log.info(
                        "agent.skill_queue_created",
                        session_id=session_id,
                        queue_len=len(skill_queue),
                        skill_ids=[item["skill_id"] for item in skill_queue],
                    )
                else:
                    # LLM 拆解失败，回退到现有并列锁定
                    locked_skill_ids = lockable_routed_skill_ids
                    lock_is_explicit = bool(lockable_routed_skill_ids)
                    lock_waiting_done = False
                    skill_announce_pending = bool(lockable_routed_skill_ids)
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata["skill_lock_announce_pending"] = True
                    _snapshot_image_history_at_activation(session)
                    metadata_dirty = True
            else:
                # 用户说"使用xxx技能"明确激活时，即使 lock_session=False 也锁定
                _explicit_activation = _looks_like_skill_activation_message(
                    message, skill_activation_patterns=_SKILL_ACTIVATION_PATTERNS,
                )
                effective_lock_ids = (
                    lockable_routed_skill_ids
                    or (routed_skill_ids if _explicit_activation else [])
                )
                locked_skill_ids = effective_lock_ids
                lock_is_explicit = bool(effective_lock_ids)
                lock_waiting_done = False
                skill_announce_pending = bool(effective_lock_ids)
                if session is not None and effective_lock_ids:
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata["skill_lock_announce_pending"] = True
                    _snapshot_image_history_at_activation(session)
                    metadata_dirty = True
        elif not locked_skill_ids and lockable_routed_skill_ids:
            pending_lock_skill_ids = lockable_routed_skill_ids

        if (
            lock_is_explicit
            and locked_skill_ids
            and routed_skill_ids
            and routed_skill_ids != locked_skill_ids
        ):
            _keep_locked_route = False
            for _sid in locked_skill_ids:
                _locked_skill_list = _assembler.route_skills("", forced_skill_ids=[_sid])
                for _ls in _locked_skill_list:
                    _lh = get_skill_hooks(_ls)
                    if _lh is not None and (
                        _is_numbered_image_reference_edit_message(message)
                        or _message_requests_image_regenerate(message)
                    ):
                        _keep_locked_route = True
                        break
            if _keep_locked_route:
                routed_skill_ids = []
                routed_skills = []
                lockable_routed_skill_ids = []
            else:
                # 只有明确提及了用户安装技能名称时才拦截，否则静默丢弃
                explicit_switch = any(
                    (
                        _skill_explicitly_mentioned(skill, message)
                        or _skill_trigger_mentioned(skill, message)
                        or _raw_skill_trigger_mentioned(skill, message)
                    )
                    for skill in routed_skills
                    if skill.id.strip().lower() not in set(locked_skill_ids)
                    and skill.is_user_installed
                )
                if explicit_switch:
                    requested_names = "、".join(routed_skill_ids)
                    current_names = "、".join(locked_skill_ids)
                    if session is not None:
                        session.metadata["pending_skill_switch_ids"] = routed_skill_ids
                        session.metadata["pending_skill_switch_message"] = llm_message
                        metadata_dirty = True
                    return (
                        f"当前会话仍锁定在 {current_names} 技能。"
                        f"如果你确实要切换到 {requested_names}，请先回复“任务完成”，"
                        "之后再重新输入目标命令。"
                    )
                routed_skill_ids = []
                routed_skills = []
                lockable_routed_skill_ids = []

        active_skills_for_images = routed_skills
        if lock_is_explicit and locked_skill_ids:
            active_skills_for_images = _assembler.route_skills(
                llm_message, forced_skill_ids=locked_skill_ids
            )
        _early_hooks_non_exec = False
        if lock_is_explicit and locked_skill_ids:
            _input_paths_for_check = _extract_input_image_paths_from_text(llm_message)
            _has_input_for_check = bool(_input_paths_for_check)
            for _esk in active_skills_for_images:
                _eh = get_skill_hooks(_esk)
                if _eh is not None and _eh.is_execution_request(
                    llm_message, has_new_input_images=_has_input_for_check, session=session,
                ) is False:
                    _early_hooks_non_exec = True
                    break
        if not images:
            recovered_images: list[ImageContent] = []
            reuse_reason = ""
            skill_needs_images = _skill_requires_images(active_skills_for_images)
            if _early_hooks_non_exec:
                skill_needs_images = False
            # guard 阶段且从未执行过（没有生成过图片），不要注入旧图
            if (
                skill_needs_images
                and lock_is_explicit
                and not lock_waiting_done
                and not (session and session.metadata.get("last_generated_image_path"))
            ):
                skill_needs_images = False
            if _message_requests_image_regenerate(llm_message):
                _skip_reuse = False
                if lock_is_explicit and locked_skill_ids:
                    for _rsk in active_skills_for_images:
                        _rh = get_skill_hooks(_rsk)
                        if _rh is not None:
                            _rh_state = (session.metadata.get("skill_param_state", {}) or {}).get(_rsk.id, {}) if session else {}
                            _rh_mode = _rh_state.get("__mode__") or _rh_state.get("__last_mode__", "")
                            if str(_rh_mode).strip().lower() == "text":
                                _skip_reuse = True
                                break
                if not _skip_reuse:
                    recovered_images = _recover_last_input_images(session)
                    reuse_reason = "last_input_images"
            elif _message_requests_image_edit(llm_message) and (
                _message_may_need_prior_images(llm_message) or skill_needs_images
            ):
                recovered_images = _recover_latest_generated_image(session)
                reuse_reason = "latest_generated_image"
                if not recovered_images:
                    recovered_images = _recover_last_input_images(session)
                    reuse_reason = "last_input_images_fallback"
            if not recovered_images and (
                _message_may_need_prior_images(llm_message)
                or (skill_needs_images and lock_is_explicit)
            ):
                recovered_images = _recover_last_input_images(session)
                reuse_reason = "skill_or_reference_images"
            if not recovered_images:
                inline_paths = _extract_input_image_paths_from_text(llm_message)
                if inline_paths:
                    recovered_images = _load_images_from_paths(inline_paths)
                    reuse_reason = "inline_markdown_paths"
            if recovered_images:
                images = recovered_images
                log.info(
                    "agent.reused_recent_images",
                    session_id=session_id,
                    count=len(recovered_images),
                    reason=reuse_reason,
                )
        has_new_input_images = False
        if session is not None:
            current_input_paths = _extract_input_image_paths_from_text(llm_message)
            has_new_input_images = bool(current_input_paths)
            if current_input_paths:
                session.metadata["last_input_image_paths"] = current_input_paths
                _append_image_reference_history(session.metadata, current_input_paths)
                metadata_dirty = True
        _hooks_control_only = False
        _hooks_activation_only = False
        _hooks_execution_request = False
        _hooks_skip_guard_ids: set[str] = set()
        if lock_is_explicit and locked_skill_ids:
            _hooks_locked_skills = _assembler.route_skills(llm_message, forced_skill_ids=locked_skill_ids)
            for _hls in _hooks_locked_skills:
                _hlh = get_skill_hooks(_hls)
                if _hlh is None:
                    continue
                if _hlh.is_control_message(llm_message):
                    _hooks_control_only = True
                if _hlh.is_activation_message(llm_message):
                    _hooks_activation_only = True
                _exec_req = _hlh.is_execution_request(
                    llm_message, has_new_input_images=has_new_input_images, session=session,
                )
                if _exec_req is True:
                    _hooks_execution_request = True
                if not _hooks_control_only and _exec_req is False:
                    _hooks_skip_guard_ids.add(_hls.id)
                    active_skills_for_images = []

        if (
            _is_compound_with_other_steps
            and lock_is_explicit
            and locked_skill_ids
            and session is not None
        ):
            pending_lock_skill_ids = list(locked_skill_ids)
            locked_skill_ids = []
            lock_is_explicit = False
            lock_waiting_done = False
            skill_announce_pending = False
            session.metadata.pop("locked_skill_ids", None)
            session.metadata.pop("skill_lock_waiting_done", None)
            session.metadata.pop("skill_lock_announce_pending", None)
            session.metadata.pop("skill_param_state", None)
            metadata_dirty = True

        assistant_name = _DEFAULT_ASSISTANT_NAME
        if lock_is_explicit and locked_skill_ids and not lock_waiting_done and session is not None:
            forced_skill_ids = locked_skill_ids
            if _hooks_skip_guard_ids:
                forced_skill_ids = [
                    skill_id for skill_id in locked_skill_ids if skill_id not in _hooks_skip_guard_ids
                ]
            locked_skills = _assembler.route_skills(llm_message, forced_skill_ids=forced_skill_ids)
            guards = _guarded_skills(locked_skills)
            if guards:
                state_map_raw = session.metadata.get("skill_param_state")
                state_map: dict[str, dict[str, object]] = {}
                if isinstance(state_map_raw, dict):
                    raw_state_map = cast(dict[object, object], state_map_raw)
                    for key, value in raw_state_map.items():
                        if not isinstance(key, str) or not isinstance(value, dict):
                            continue
                        normalized_value: dict[str, object] = {
                            item_key: item_value
                            for item_key, item_value in cast(dict[object, object], value).items()
                            if isinstance(item_key, str)
                        }
                        state_map[key] = normalized_value
                missing_any = False
                for skill in guards:
                    guard = skill.param_guard
                    if guard is None:
                        continue
                    skill_state_raw = state_map.get(skill.id, {})
                    skill_state: dict[str, object] = (
                        skill_state_raw.copy()  # pyright: ignore[reportUnknownMemberType]
                        if isinstance(skill_state_raw, dict)  # pyright: ignore[reportUnnecessaryIsInstance]
                        else {}
                    )
                    _key_persisted = _persist_param_api_key(guard.params, llm_message)
                    if _key_persisted:
                        skill_state["api_key"] = "__present__"
                        state_map[skill.id] = skill_state
                        session.metadata["skill_param_state"] = state_map
                        if session_manager is not None:
                            await session_manager.update_metadata(session, session.metadata)
                        return "API Key 已保存。需要生图时直接发提示词即可。"
                    _skill_hooks = get_skill_hooks(skill)
                    if _skill_hooks is not None and skill.id in _hooks_skip_guard_ids:
                        state_map[skill.id] = skill_state
                        continue
                    # 只有本轮确实有新图片时才把 images 传给 guard，
                    # 避免恢复的旧图片被误计入 guard state
                    _guard_images = images if has_new_input_images else None
                    if _skill_hooks is not None:
                        for _gp in guard.params:
                            if _gp.type == "api_key" and not skill_state.get(_gp.key):
                                from whaleclaw.agent.helpers.skill_helpers import capture_param_value, param_satisfied
                                _captured = capture_param_value(_gp, llm_message, _guard_images, skill_state.get(_gp.key))
                                if param_satisfied(_gp, _captured):
                                    skill_state[_gp.key] = _captured
                        hooks_updated = _skill_hooks.update_guard_state(
                            skill_state, llm_message, _guard_images,
                            session=session, has_new_input_images=has_new_input_images,
                        )
                        if hooks_updated is not None:
                            updated = hooks_updated
                            _effective_ctrl_only = _hooks_control_only and not _hooks_execution_request
                            hooks_missing = _skill_hooks.missing_required(
                                updated, control_message_only=_effective_ctrl_only,
                            )
                            missing = hooks_missing if hooks_missing is not None else True
                        else:
                            updated, missing = _update_guard_state(
                                guard.params, skill_state, llm_message, _guard_images
                            )
                            _effective_ctrl_only2 = _hooks_control_only and not _hooks_execution_request
                            hooks_missing = _skill_hooks.missing_required(
                                updated, control_message_only=_effective_ctrl_only2,
                            )
                            if hooks_missing is not None:
                                missing = hooks_missing
                    else:
                        updated, missing = _update_guard_state(
                            guard.params, skill_state, llm_message, _guard_images
                        )
                    state_map[skill.id] = updated
                    if skill.id == "nano-banana-image-t8":
                        model_display = updated.get("__model_display__")
                        if model_display in {"香蕉2", "香蕉pro"}:
                            session.metadata["last_nano_banana_model_display"] = model_display
                    missing_any = missing_any or missing
                session.metadata["skill_param_state"] = state_map
                metadata_dirty = True
                # 通用 hooks: 控制消息直接返回
                if _hooks_control_only and not _hooks_activation_only and not _hooks_execution_request:
                    for _ctrl_skill in guards:
                        _ctrl_hooks = get_skill_hooks(_ctrl_skill)
                        if _ctrl_hooks is not None and _ctrl_hooks.is_control_message(llm_message):
                            if session_manager is not None:
                                await session_manager.update_metadata(session, session.metadata)
                            _ctrl_reply = _ctrl_hooks.handle_control_message(
                                llm_message, state_map.get(_ctrl_skill.id, {}), session,
                            )
                            if _ctrl_reply is not None:
                                return _ctrl_reply
                if missing_any:
                    if session_manager is not None:
                        await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            session.id,
                            metadata=session.metadata,
                        )
                    blocks = [
                        _build_skill_param_guard_reply(
                            s.id, s.param_guard.params, state_map.get(s.id, {}),  # pyright: ignore[reportUnknownArgumentType]
                            hooks=get_skill_hooks(s),
                        )
                        for s in guards
                        if s.param_guard is not None
                    ]
                    return "\n\n".join(blocks)
                # 通用 hooks: 命令构建
                for _cmd_skill in guards:
                    _cmd_hooks = get_skill_hooks(_cmd_skill)
                    if (
                        _cmd_hooks is not None
                        and _hooks_execution_request
                        and registry.get("bash") is not None
                    ):
                        _cmd_state = state_map.get(_cmd_skill.id, {})
                        _recommended_cmd = _cmd_hooks.build_command(_cmd_state, session)
                        if _recommended_cmd:
                            _skill_recommended_commands[_cmd_skill.id] = _recommended_cmd
                            metadata_dirty = True
        assistant_name = _DEFAULT_ASSISTANT_NAME
        if memory_manager is not None:
            try:
                current_name = await memory_manager.get_assistant_name()
                if current_name:
                    assistant_name = current_name
            except Exception as exc:
                log.debug("agent.assistant_name_load_failed", session_id=session_id, error=str(exc))
    name_action, requested_name = _detect_assistant_name_update(
        message,
        reset_patterns=_ASSISTANT_NAME_RESET_PATTERNS,
        set_patterns=_ASSISTANT_NAME_SET_PATTERNS,
    )
    if name_action == "set":
        assistant_name = requested_name
        if memory_manager is not None:
            try:
                changed = await memory_manager.set_assistant_name(
                    requested_name,
                    source=f"session:{session_id}",
                )
                if changed:
                    log.info(
                        "agent.assistant_name_updated",
                        session_id=session_id,
                        assistant_name=requested_name,
                    )
            except Exception as exc:
                log.debug(
                    "agent.assistant_name_save_failed",
                    session_id=session_id,
                    error=str(exc),
                )
    elif name_action == "reset":
        assistant_name = _DEFAULT_ASSISTANT_NAME
        if memory_manager is not None:
            try:
                removed = await memory_manager.clear_assistant_name()
                log.info(
                    "agent.assistant_name_reset",
                    session_id=session_id,
                    removed=removed,
                )
            except Exception as exc:
                log.debug(
                    "agent.assistant_name_reset_failed",
                    session_id=session_id,
                    error=str(exc),
                )

    native_tools = router.supports_native_tools(model_id)
    selected_tool_names: set[str] | None = None
    evomap_enabled = _is_evomap_enabled(config) and not multi_agent_internal
    evomap_phase = "editing"
    if session is not None:
        raw_phase = session.metadata.get("evomap_phase")
        if isinstance(raw_phase, str) and raw_phase.strip():
            evomap_phase = raw_phase.strip().lower()
    if _is_tasky_message_for_evomap(llm_message):
        # First-turn task requests are treated as NEW_TASK directly to avoid
        # an extra classifier LLM call before normal planning/execution.
        if session is None or not session.messages:
            phase = "NEW_TASK"
        else:
            phase = await _llm_judge_task_phase(
                router,
                model_id,
                session=session,
                message=llm_message,
            )
    else:
        phase = "EDITING"
    if session is not None and session_manager is not None:
        desired = "start" if phase == "NEW_TASK" else "editing"
        if desired != evomap_phase:
            session.metadata["evomap_phase"] = desired
            await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                session.id,
                metadata=session.metadata,
            )
    evomap_allowed_for_turn = evomap_enabled and phase == "NEW_TASK"
    available_tool_names = {d.name for d in registry.list_tools()}
    evo_first_mode = evomap_allowed_for_turn and "evomap_fetch" in available_tool_names
    evomap_hint_hit = _extra_memory_has_evomap_hint(extra_memory)

    _DEFERRED_ONLY_TOOLS = {"cron", "reminder"}

    if native_tools:
        selected_tool_names = _select_native_tool_names(registry, llm_message)
        selected_tool_names = _force_include_office_edit_tools(
            selected_tool_names,
            available=available_tool_names,
            session=session,
            llm_message=llm_message,
        )
        if not evomap_allowed_for_turn:
            selected_tool_names = {
                name for name in selected_tool_names if not name.startswith("evomap_")
            }
        elif evo_first_mode:
            selected_tool_names.add("evomap_fetch")
        if _skill_recommended_commands:
            for _rc_skill_id, _rc_cmd in _skill_recommended_commands.items():
                _rc_skills = _assembler.route_skills("", forced_skill_ids=[_rc_skill_id])
                for _rc_s in _rc_skills:
                    _rc_h = get_skill_hooks(_rc_s)
                    if _rc_h is not None:
                        for _extra_t in _rc_h.extra_tool_names():
                            if _extra_t in available_tool_names:
                                selected_tool_names.add(_extra_t)
        if is_deferred and selected_tool_names is not None:
            selected_tool_names = selected_tool_names & _DEFERRED_ONLY_TOOLS
            if not selected_tool_names:
                selected_tool_names = _DEFERRED_ONLY_TOOLS & available_tool_names
            log.info(
                "agent.deferred_intent_tool_restriction",
                session_id=session_id,
                restricted_to=sorted(selected_tool_names),
            )
        tool_schemas = registry.to_llm_schemas(include_names=selected_tool_names)
        dropped_names = sorted(available_tool_names - selected_tool_names)
        log.info(
            "agent.tools_selected",
            session_id=session_id,
            selected=sorted(selected_tool_names),
            selected_count=len(selected_tool_names),
            dropped_count=len(dropped_names),
            dropped=dropped_names,
        )
    else:
        tool_schemas = None
    fallback_names: set[str] | None = None
    if not native_tools and not evomap_allowed_for_turn:
        fallback_names = {d.name for d in registry.list_tools() if not d.name.startswith("evomap_")}
    if is_deferred and fallback_names is not None:
        fallback_names = fallback_names & _DEFERRED_ONLY_TOOLS
        if not fallback_names:
            fallback_names = _DEFERRED_ONLY_TOOLS & available_tool_names
    fallback_text = (
        "" if native_tools else registry.to_prompt_fallback(include_names=fallback_names)
    )

    effective_skill_ids = locked_skill_ids or pending_lock_skill_ids or None
    _model_display = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    system_messages = _assembler.build(
        config,
        llm_message,
        tool_fallback_text=fallback_text,
        assistant_name=assistant_name,
        forced_skill_ids=effective_skill_ids,
        model_id=model_id,
        max_context_tokens=ContextWindow.get_max_context(_model_display),
    )
    if lock_is_explicit and locked_skill_ids:
        system_messages.append(_build_skill_lock_system_message(locked_skill_ids))
        if skill_lock_resumed_from_waiting:
            joined_ids = ", ".join(locked_skill_ids)
            _is_confirm_word = bool(re.match(
                r"^\s*(?:确认|ok|好的?|可以|没问题|就这样|对的?|行|嗯|发吧|开始吧|go|开始)\s*[!！。.]*$",
                llm_message, re.IGNORECASE,
            ))
            if _is_confirm_word:
                system_messages.append(Message(
                    role="system",
                    content=(
                        f"用户已确认，当前锁定在 {joined_ids} 技能。\n"
                        "请立即按照技能规则继续执行下一步，直接调用工具完成任务。\n"
                        "禁止输出使用说明或重复展示已有内容，直接动手做。"
                    ),
                ))
            else:
                system_messages.append(Message(
                    role="system",
                    content=(
                        f"用户刚刚发送了新的任务请求，当前仍锁定在 {joined_ids} 技能。\n"
                        "请立即按照技能规则执行用户的新请求，直接调用工具完成任务。\n"
                        "禁止输出使用说明、操作指南或功能介绍，直接动手做。"
                    ),
                ))
        # 通用 hooks: 执行约束 system message
        _exec_locked_skills = _assembler.route_skills(llm_message, forced_skill_ids=locked_skill_ids)
        for _exec_skill in _exec_locked_skills:
            _exec_hooks = get_skill_hooks(_exec_skill)
            if _exec_hooks is not None:
                _exec_msg = _exec_hooks.build_execution_system_message(session)
                if _exec_msg is not None:
                    system_messages.append(_exec_msg)
    # 通用 hooks: 命令模板 system message
    for _tpl_skill_id, _tpl_cmd in _skill_recommended_commands.items():
        _tpl_skills = _assembler.route_skills("", forced_skill_ids=[_tpl_skill_id])
        for _tpl_s in _tpl_skills:
            _tpl_h = get_skill_hooks(_tpl_s)
            if _tpl_h is not None:
                _tpl_msg = _tpl_h.build_command_template_system_message(_tpl_cmd)
                if _tpl_msg is not None:
                    system_messages.append(_tpl_msg)
    # 通用 hooks: 阶段规则 system message
    if lock_is_explicit and locked_skill_ids and session is not None:
        _stage_locked_skills = _assembler.route_skills(llm_message, forced_skill_ids=locked_skill_ids)
        _skill_param_state_raw = session.metadata.get("skill_param_state") or {}
        for _stage_skill in _stage_locked_skills:
            _stage_hooks = get_skill_hooks(_stage_skill)
            if _stage_hooks is None:
                continue
            _stage_state = {}
            if isinstance(_skill_param_state_raw, dict):
                _stage_state = _skill_param_state_raw.get(_stage_skill.id, {})
                if not isinstance(_stage_state, dict):
                    _stage_state = {}
            for _rule in _stage_hooks.stage_rules:
                if _rule.condition(_stage_state, llm_message):
                    system_messages.append(Message(role="system", content=_rule.get_hint(_stage_state, llm_message, session)))
    _append_office_system_hints(system_messages, session, llm_message)

    if (
        pending_lock_skill_ids
        and _NON_SKILL_STEP_RE.search(llm_message) is not None
    ):
        system_messages.append(Message(
            role="system",
            content=(
                "这是一个复合任务，用户消息中包含多个步骤。\n"
                "请按用户描述的顺序逐步执行，每一步都调用工具完成。\n"
                "用户消息中已包含全部所需信息，不要向用户追问参数。\n"
                "技能文档中关于追问缺失参数的指引在此场景下不适用。"
            ),
        ))

    image_api_probe_guard_enabled = (
        _is_image_generation_request(llm_message)
        and _NON_SKILL_STEP_RE.search(llm_message) is None
    )
    if image_api_probe_guard_enabled:
        system_messages.append(_build_image_generation_system_message())
    if evo_first_mode:
        system_messages.append(_build_evomap_first_system_message())
        if session is not None and session_manager is not None:
            session.metadata["evomap_phase"] = "editing"
            await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                session.id,
                metadata=session.metadata,
            )
    import time as _time

    _t_pre_llm_start = _time.monotonic()
    _memory_stage_ms = 0
    _extra_memory_stage_ms = 0
    _group_compress_stage_ms = 0
    if memory_manager is not None and agent_cfg.memory.enabled:
        _t_memory_start = _time.monotonic()
        memory_cfg = agent_cfg.memory
        style_directive = ""
        try:
            style_directive = (
                await memory_manager.get_global_style_directive()
                if memory_cfg.global_style_enabled
                else ""
            )
            if style_directive:
                system_messages.append(_build_global_style_system_message(style_directive))
                log.info(
                    "agent.memory_style_applied",
                    session_id=session_id,
                    chars=len(style_directive),
                )
        except Exception as exc:
            log.debug("agent.memory_style_load_failed", error=str(exc), session_id=session_id)
        try:
            should_recall, include_raw = memory_manager.recall_policy(llm_message)
            if not should_recall and _is_creation_task_message(llm_message):
                should_recall = True
                include_raw = False
            recalled = ""
            if should_recall:
                profile_block = await memory_manager.build_profile_for_injection(
                    max_tokens=memory_cfg.recall_profile_max_tokens,
                    router=router,
                    model_id=memory_cfg.organizer_model,
                    exclude_style=bool(style_directive.strip()),
                )
                raw_block = ""
                if include_raw:
                    raw_block = await memory_manager.recall(
                        llm_message,
                        max_tokens=memory_cfg.recall_raw_max_tokens,
                        limit=memory_cfg.recall_limit,
                        include_profile=False,
                        include_raw=True,
                    )
                recalled = _merge_recall_blocks(profile_block, raw_block)
            if recalled.strip():
                system_messages.append(_build_memory_system_message(recalled))
                log.info("agent.memory_recalled", session_id=session_id, chars=len(recalled))
        except Exception as exc:
            log.debug("agent.memory_recall_failed", error=str(exc), session_id=session_id)
        _memory_stage_ms = int((_time.monotonic() - _t_memory_start) * 1000)
    if extra_memory.strip():
        _t_extra_start = _time.monotonic()
        normalized_extra = extra_memory.strip()
        compress_model = summarizer_cfg.model.strip()
        can_compress = bool(compress_model)
        should_compress_extra = _est_tokens(normalized_extra) > _EVOMAP_MAX_TOKENS
        if can_compress and should_compress_extra:
            try:
                router.resolve(compress_model)
                compressed = await asyncio.wait_for(
                    _compress_external_memory_with_llm(
                        router=router,
                        model_id=compress_model,
                        text=normalized_extra,
                        max_tokens=_EVOMAP_MAX_TOKENS,
                    ),
                    timeout=_EXTRA_MEMORY_COMPRESS_TIMEOUT_SECONDS,
                )
                if compressed:
                    normalized_extra = compressed
                else:
                    normalized_extra = _truncate_to_tokens(
                        normalized_extra,
                        _EVOMAP_MAX_TOKENS,
                    )
            except Exception:
                normalized_extra = _truncate_to_tokens(
                    normalized_extra,
                    _EVOMAP_MAX_TOKENS,
                )
        else:
            normalized_extra = _truncate_to_tokens(
                normalized_extra,
                _EVOMAP_MAX_TOKENS,
            )
        system_messages.append(_build_external_memory_system_message(normalized_extra))
        _extra_memory_stage_ms = int((_time.monotonic() - _t_extra_start) * 1000)

    conversation: list[Message] = []
    if session:
        conversation = list(session.messages)
    current_user_message = Message(role="user", content=llm_message, images=images)
    if (
        conversation
        and conversation[-1].role == "user"
        and conversation[-1].content == llm_message
    ):
        # 某些入口会先把当前用户消息写入 session，再调用 run_agent。
        # 这里直接替换最后一条，避免把同一轮请求以“无图一次 + 带图一次”重复发送给模型。
        previous_images = conversation[-1].images
        conversation[-1] = Message(
            role="user",
            content=llm_message,
            images=images or previous_images,
        )
    else:
        conversation.append(current_user_message)
    conversation_message_count = len(conversation)

    if (
        group_compressor is not None
        and session_store is not None
        and session is not None
        and summarizer_cfg.model.strip()
    ):
        _t_group_start = _time.monotonic()
        try:
            conversation_for_compress = conversation
            if isinstance(session.metadata, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                raw_cutoff = session.metadata.get("compression_resume_message_index")
                if isinstance(raw_cutoff, int) and raw_cutoff > 0:
                    cutoff = max(0, min(raw_cutoff, len(conversation) - 1))
                    if cutoff > 0:
                        conversation_for_compress = conversation[cutoff:]
            conversation = await group_compressor.build_window_messages(
                session_id=session_id,
                messages=conversation_for_compress,
                router=router,
                model_id=summarizer_cfg.model.strip(),
            )
        except Exception as exc:
            log.debug("agent.group_compress_failed", session_id=session_id, error=str(exc))
        _group_compress_stage_ms = int((_time.monotonic() - _t_group_start) * 1000)

    _pre_llm_elapsed_ms = int((_time.monotonic() - _t_pre_llm_start) * 1000)
    log.debug(
        "agent.pre_llm_stages",
        session_id=session_id,
        memory_ms=_memory_stage_ms,
        extra_memory_ms=_extra_memory_stage_ms,
        group_compress_ms=_group_compress_stage_ms,
        total_ms=_pre_llm_elapsed_ms,
    )

    model_short: str = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    trigger_preview = trigger_text_preview.strip() or _preview_text(llm_message)

    log.info(
        "agent.run",
        model=model_id,
        session_id=session_id,
        native_tools=native_tools,
        history_messages=len(conversation),
        trigger_event_id=trigger_event_id,
        trigger_preview=trigger_preview,
    )

    final_text_parts: list[str] = []
    real_image_paths: list[str] = []
    total_input = 0
    total_output = 0
    announced_plan = False
    db_summaries: list[SummaryRow] = []
    pending_office_paths: list[str] = []
    if session_store and summarizer_cfg.enabled:
        try:
            db_summaries = await _load_session_summaries_typed(session_store, session_id)
        except Exception as exc:
            log.debug("agent.summaries_load_failed", error=str(exc))

    guard_state = ToolGuardState()
    invalid_tool_rounds = 0
    empty_reply_rounds = 0
    office_block_bash_probe = False
    office_block_message = ""
    office_edit_only = False
    office_edit_path = ""
    if session is not None:
        is_office_request = _is_office_edit_request(llm_message) or (
            _is_followup_edit_message(llm_message) and _has_any_last_office_path(session.metadata)
        )
        if is_office_request and _has_any_last_office_path(session.metadata):
            office_block_bash_probe = True
            office_block_message = _build_office_path_block_message(session.metadata)
            _msg_lower = llm_message.lower()
            _msg_mentions_ppt = any(
                kw in _msg_lower for kw in ("ppt", "pptx", "幻灯片")
            )
            last_pptx = session.metadata.get("last_pptx_path")
            if isinstance(last_pptx, str) and last_pptx.strip() and _msg_mentions_ppt:
                office_edit_only = True
                office_edit_path = last_pptx.strip()
    max_tool_rounds = max(1, int(agent_cfg.max_tool_rounds))

    round_idx = -1
    successful_tool_calls = 0
    _cron_reminder_notices: list[str] = []
    browser_locked_by_evomap = evo_first_mode and not evomap_hint_hit
    while total_output < _MAX_OUTPUT_TOKENS:
        round_idx += 1
        if round_idx >= max_tool_rounds:
            final_text_parts.append(
                f"工具调用轮次已达上限（{max_tool_rounds} 轮），"
                "为避免长时间卡住已暂停。请让我改用更直接的编辑方式继续。"
            )
            break
        _trim_result: TrimResult = _context_window.trim_with_metadata(
            [*system_messages, *conversation],
            model_short,
            summaries=db_summaries or None,  # pyright: ignore[reportUnknownArgumentType]
        )
        all_messages = _trim_result.messages
        if _trim_result.was_truncated and round_idx == 0:
            all_messages.append(Message(
                role="system",
                content=(
                    f"【上下文截断提示】本会话共 {_trim_result.original_count} 条消息，"
                    f"因上下文预算限制已省略 {_trim_result.dropped_count} 条较早的消息。"
                    "如需引用早期内容，请让用户重新提供关键信息。"
                ),
            ))

        _llm_t0 = _time.monotonic()
        _llm_retries = 0
        _llm_last_err: Exception | None = None
        response: AgentResponse | None = None
        while _llm_retries <= 2:
            try:
                if on_stream is None:
                    response = await router.chat(
                        model_id,
                        all_messages,
                        tools=tool_schemas or None,
                    )
                else:
                    stream_handler: StreamCallback = on_stream  # pyright: ignore[reportAssignmentType]
                    response = await router.chat(
                        model_id,
                        all_messages,
                        tools=tool_schemas or None,
                        on_stream=stream_handler,
                    )
                _llm_last_err = None
                break
            except ProviderAuthError:
                raise
            except ProviderRateLimitError:
                raise
            except ProviderError as exc:
                _llm_last_err = exc
                _llm_retries += 1
                if _llm_retries > 2:
                    # 如果已有部分成功的工具调用结果，保留进度而非直接丢弃
                    if successful_tool_calls > 0 and final_text_parts:
                        log.warning(
                            "agent.llm_exhausted_with_progress",
                            error=str(exc)[:200],
                            model=model_id,
                            successful_tools=successful_tool_calls,
                            session_id=session_id,
                        )
                        final_text_parts.append(
                            f"\n\n（模型调用出现异常: {str(exc)[:100]}，"
                            "已完成的部分结果如上。请稍后重试剩余部分。）"
                        )
                        break
                    raise
                log.warning(
                    "agent.llm_retry",
                    attempt=_llm_retries,
                    error=str(exc)[:200],
                    model=model_id,
                    session_id=session_id,
                )
                await asyncio.sleep(1.5 * _llm_retries)
        if response is None:
            if _llm_last_err is not None:
                if successful_tool_calls > 0 and final_text_parts:
                    log.warning(
                        "agent.llm_failed_with_progress",
                        error=str(_llm_last_err)[:200],
                        model=model_id,
                        successful_tools=successful_tool_calls,
                        session_id=session_id,
                    )
                    final_text_parts.append(
                        f"\n\n（模型调用失败: {str(_llm_last_err)[:100]}，"
                        "已完成的部分结果如上。请稍后重试。）"
                    )
                    break
                raise _llm_last_err
            final_text_parts.append("我这边没收到模型有效回复。请再发一次需求，我会继续处理。")
            break
        _llm_ms = int((_time.monotonic() - _llm_t0) * 1000)
        round_input = response.input_tokens
        round_output = response.output_tokens
        total_input += round_input
        total_output += round_output
        log.info(
            "agent.llm_call",
            round=round_idx,
            elapsed_ms=_llm_ms,
            model=model_id,
            round_input_tokens=round_input,
            round_output_tokens=round_output,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            trigger_event_id=trigger_event_id,
            trigger_preview=trigger_preview,
            session_id=session_id,
        )

        tool_calls = response.tool_calls
        if not tool_calls and not native_tools and response.content:
            tool_calls = _parse_fallback_tool_calls(response.content)
        if tool_calls:
            valid_tool_calls = [tc for tc in tool_calls if tc.name.strip()]
            dropped = len(tool_calls) - len(valid_tool_calls)
            if dropped > 0:
                if valid_tool_calls:
                    log.debug(
                        "agent.invalid_tool_calls_dropped",
                        dropped=dropped,
                        kept=len(valid_tool_calls),
                        session_id=session_id,
                    )
                else:
                    log.warning(
                        "agent.invalid_tool_calls_dropped",
                        dropped=dropped,
                        kept=0,
                        session_id=session_id,
                    )
            repaired_calls: list[ToolCall] = []
            for tc in valid_tool_calls:
                repaired, reason = _repair_tool_call(tc, message)
                if reason:
                    log.info(
                        "agent.tool_call_repaired",
                        tool=tc.name,
                        reason=reason,
                        before=str(tc.arguments)[:200],
                        after=str(repaired.arguments)[:200],
                        session_id=session_id,
                    )
                repaired_calls.append(repaired)
            tool_calls = repaired_calls
            blocked_reasons = blocked_tool_reasons(tool_calls, guard_state)
            if blocked_reasons:
                invalid_tool_rounds += 1
                log.warning(
                    "agent.blocked_tool_calls_dropped",
                    reasons=blocked_reasons,
                    round=round_idx,
                    session_id=session_id,
                )
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 该工具已被熔断："
                            f"{'; '.join(blocked_reasons)}。"
                            "请改用其他可用工具完成任务。"
                        ),
                    )
                )
                if invalid_tool_rounds >= MODEL_REPAIR_RETRY_LIMIT:
                    final_text_parts.append("工具调用连续无效，已停止自动重试。请明确参数后重试。")
                    break
                continue
            invalid_reasons = [
                reason
                for tc in tool_calls
                if (reason := _validate_tool_call_args(tc, registry)) is not None
            ]
            if invalid_reasons:
                invalid_tool_rounds += 1
                log.warning(
                    "agent.invalid_tool_call_args",
                    reasons=invalid_reasons,
                    round=round_idx,
                    session_id=session_id,
                )
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 你上一轮的工具调用参数无效："
                            f"{'; '.join(invalid_reasons)}。"
                            "请只重发一个有效的工具调用，必填参数必须完整且非空。"
                        ),
                    )
                )
                if invalid_tool_rounds >= MODEL_REPAIR_RETRY_LIMIT:
                    final_text_parts.append(
                        "工具调用参数连续无效，已停止自动重试。请明确参数后重试。"
                    )
                    break
                continue
            invalid_tool_rounds = 0

        content = response.content or ""
        if content:
            if guard_state.planned_image_count is None:
                update_planned_image_count(guard_state, content)
                if guard_state.planned_image_count is not None:
                    log.info(
                        "agent.planned_image_count_detected",
                        session_id=session_id,
                        planned_image_count=guard_state.planned_image_count,
                        search_images_limit=guard_state.search_images_limit,
                    )
            empty_reply_rounds = 0
            if tool_calls and not native_tools:
                clean = _strip_tool_json(content)
                if clean:
                    final_text_parts.append(clean)
            else:
                final_text_parts.append(content)

        if not tool_calls and not content.strip():
            empty_reply_rounds += 1
            log.warning(
                "agent.empty_response",
                session_id=session_id,
                round=round_idx,
                model=model_id,
                empty_reply_rounds=empty_reply_rounds,
            )
            if empty_reply_rounds == 1:
                conversation.append(
                    Message(
                        role="user",
                        content=(
                            "[系统提示] 你上一轮没有输出任何内容。"
                            "请直接给出简短回复；如果需求不明确，请先问用户要做什么。"
                        ),
                    )
                )
                continue
            final_text_parts.append("我这边没收到模型有效回复。请再发一次需求，我会继续处理。")
            break

        if not tool_calls:
            break

        if not announced_plan and on_stream:
            announced_plan = True
            has_text = content.strip() if content else ""
            if not has_text:
                tool_names = [tc.name for tc in tool_calls]
                plan = _make_plan_hint(tool_names, llm_message)
                await on_stream(plan)

        log.info(
            "agent.tool_calls",
            round=round_idx,
            count=len(tool_calls),
            tools=[tc.name for tc in tool_calls],
        )

        # Pre-flight: run evomap_fetch first; if it fails, silently remove it
        # so the LLM never sees the failed call — it just proceeds normally.
        _evomap_preflight_results: dict[str, tuple[str, ToolResult]] = {}
        _evomap_failed_ids: set[str] = set()
        for _pf_tc in tool_calls:
            if _pf_tc.name != "evomap_fetch":
                continue
            _pf_id, _pf_result = await _execute_tool(
                registry,
                _pf_tc,
                evomap_enabled=evomap_allowed_for_turn,
                browser_allowed=True,
                office_block_bash_probe=False,
                office_block_message="",
                office_edit_only=False,
                office_edit_path="",
                on_tool_call=on_tool_call,
                on_tool_result=on_tool_result,
            )
            if not _pf_result.success:
                log.info(
                    "agent.evomap_fetch_failed_fallback",
                    session_id=session_id,
                    error=_pf_result.error or _pf_result.output,
                )
                _evomap_failed_ids.add(_pf_tc.id)
            else:
                _evomap_preflight_results[_pf_tc.id] = (_pf_id, _pf_result)

        if _evomap_failed_ids:
            tool_calls = [tc for tc in tool_calls if tc.id not in _evomap_failed_ids]

        if not tool_calls:
            break

        tool_names_str = ", ".join(tc.name for tc in tool_calls)
        assistant_content = response.content or ""
        assistant_persist = (
            assistant_content
            if assistant_content
            else f"(调用工具: {tool_names_str}) {assistant_content}".strip()
        )
        assistant_msg = Message(
            role="assistant",
            content=assistant_content,
            tool_calls=tool_calls if native_tools else None,
        )
        conversation.append(assistant_msg)

        if session_manager and session:
            session_manager_obj = session_manager
            session_obj = session
            await _persist_message(session_manager_obj, session_obj, "assistant", assistant_persist)
            if assistant_content:
                for office_path in _extract_office_paths(assistant_content):
                    if _remember_office_path(session_obj.metadata, office_path):
                        metadata_dirty = True

        for _tname in ("reminder", "cron"):
            _tool = registry.get(_tname)
            if _tool is not None and hasattr(_tool, "current_session_id"):
                _tool.current_session_id = session_id  # type: ignore[union-attr]

        parallel_nano_batches: dict[int, list[ToolCall]] = {}
        batch_start: int | None = None
        batch_calls: list[ToolCall] = []
        _ph_skill_ids = locked_skill_ids or pending_lock_skill_ids or []
        _parallel_hooks: list[SkillHooks] = [
            h for s in _assembler.route_skills(llm_message, forced_skill_ids=_ph_skill_ids)
            if (h := get_skill_hooks(s)) is not None
        ] or get_all_hooks()
        for idx, candidate in enumerate(tool_calls):
            _is_parallel = any(h.is_parallelizable_bash_call(candidate) for h in _parallel_hooks)
            if _is_parallel:
                if batch_start is None:
                    batch_start = idx
                    batch_calls = [candidate]
                else:
                    batch_calls.append(candidate)
                continue
            if batch_start is not None and len(batch_calls) > 1:
                parallel_nano_batches[batch_start] = list(batch_calls)
            batch_start = None
            batch_calls = []
        if batch_start is not None and len(batch_calls) > 1:
            parallel_nano_batches[batch_start] = list(batch_calls)

        stop_for_evomap_choice = False
        stop_for_guard_abort = False
        skipped_parallel_tool_ids: set[str] = set()
        for idx, tc in enumerate(tool_calls):
            if tc.id in skipped_parallel_tool_ids:
                continue
            if tc.name == "file_write" and isinstance(tc.arguments.get("content"), str):
                for office_path in _extract_office_paths(str(tc.arguments.get("content", ""))):
                    if office_path not in pending_office_paths:
                        pending_office_paths.append(office_path)
            if session is not None and tc.name in {"ppt_edit", "docx_edit", "xlsx_edit"}:
                has_path = isinstance(tc.arguments.get("path"), str) and bool(
                    str(tc.arguments.get("path", "")).strip()
                )
                if not has_path:
                    default_path = _get_default_office_edit_path(tc.name, session.metadata)
                    if default_path:
                        fixed_args = dict(tc.arguments)
                        fixed_args["path"] = default_path
                        tc = ToolCall(id=tc.id, name=tc.name, arguments=fixed_args)
                        log.info(
                            "agent.office_path_autofill",
                            tool=tc.name,
                            path=default_path,
                            session_id=session_id,
                        )

            executed_calls: list[tuple[ToolCall, str, ToolResult]]
            if idx in parallel_nano_batches:
                batch = parallel_nano_batches[idx]
                skipped_parallel_tool_ids.update(item.id for item in batch[1:])
                log.info(
                    "agent.parallel_skill_batch",
                    round=round_idx,
                    count=len(batch),
                    session_id=session_id,
                )
                batch_results = await asyncio.gather(
                    *[
                        _execute_tool(
                            registry,
                            batch_tc,
                            evomap_enabled=evomap_allowed_for_turn,
                            browser_allowed=not browser_locked_by_evomap,
                            office_block_bash_probe=office_block_bash_probe,
                            office_block_message=office_block_message,
                            office_edit_only=office_edit_only,
                            office_edit_path=office_edit_path,
                            on_tool_call=on_tool_call,
                            on_tool_result=on_tool_result,
                        )
                        for batch_tc in batch
                    ]
                )
                executed_calls = [
                    (batch_tc, tc_id, result)
                    for batch_tc, (tc_id, result) in zip(batch, batch_results, strict=True)
                ]
            elif tc.id in _evomap_preflight_results:
                tc_id, result = _evomap_preflight_results[tc.id]
                executed_calls = [(tc, tc_id, result)]
            else:
                tc_id, result = await _execute_tool(
                    registry,
                    tc,
                    evomap_enabled=evomap_allowed_for_turn,
                    browser_allowed=not browser_locked_by_evomap,
                    office_block_bash_probe=office_block_bash_probe,
                    office_block_message=office_block_message,
                    office_edit_only=office_edit_only,
                    office_edit_path=office_edit_path,
                    on_tool_call=on_tool_call,
                    on_tool_result=on_tool_result,
                )
                executed_calls = [(tc, tc_id, result)]
            for tc, tc_id, result in executed_calls:
                if (
                    tc.name == "ppt_edit"
                    and result.success
                    and _mentions_specific_dark_bar_target(llm_message)
                ):
                    action = str(tc.arguments.get("action", "replace_text")).strip().lower()
                    if action == "apply_business_style" and "重设深色条 0 处" in (result.output or ""):
                        result = ToolResult(
                            success=False,
                            output=result.output,
                            error="未命中用户指定对象：黑色横条仍未被替换，请继续定向修改该元素",
                        )
                    elif action == "set_background":
                        result = ToolResult(
                            success=False,
                            output=result.output,
                            error="用户要求修改黑色横条，仅设置背景不算完成，请继续定向修改该横条",
                        )
                if tc.name == "evomap_fetch" and result.success:
                    candidates = _parse_evomap_fetch_candidates(result.output or "")
                    if len(candidates) > 3 and session is not None:
                        session_obj: Session = session
                        session_metadata: dict[str, Any] = session_obj.metadata
                        top3 = _pick_top_evomap_candidates(llm_message, candidates, limit=3)
                        session_metadata["evomap_pending_choices"] = {
                            "origin_message": llm_message,
                            "options": [{"asset_id": aid, "summary": summary} for aid, summary in top3],
                        }
                        if session_manager is not None:
                            await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                                session_obj.id,
                                metadata=session_metadata,
                            )
                        final_text_parts.append(_build_evomap_choice_prompt(top3))
                        stop_for_evomap_choice = True
                        break
                    browser_locked_by_evomap = not _is_no_match_evomap_output(result)
                _active_skill_hooks: list[SkillHooks] = []
                if locked_skill_ids:
                    _active_skill_hooks = [
                        hook
                        for skill in _assembler.route_skills("", forced_skill_ids=locked_skill_ids)
                        if (hook := get_skill_hooks(skill)) is not None
                    ]
                guard_update = apply_tool_result_guards(
                    guard_state,
                    tc,
                    result,
                    session_id=session_id,
                    skill_hooks=_active_skill_hooks or None,
                )
                for event in guard_update.log_events:
                    getattr(log, event.level)(event.event, **event.fields)
                for prompt in guard_update.conversation_messages:
                    conversation.append(Message(role="user", content=prompt))
                final_text_parts.extend(guard_update.final_texts)

                if result.success and result.output:
                    successful_tool_calls += 1
                    if tc.name in ("cron", "reminder") and "定时任务 +1" in result.output:
                        _cron_reminder_notices.append(result.output)
                    for path_match in re.finditer(
                        r"(/[^\s]+\.(?:jpg|jpeg|png|gif|webp))", result.output
                    ):
                        image_path = path_match.group(1)
                        real_image_paths.append(image_path)
                        if (
                            session is not None
                            and Path(image_path).expanduser().is_file()
                            and session.metadata.get("last_generated_image_path") != image_path
                        ):
                            session.metadata["last_generated_image_path"] = image_path
                            _append_image_reference_history(session.metadata, [image_path])
                            metadata_dirty = True
                elif result.success:
                    successful_tool_calls += 1
                    if session is not None:
                        for office_path in _extract_office_paths(result.output):
                            if _remember_office_path(session.metadata, office_path):
                                metadata_dirty = True
                if session is not None and pending_office_paths:
                    for office_path in list(pending_office_paths):
                        if Path(office_path).expanduser().exists():
                            if _remember_office_path(session.metadata, office_path):
                                metadata_dirty = True
                            pending_office_paths.remove(office_path)
                if (
                    session is not None
                    and tc.name == "bash"
                    and result.success
                    and _capture_latest_pptx(
                        session.metadata,
                        roots=(
                            Path("/tmp"),
                            Path("/private/tmp"),
                            Path.home() / "Downloads",
                            Path.home() / ".whaleclaw" / "workspace",
                        ),
                        window_seconds=240,
                    )
                ):
                    metadata_dirty = True

                if session is not None and tc.name == "bash" and result.success and locked_skill_ids:
                    _bash_locked_skills = _assembler.route_skills("", forced_skill_ids=locked_skill_ids)
                    for _bash_skill in _bash_locked_skills:
                        _bash_hooks = get_skill_hooks(_bash_skill)
                        if _bash_hooks is None:
                            continue
                        _bash_updates = _bash_hooks.on_bash_success(tc, result, session)
                        if _bash_updates is None:
                            continue
                        session_metadata_ref: dict[str, Any] = session.metadata
                        for _bk, _bv in _bash_updates.items():
                            if _bk == "__update_skill_param_model__":
                                skill_param_state = session_metadata_ref.get("skill_param_state")
                                if isinstance(skill_param_state, dict):
                                    sp_dict = cast(dict[object, object], skill_param_state)
                                    ns_obj = sp_dict.get(_bash_skill.id)
                                    if isinstance(ns_obj, dict):
                                        ns_obj["__model_display__"] = _bv
                            elif _bk == "__update_skill_param_mode__":
                                skill_param_state = session_metadata_ref.get("skill_param_state")
                                if isinstance(skill_param_state, dict):
                                    sp_dict = cast(dict[object, object], skill_param_state)
                                    ns_obj = sp_dict.get(_bash_skill.id)
                                    if isinstance(ns_obj, dict):
                                        ns_obj["__last_mode__"] = _bv
                            elif _bk == "__append_image_reference__":
                                if isinstance(_bv, list):
                                    _append_image_reference_history(session_metadata_ref, _bv)
                            else:
                                session_metadata_ref[_bk] = _bv
                        metadata_dirty = True

                if session is not None and tc.name in {"ppt_edit", "docx_edit", "xlsx_edit"}:
                    arg_path = tc.arguments.get("path")
                    if (
                        isinstance(arg_path, str)
                        and arg_path.strip()
                        and _remember_office_path(session.metadata, arg_path.strip())
                    ):
                        metadata_dirty = True

                # browser type 时缓存标题/正文到 skill_param_state，
                # 防止跨轮次上下文压缩后 LLM 丢失这些信息
                if (
                    session is not None
                    and tc.name == "browser"
                    and result.success
                    and locked_skill_ids
                ):
                    _br_action = str(tc.arguments.get("action", "")).strip().lower()
                    if _br_action == "type":
                        _br_selector = str(tc.arguments.get("selector", ""))
                        _br_text = str(tc.arguments.get("text", "")).strip()
                        if _br_text:
                            _sel_lower = _br_selector.lower()
                            # :not(...title...) 是"排除标题"，不应被判为标题
                            _has_title_kw = (
                                "标题" in _br_selector or "title" in _sel_lower
                            )
                            _is_negated_title = ":not(" in _sel_lower and "title" in _sel_lower
                            _is_title = _has_title_kw and not _is_negated_title
                            _is_body = not _is_title and (
                                "prosemirror" in _sel_lower
                                or "contenteditable" in _sel_lower
                                or "tiptap" in _sel_lower
                            )
                            _cache_key = (
                                "__draft_title__" if _is_title
                                else "__draft_body__" if _is_body
                                else ""
                            )
                            if _cache_key:
                                _sps = session.metadata.get("skill_param_state")
                                if isinstance(_sps, dict):
                                    for _lk_sid in locked_skill_ids:
                                        _ns = cast(dict[str, object], _sps).get(_lk_sid)
                                        if isinstance(_ns, dict):
                                            cast(dict[str, object], _ns)[_cache_key] = _br_text
                                            metadata_dirty = True
                                            log.info(
                                                "agent.browser_field_cached",
                                                skill=_lk_sid,
                                                field=_cache_key,
                                                preview=_br_text[:50],
                                                session_id=session_id,
                                            )

                tool_output = _format_tool_output(result)

                if native_tools:
                    tool_msg = Message(
                        role="tool",
                        content=tool_output,
                        tool_call_id=tc_id,
                    )
                else:
                    tool_msg = Message(
                        role="user",
                        content=(f"[工具 {tc.name} 执行结果]\n{tool_output}"),
                    )
                conversation.append(tool_msg)

                if session_manager and session and not _is_transient_cli_usage_error(result):
                    session_manager_obj = session_manager
                    session_obj = session
                    snippet = tool_output[:500] if len(tool_output) > 500 else tool_output
                    await _persist_message(
                        session_manager_obj,
                        session_obj,
                        "tool",
                        f"[{tc.name}] {snippet}",
                        tool_call_id=tc_id,
                        tool_name=tc.name,
                    )

                log.debug(
                    "agent.tool_result",
                    tool=tc.name,
                    success=result.success,
                    output_len=len(result.output),
                )
                if guard_update.stop_for_repeat_loop:
                    stop_for_guard_abort = True
                    break
            if stop_for_evomap_choice:
                break

        if stop_for_evomap_choice:
            break
        if stop_for_guard_abort:
            if metadata_dirty and session is not None and session_manager is not None:
                session_obj: Session = session
                session_manager_obj: SessionManager = session_manager
                await session_manager_obj.update_metadata(session_obj, session_obj.metadata)
                metadata_dirty = False
            break
        post_round_update = apply_post_round_guards(
            guard_state,
            tool_calls,
            round_idx=round_idx,
            session_id=session_id,
        )
        for event in post_round_update.log_events:
            getattr(log, event.level)(event.event, **event.fields)
        for prompt in post_round_update.conversation_messages:
            conversation.append(Message(role="user", content=prompt))
        final_text_parts.extend(post_round_update.final_texts)
        if metadata_dirty and session is not None and session_manager is not None:
            session_obj: Session = session
            session_manager_obj: SessionManager = session_manager
            await session_manager_obj.update_metadata(session_obj, session_obj.metadata)
            metadata_dirty = False
        if post_round_update.stop_for_repeat_loop:
            break

        final_text_parts.clear()
    else:
        log.warning(
            "agent.token_budget_exhausted",
            session_id=session_id,
            rounds=round_idx + 1,
            total_output=total_output,
        )

    final_text = "".join(final_text_parts)
    final_text = _fix_image_paths(final_text, real_image_paths)

    if _cron_reminder_notices:
        _last_notice = _cron_reminder_notices[-1]
        _count_match = re.search(r"（定时任务 \+1，合计 (\d+)）", _last_notice)
        if _count_match and f"合计 {_count_match.group(1)}" not in final_text:
            final_text = f"{final_text}\n（定时任务 +1，合计 {_count_match.group(1)}）"

    if lock_is_explicit and locked_skill_ids and skill_announce_pending:
        announce = _skill_announcement(locked_skill_ids, previous_locked_skill_ids)
        final_text = f"{announce}\n\n{final_text}" if final_text else announce
        skill_announce_pending = False
        if session is not None:
            session_obj: Session = session
            session_metadata: dict[str, Any] = session_obj.metadata
            session_metadata["skill_lock_announce_pending"] = False
            metadata_dirty = True

    _lock_confirm_tip = "回复“任务完成”以解除技能锁定；若继续修改请直接说需求。"

    # Deferred lock: first run completed with auto-routed skills -> lock them now.
    if (
        not lock_is_explicit
        and pending_lock_skill_ids
        and session is not None
        and successful_tool_calls > 0
    ):
        session_obj: Session = session
        session_metadata: dict[str, Any] = session_obj.metadata
        locked_skill_ids = pending_lock_skill_ids
        lock_is_explicit = True
        session_metadata["locked_skill_ids"] = locked_skill_ids
        session_metadata["skill_lock_waiting_done"] = True
        metadata_dirty = True
        if not _contains_lock_confirm_tip(final_text):
            final_text = f"{final_text}\n{_lock_confirm_tip}" if final_text else _lock_confirm_tip

    # Explicit lock: require user confirmation to release after successful tool use.
    elif (
        lock_is_explicit
        and locked_skill_ids
        and session is not None
        and successful_tool_calls > 0
        and not lock_waiting_done
    ):
        session_obj: Session = session
        session_metadata: dict[str, Any] = session_obj.metadata
        session_metadata["locked_skill_ids"] = locked_skill_ids
        session_metadata["skill_lock_waiting_done"] = True
        metadata_dirty = True
        if has_skill_queue and _skill_queue_has_next(skill_queue, skill_queue_index):
            if skill_queue_index < len(skill_queue):
                skill_queue[skill_queue_index]["status"] = "done"
                session_metadata["skill_queue"] = skill_queue
            _q_advance_tip = _build_skill_queue_advance_message(
                skill_queue, skill_queue_index,
            )
            final_text = f"{final_text}\n\n{_q_advance_tip}" if final_text else _q_advance_tip
        else:
            if not _contains_lock_confirm_tip(final_text):
                final_text = f"{final_text}\n{_lock_confirm_tip}" if final_text else _lock_confirm_tip

    if metadata_dirty and session is not None and session_manager is not None:
        session_obj: Session = session
        session_manager_obj: SessionManager = session_manager
        await session_manager_obj.update_metadata(session_obj, session_obj.metadata)

    # Background: generate L0/L1 summaries for older messages if needed
    if (
        session_store
        and router
        and summarizer_cfg.enabled
        and session
        and group_compressor is None
        and _compressor.should_compress(conversation_message_count)
    ):
        try:
            latest = await session_store.get_latest_summary(session_id, "L0")
            msg_rows = await session_store.get_messages(session_id)

            already_covered = latest.source_msg_end if latest else 0
            uncovered = [r for r in msg_rows if r.id > already_covered]
            protected = min(RECENT_PROTECTED, len(uncovered))
            to_compress = uncovered[:-protected] if protected < len(uncovered) else []

            if len(to_compress) >= 8:
                compress_msgs = [
                    Message(role=r.role if r.role != "tool" else "assistant", content=r.content)  # pyright: ignore[reportArgumentType]
                    for r in to_compress
                ]
                start_id = to_compress[0].id
                end_id = to_compress[-1].id
                _store_ref = session_store
                _router_ref = router
                _model_ref: str = summarizer_cfg.model

                async def _bg_compress() -> None:
                    try:
                        await _compressor.compress_segment(
                            session_id=session_id,
                            messages=compress_msgs,
                            msg_id_start=start_id,
                            msg_id_end=end_id,
                            store=_store_ref,
                            router=_router_ref,
                            model=_model_ref,
                        )
                    except Exception as exc:
                        log.debug("agent.bg_compress_failed", error=str(exc))

                asyncio.create_task(_bg_compress())
        except Exception as exc:
            log.debug("agent.bg_compress_prep_failed", error=str(exc))

    _done_rounds = max(0, round_idx + 1)
    log.info(
        "agent.done",
        model=model_id,
        llm_rounds=_done_rounds,
        input_tokens=total_input,
        output_tokens=total_output,
        session_id=session_id,
        trigger_event_id=trigger_event_id,
        trigger_preview=trigger_preview,
    )

    if on_done is not None:
        try:
            await on_done(AgentDoneInfo(
                model=model_id,
                input_tokens=total_input,
                output_tokens=total_output,
                llm_rounds=_done_rounds,
            ))
        except Exception:
            log.debug("agent.on_done_callback_failed", session_id=session_id)

    if session_store and total_input + total_output > 0:
        try:
            await session_store.record_token_usage(
                session_id=session_id,
                model=model_id,
                input_tokens=total_input,
                output_tokens=total_output,
            )
        except Exception:
            log.debug("agent.token_usage_save_failed")

    if memory_manager is not None and agent_cfg.memory.enabled:
        memory_cfg = agent_cfg.memory
        captured = False
        organized = False
        organizer_ready = True
        try:
            captured = await memory_manager.auto_capture_user_message(
                message,
                source=f"session:{session_id}",
                mode=memory_cfg.auto_capture_mode,
                cooldown_seconds=memory_cfg.cooldown_seconds,
                max_per_hour=memory_cfg.max_per_hour,
                batch_size=memory_cfg.capture_batch_size,
                merge_window_seconds=memory_cfg.capture_merge_window_seconds,
            )
            if captured:
                log.info("agent.memory_captured", session_id=session_id)
        except Exception as exc:
            log.debug("agent.memory_capture_failed", error=str(exc), session_id=session_id)
        if memory_cfg.organizer_enabled:
            try:
                router.resolve(memory_cfg.organizer_model)
            except Exception:
                organizer_ready = False
            if memory_cfg.organizer_background:
                if organizer_ready:
                    _schedule_memory_organizer_task(
                        session_id,
                        memory_manager=memory_manager,
                        router=router,
                        model_id=memory_cfg.organizer_model,
                        organizer_min_new_entries=memory_cfg.organizer_min_new_entries,
                        organizer_interval_seconds=memory_cfg.organizer_interval_seconds,
                        organizer_max_raw_window=memory_cfg.organizer_max_raw_window,
                        keep_profile_versions=memory_cfg.keep_profile_versions,
                        max_raw_entries=memory_cfg.max_raw_entries,
                    )
            else:
                try:
                    organized = await memory_manager.organize_if_needed(
                        router=router,
                        model_id=memory_cfg.organizer_model,
                        organizer_min_new_entries=memory_cfg.organizer_min_new_entries,
                        organizer_interval_seconds=memory_cfg.organizer_interval_seconds,
                        organizer_max_raw_window=memory_cfg.organizer_max_raw_window,
                        keep_profile_versions=memory_cfg.keep_profile_versions,
                        max_raw_entries=memory_cfg.max_raw_entries,
                    )
                    if organized:
                        log.info("agent.memory_organized", session_id=session_id)
                except Exception as exc:
                    organizer_ready = False
                    log.debug("agent.memory_organize_failed", error=str(exc), session_id=session_id)

        if captured and (not memory_cfg.organizer_enabled or not organizer_ready or not organized):
            try:
                updated = await memory_manager.upsert_profile_from_capture(
                    message,
                    router=router,
                    model_id=model_id,
                    max_tokens=memory_cfg.recall_profile_max_tokens,
                    keep_profile_versions=memory_cfg.keep_profile_versions,
                )
                if updated:
                    log.info("agent.memory_profile_fallback_updated", session_id=session_id)
            except Exception as exc:
                log.debug(
                    "agent.memory_profile_fallback_failed",
                    error=str(exc),
                    session_id=session_id,
                )

    final_text = _cleanup_final_reply_text(final_text)
    if locked_skill_ids:
        _pp_skills = _assembler.route_skills("", forced_skill_ids=locked_skill_ids)
        for _pp_skill in _pp_skills:
            _pp_hooks = get_skill_hooks(_pp_skill)
            if _pp_hooks is not None:
                final_text = _pp_hooks.postprocess_reply(final_text, session)
    final_text = _canonicalize_lock_confirm_tip(final_text, _lock_confirm_tip)
    return final_text
