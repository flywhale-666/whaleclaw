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
import shlex
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from whaleclaw.agent.context import OnToolCall, OnToolResult
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
from whaleclaw.agent.helpers.skill_lock import (
    build_nano_banana_execution_system_message as _build_nano_banana_execution_system_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    build_skill_lock_system_message as _build_skill_lock_system_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    build_skill_param_guard_reply as _build_skill_param_guard_reply,
)
from whaleclaw.agent.helpers.skill_lock import (
    detect_assistant_name_update as _detect_assistant_name_update,
)
from whaleclaw.agent.helpers.skill_lock import (
    detect_nano_banana_model_display as _detect_nano_banana_model_display,
)
from whaleclaw.agent.helpers.skill_lock import guarded_skills as _guarded_skills
from whaleclaw.agent.helpers.skill_lock import (
    is_nano_banana_activation_message as _is_nano_banana_activation_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    is_nano_banana_control_message as _is_nano_banana_control_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    is_task_done_confirmation as _is_task_done_confirmation,
)
from whaleclaw.agent.helpers.skill_lock import (
    load_saved_nano_banana_model_display as _load_saved_nano_banana_model_display,
)
from whaleclaw.agent.helpers.skill_lock import (
    looks_like_skill_activation_message as _looks_like_skill_activation_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    nano_banana_missing_required as _nano_banana_missing_required,
)
from whaleclaw.agent.helpers.skill_lock import normalize_for_match as _normalize_for_match
from whaleclaw.agent.helpers.skill_lock import normalize_skill_ids as _normalize_skill_ids
from whaleclaw.agent.helpers.skill_lock import parse_use_command as _parse_use_command
from whaleclaw.agent.helpers.skill_lock import preview_text as _preview_text
from whaleclaw.agent.helpers.skill_lock import (
    select_native_tool_names as _select_native_tool_names,
)
from whaleclaw.agent.helpers.skill_lock import skill_announcement as _skill_announcement
from whaleclaw.agent.helpers.skill_lock import (
    skill_explicitly_mentioned as _skill_explicitly_mentioned,
)
from whaleclaw.agent.helpers.skill_lock import (
    skill_trigger_mentioned as _skill_trigger_mentioned,
)
from whaleclaw.agent.helpers.skill_lock import update_guard_state as _update_guard_state
from whaleclaw.agent.helpers.skill_lock import (
    build_skill_queue_advance_message as _build_skill_queue_advance_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    build_skill_queue_plan_messages as _build_skill_queue_plan_messages,
)
from whaleclaw.agent.helpers.skill_lock import (
    build_skill_queue_status_message as _build_skill_queue_status_message,
)
from whaleclaw.agent.helpers.skill_lock import (
    is_queue_advance_confirmation as _is_queue_advance_confirmation,
)
from whaleclaw.agent.helpers.skill_lock import (
    parse_skill_queue_plan as _parse_skill_queue_plan,
)
from whaleclaw.agent.helpers.skill_lock import (
    skill_queue_has_next as _skill_queue_has_next,
)
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
from whaleclaw.sessions.context_window import RECENT_PROTECTED, ContextWindow
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
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

_assembler = PromptAssembler()
_context_window = ContextWindow()
_compressor = ContextCompressor()
_memory_organizer_tasks: dict[str, asyncio.Task[None]] = {}

_MAX_OUTPUT_TOKENS = 200_000
_EVOMAP_MAX_TOKENS = 1000
_EXTRA_MEMORY_COMPRESS_TIMEOUT_SECONDS = 8
_DEFAULT_ASSISTANT_NAME = "WhaleClaw"

_IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((/[^)]+)\)")
_ABS_IMAGE_PATH_RE = re.compile(
    r"(/[^\s\"')]+?\.(?:png|jpg|jpeg|gif|webp))(?=[\s\"')]|$)",
    re.IGNORECASE,
)
_IMAGE_REFERENCE_RE = re.compile(
    r"(这张图|这张图片|这幅图|这幅图片|图里|图中|参考图|按这张|基于这张|用这张)",
    re.IGNORECASE,
)
_NUMBERED_IMAGE_REFERENCE_RE = re.compile(
    r"(?:图\s*[1-9一二三四五六七八九两]|第[一二三四五六七八九两123456789]+张图)",
    re.IGNORECASE,
)
_IMAGE_EDIT_FOLLOWUP_RE = re.compile(
    r"(改(?:成|下|一下)?|修改|调整|优化|增强|变得|变成|换成|把.+(?:改|变|改成|变成|换成)|"
    r"让.+(?:改成|变成|换成)|"
    r"加上|加个|加一(?:个|只)?|加只|添加|增加|补上|再来(?:个|只)?|放一(?:个|只)?)",
    re.IGNORECASE,
)
_IMAGE_EDIT_SUBJECT_CONTINUATION_RE = re.compile(
    r"^\s*(?:请)?(?:帮我)?(?:让|给)\s*(?:这|那|它|他|她)"
    r"(?:只|个|头|名|位|条|匹|张|幅)?",
    re.IGNORECASE,
)
_IMAGE_REGENERATE_RE = re.compile(
    r"(重试|重做|重新生成|重生成|重新来|再来一版|再生成一次|再试一次|重画|这图不好看)",
    re.IGNORECASE,
)
_NANO_BANANA_RATIO_CLAUSE_RE = re.compile(
    r"[，,、;\s]*(?:图片)?(?:尺寸|比例|画幅|宽高比)"
    r"(?:改成|改为|是|为|设为|设置为|调整为|调成|[:：])?\s*"
    r"(?:\d{1,2}\s*:\s*\d{1,2}|\d{3,5}\s*[xX]\s*\d{3,5})",
    re.IGNORECASE,
)
_NANO_BANANA_REGENERATE_PREFIX_RE = re.compile(
    r"^\s*(?:请)?(?:帮我)?(?:再)?(?:重新生成|重生成|重试|再试一次|再生成一次|重新来|重做|重画)"
    r"(?:一下|下)?[，,、:\s]*",
    re.IGNORECASE,
)
_NANO_BANANA_MODEL_PREFIX_RE = re.compile(
    r"^\s*(?:用|改用|切换到)?\s*(?:香蕉2|香蕉pro)\s*",
    re.IGNORECASE,
)
_PREVIOUS_IMAGE_REF_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (2, re.compile(r"(?:再上一张|上上张|前前一张)图?", re.IGNORECASE)),
    (1, re.compile(r"(?:上一张|前一张)图?", re.IGNORECASE)),
    (0, re.compile(r"(?:当前这张|最新一张|这张图|这幅图)")),
)
_NUMBERED_IMAGE_TOKEN_RE = re.compile(
    r"(?:图\s*([1-9])|第([123456789一二三四五六七八九两])张图)",
    re.IGNORECASE,
)
_IMAGE_LOOKUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:长啥样|什么样|是哪张|发(?:来)?看|看一下|看看|给我看)"),
    re.compile(r"^(?:那)?(?:再)?(?:上[一二两]?张|前[一二两]?张|这张)呢[？?]?$"),
)
_MAX_IMAGE_REFERENCE_HISTORY = 6
_NANO_BANANA_TEXT_TO_IMAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:文生图|生图|出图|作画)", re.IGNORECASE),
    re.compile(
        r"(?:生成|做|做个|做张|画|来|给我|帮我)(?:一张|个|张)?(?:图|图片|海报|插画)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:这是一张|一张).+(?:的图|图片|海报|插画)", re.IGNORECASE),
)
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

_NANO_BANANA_NON_EXECUTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(?:你好|嗨|hi|hello|在吗|早上好|中午好|晚上好)\s*[!！,.。]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:谢谢|谢了|多谢|辛苦了|拜拜|再见|晚安)\s*[!！,.。]*\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:请)?(?:给我|给咱|给大家|来个|讲个|讲一个|说个|说一个)?"
        r"(?:笑话|段子|故事|脑筋急转弯)(?:吧|呀|啊|呗)?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:聊聊|聊一聊|继续聊|继续说|解释一下|介绍一下|分析一下|总结一下|"
        r"告诉我|回答我|帮我解释|帮我分析|帮我总结|怎么看|为什么|怎么|是什么)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:桌面截图|截个图|截图给我|屏幕截图|桌面|desktop|screenshot|截屏)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:做个PPT|制作PPT|生成PPT|幻灯片|文档|表格|excel|word|pptx?|docx?|xlsx?)",
        re.IGNORECASE,
    ),
)
_NANO_BANANA_PROMPT_QUESTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[?？]"),
    re.compile(r"(?:什么|为何|为什么|怎么|如何|哪个|哪张|哪一张|是不是|是否|吗|呢)\s*$"),
    re.compile(r"(?:什么样|长啥样|发来看看|给我看|看一下|看看)"),
)
_EVOMAP_LINE_RE = re.compile(r"^\s*-\s*([^:]+):\s*(.+?)\s*$")
_VERSION_SUFFIX_RE = re.compile(r"_V\d+$", re.IGNORECASE)
_COORDINATOR_ASK_RE = re.compile(
    r"(?:你要(?:我|什么)|需要你(?:提供|告诉|回复|回答|确认|选择)|"
    r"请(?:告诉|选择|提供|告知)我|"
    r"(?:按|用)(?:下面|以下)(?:模板|格式)(?:回|填|答))",
)
_EVOMAP_CHOICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:选|选择)?\s*([ABCabc])\s*$"),
    re.compile(r"^\s*(?:选|选择)?\s*([123])\s*$"),
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
_MULTI_AGENT_CONFIRM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(确认|确认开始|开始执行|开始多agent|开始多\s*agent|执行吧|开始吧)\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*/multi\s+go\s*$", re.IGNORECASE),
)
_MULTI_AGENT_CANCEL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(取消|取消执行|先别执行|暂停|停止)\s*$", re.IGNORECASE),
    re.compile(r"^\s*/multi\s+cancel\s*$", re.IGNORECASE),
)
_MULTI_AGENT_ROUNDS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:改为|改成|设为|设置为|设置|调整为|改到)\s*(\d{1,2})\s*轮",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*(\d{1,2})\s*轮\s*$", re.IGNORECASE),
    re.compile(r"^\s*/multi\s+rounds\s+(\d{1,2})\s*$", re.IGNORECASE),
)
_MULTI_AGENT_DISCUSS_DONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(需求确认|确认需求|需求已确认|信息齐了|进入执行确认)\s*$"),
    re.compile(r"^\s*/multi\s+ready\s*$", re.IGNORECASE),
)
_MULTI_AGENT_SCENARIO_LABELS: dict[str, str] = {
    "product_design": "产品设计",
    "content_creation": "内容创作",
    "software_development": "软件开发",
    "data_analysis_decision": "数据分析决策",
    "scientific_research": "科研",
    "intelligent_assistant": "智能助理",
    "workflow_automation": "自动化工作流",
}

# Public aliases for cross-module reuse.
MULTI_AGENT_SCENARIO_LABELS = _MULTI_AGENT_SCENARIO_LABELS
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

def _is_evomap_enabled(config: WhaleclawConfig) -> bool:
    plugins_cfg = getattr(config, "plugins", None)
    if not isinstance(plugins_cfg, dict):
        return False
    evomap_cfg_raw: object = plugins_cfg.get("evomap", None)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(evomap_cfg_raw, dict):
        return False
    evomap_cfg: dict[str, object] = evomap_cfg_raw  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    return bool(evomap_cfg.get("enabled", False))


def _is_evomap_status_question(text: str) -> bool:
    q = text.lower().strip()
    if not q:
        return False
    if "evomap" not in q and "evo map" not in q:
        return False
    status_hints = (
        "开着",
        "开启",
        "启用",
        "打开",
        "关闭",
        "状态",
        "on",
        "off",
        "enabled",
    )
    return any(h in q for h in status_hints)


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


def _dedup_consecutive_errors(
    conversation: list[Message],
    tool_name: str,
    current_error: str,
) -> None:
    """回溯 conversation，把与当前错误相似的连续历史 tool 错误替换为精简摘要。

    相似度判定：取错误文本前 120 字符比较（忽略时间戳/路径等细节差异）。
    只替换 *之前* 的消息内容，当前这条不受影响（它还没 append）。
    """
    if not conversation or not current_error:
        return
    sig = current_error[:120]
    replaced = 0
    for msg in reversed(conversation):
        content = msg.content or ""
        if msg.role == "tool" or (msg.role == "user" and content.startswith("[工具 ")):
            if "[ERROR]" not in content and "[重复错误" not in content:
                break
            prev_sig = content[:120] if msg.role == "tool" else content[content.find("\n") + 1:][:120]
            if prev_sig == sig:
                short = f"[重复错误，与下一条相同，已省略] 工具: {tool_name}"
                msg.content = short
                replaced += 1
            else:
                break
        elif msg.role == "assistant":
            continue
        else:
            break
    if replaced:
        log.debug("agent.dedup_errors", tool=tool_name, replaced=replaced)


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


def _is_tasky_message_for_evomap(text: str) -> bool:
    low = _normalize_for_match(text)
    if not low:
        return False
    keys = (
        "做",
        "制作",
        "生成",
        "写",
        "整理",
        "设计",
        "计划",
        "PPT",
        "ppt",
        "幻灯片",
        "演示文稿",
        "报告",
        "文档",
        "方案",
        "简历",
        "海报",
        "脚本",
        "代码",
        "页面",
        "表格",
        "excel",
        "xlsx",
        "word",
        "docx",
        "evomap",
        "evo map",
        "方案库",
        "协作经验",
    )
    return any(k in low for k in keys)


def _infer_task_kind(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("ppt", "幻灯片", "演示文稿", "slides", "deck")):
        return "ppt"
    if any(k in low for k in ("网页", "网站", "html", "web", "landing page", "前端")):
        return "web"
    if any(k in low for k in ("检索", "汇总", "调研", "信息", "collect", "research", "summarize")):
        return "research"
    return "general"


def _extract_topic_terms(text: str, *, limit: int = 2) -> list[str]:
    low = _normalize_for_match(text)
    stop = {
        "做",
        "制作",
        "生成",
        "创建",
        "一个",
        "关于",
        "给我",
        "帮我",
        "ppt",
        "网页",
        "网站",
        "html",
        "web",
        "方案",
        "检索",
        "汇总",
        "信息",
        "today",
        "todays",
        "today's",
    }
    terms: list[str] = []
    for t in re.findall(r"[\w\u4e00-\u9fff]{2,8}", low):
        if t in stop:
            continue
        if t not in terms:
            terms.append(t)
        if len(terms) >= max(1, limit):
            break
    return terms


def _recommended_evomap_signals(text: str) -> str:  # pyright: ignore[reportUnusedFunction]
    kind = _infer_task_kind(text)
    if kind == "ppt":
        base = [
            "ppt",
            "presentation",
            "slides",
            "storyline",
            "deck structure",
            "visual layout",
            "python-pptx",
        ]
    elif kind == "web":
        base = [
            "web page",
            "html",
            "css",
            "frontend",
            "responsive layout",
            "content structure",
        ]
    elif kind == "research":
        base = [
            "information retrieval",
            "multi-source collection",
            "source validation",
            "structured summary",
            "fact-check",
        ]
    else:
        base = [
            "workflow",
            "execution plan",
            "quality checklist",
        ]
    return ",".join(base + _extract_topic_terms(text, limit=2))


def _extra_memory_has_evomap_hint(extra_memory: str) -> bool:
    text = extra_memory.strip()
    if not text:
        return False
    return "EvoMap 协作经验候选" in text


def _is_no_match_evomap_output(result: ToolResult) -> bool:
    if not result.success:
        return False
    out = (result.output or "").strip()
    if not out:
        return True
    hints = ("未找到匹配方案", "暂无可用任务", "无已认领任务")
    return any(h in out for h in hints)


def _build_evomap_first_system_message() -> Message:
    return Message(
        role="system",
        content=(
            "执行策略：本轮是流程任务，优先复用 EvoMap 成功经验。\n"
            "1) 必须先调用 evomap_fetch 获取经验候选；\n"
            "2) 只有当 evomap_fetch 无命中或失败时，才可调用 browser；\n"
            "3) 若 evomap_fetch 命中，请先按命中方案执行。"
        ),
    )


def _extract_evomap_choice_index(text: str, options_count: int) -> int | None:
    if options_count <= 0:
        return None
    raw = text.strip()
    if not raw:
        return None
    for p in _EVOMAP_CHOICE_PATTERNS:
        m = p.match(raw)
        if not m:
            continue
        token = m.group(1).strip().upper()
        if token in {"A", "B", "C"}:
            idx = ord(token) - ord("A")
        elif token.isdigit():
            idx = int(token) - 1
        else:
            return None
        if 0 <= idx < options_count:
            return idx
    return None


def _parse_evomap_fetch_candidates(output: str) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    items: list[tuple[str, str]] = []
    for ln in lines:
        m = _EVOMAP_LINE_RE.match(ln)
        if not m:
            continue
        aid = m.group(1).strip()
        summary = m.group(2).strip()
        if not aid and not summary:
            continue
        items.append((aid, summary))
    return items


def _pick_top_evomap_candidates(
    user_message: str,
    candidates: list[tuple[str, str]],
    *,
    limit: int = 3,
) -> list[tuple[str, str]]:
    query = _normalize_for_match(user_message)
    terms = {t for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}

    scored: list[tuple[int, tuple[str, str]]] = []
    for item in candidates:
        aid, summary = item
        hay = _normalize_for_match(f"{aid} {summary}")
        score = 0
        for t in terms:
            if t in hay:
                score += 1
        scored.append((score, item))

    scored.sort(key=lambda x: (-x[0], x[1][0]))
    return [item for _score, item in scored[: max(1, limit)]]


def _build_evomap_choice_prompt(candidates: list[tuple[str, str]]) -> str:
    labels = ("A", "B", "C")
    lines = ["EvoMap 命中了多条可用方案，请先选一个我再执行："]
    for idx, item in enumerate(candidates[:3]):
        aid, summary = item
        label = labels[idx]
        lines.append(f"{label}. {aid} — {summary}")
    lines.append("请直接回复：选A / 选B / 选C")
    return "\n".join(lines)


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


def _fix_image_paths(text: str, known_paths: list[str] | None = None) -> str:
    """Validate image paths in markdown; fix fabricated paths using known real ones."""
    from pathlib import Path

    unused_real = list(known_paths or [])

    def _replace(m: re.Match[str]) -> str:
        alt, raw_path = m.group(1), m.group(2)
        fp = Path(raw_path)
        if fp.is_file():
            return m.group(0)

        # Priority 1: match from tool-returned real paths (by hash or order)
        for i, real in enumerate(unused_real):
            rp = Path(real)
            if rp.is_file():
                unused_real.pop(i)
                log.info("fix_image_path.known", original=raw_path, found=real)
                return f"![{alt}]({real})"

        # Priority 2: fuzzy match by hash suffix
        stem = fp.stem
        hash_m = re.search(r"_([0-9a-f]{6,8})$", stem)
        if hash_m and fp.parent.is_dir():
            suffix = hash_m.group(0) + fp.suffix
            for candidate in fp.parent.iterdir():
                if candidate.name.endswith(suffix) and candidate.is_file():
                    log.info("fix_image_path.fuzzy", original=raw_path, found=str(candidate))
                    return f"![{alt}]({candidate})"

        # Priority 3: most recent file in same directory
        if fp.parent.is_dir():
            files = sorted(
                (f for f in fp.parent.iterdir() if f.is_file() and f.suffix == fp.suffix),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if files:
                best = files[0]
                log.info("fix_image_path.recent", original=raw_path, found=str(best))
                return f"![{alt}]({best})"

        log.warning("fix_image_path.removed", path=raw_path)
        return f"[图片未找到: {alt}]"

    return _IMG_MD_RE.sub(_replace, text)


def _message_may_need_prior_images(message: str) -> bool:
    """Detect whether the user is referring to a previously uploaded image."""
    return bool(_IMAGE_REFERENCE_RE.search(message))


def _message_requests_image_edit(message: str) -> bool:
    """Detect edit follow-ups that should continue from the latest output image."""
    if _message_may_need_prior_images(message):
        return True
    if _NUMBERED_IMAGE_REFERENCE_RE.search(message):
        return True
    if _IMAGE_EDIT_SUBJECT_CONTINUATION_RE.search(message):
        return True
    return bool(_IMAGE_EDIT_FOLLOWUP_RE.search(message))


def _message_requests_image_regenerate(message: str) -> bool:
    """Detect reruns that should go back to the original input image set."""
    return bool(_IMAGE_REGENERATE_RE.search(message))


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
    raw_locked = metadata.get("locked_skill_ids")
    if isinstance(raw_locked, list):
        return [
            str(item).strip().lower()
            for item in raw_locked
            if isinstance(item, str) and str(item).strip()
        ]
    raw_forced = metadata.get("forced_skill_id")
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
    if "nano-banana-image-t8" in locked_skill_ids:
        model_display = _resolve_nano_banana_model_display(session)
        return (
            f"现在锁定在技能：{current_skills}。\n"
            f"当前本轮模型：{model_display}。\n"
            "要解除锁定，回复“任务完成”即可。"
        )
    return (
        f"现在锁定在技能：{current_skills}。\n"
        "要解除锁定，回复“任务完成”即可。"
    )


def _skill_requires_images(skills: list[Skill]) -> bool:
    """Return whether any active skill explicitly requires image inputs."""
    for skill in skills:
        guard = skill.param_guard
        if guard is None or not guard.enabled:
            continue
        for param in guard.params:
            if param.type.strip().lower() == "images":
                return True
    return False


def _mime_from_image_path(path: Path) -> str:
    """Infer an image mime type from the local file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def _recover_recent_session_images(
    session: Session | None,
    *,
    limit: int = 4,
) -> list[ImageContent]:
    """Reload recent local image references from prior user messages."""
    paths = _recover_recent_session_image_paths(session, limit=limit)
    recovered: list[ImageContent] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            data = path.read_bytes()
        except OSError:
            continue
        recovered.append(ImageContent(
            mime=_mime_from_image_path(path),
            data=base64.b64encode(data).decode("ascii"),
        ))
    return recovered


def _load_images_from_paths(paths: list[str]) -> list[ImageContent]:
    """Read local image paths into inline message payloads."""
    recovered: list[ImageContent] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            data = path.read_bytes()
        except OSError:
            continue
        recovered.append(ImageContent(
            mime=_mime_from_image_path(path),
            data=base64.b64encode(data).decode("ascii"),
        ))
    return recovered


def _recover_latest_generated_image(session: Session | None) -> list[ImageContent]:
    """Return only the latest generated image for edit-followup turns."""
    if session is None:
        return []
    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    latest_generated = str(metadata.get("last_generated_image_path", "")).strip()
    if not latest_generated:
        return []
    return _load_images_from_paths([latest_generated])


def _append_image_reference_history(metadata: dict[str, object], paths: list[str]) -> bool:
    raw_history = metadata.get("image_reference_history", [])
    history: list[str] = []
    if isinstance(raw_history, list):
        history = [str(item).strip() for item in raw_history if str(item).strip()]
    changed = False
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            continue
        normalized = str(path)
        if normalized in history:
            history.remove(normalized)
        history.insert(0, normalized)
        changed = True
    if len(history) > _MAX_IMAGE_REFERENCE_HISTORY:
        history = history[:_MAX_IMAGE_REFERENCE_HISTORY]
        changed = True
    if changed:
        metadata["image_reference_history"] = history
    return changed


def _recover_last_input_images(session: Session | None) -> list[ImageContent]:
    """Return the last explicit input image set for regenerate-followup turns."""
    if session is None:
        return []
    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    raw_paths = metadata.get("last_input_image_paths", [])
    if not isinstance(raw_paths, list):
        return []
    paths = [str(item).strip() for item in raw_paths if str(item).strip()]
    return _load_images_from_paths(paths)


def _recover_recent_session_image_paths(
    session: Session | None,
    *,
    limit: int = 4,
) -> list[str]:
    """Collect recent valid local image paths, preferring latest generated outputs."""
    if session is None:
        return []

    recovered: list[str] = []
    seen_paths: set[str] = set()

    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    history_raw = metadata.get("image_reference_history", [])
    if isinstance(history_raw, list):
        for item in history_raw:
            path = Path(str(item).strip()).expanduser()
            resolved = str(path)
            if resolved in seen_paths or not path.is_file():
                continue
            seen_paths.add(resolved)
            recovered.append(resolved)
            if len(recovered) >= limit:
                return recovered
    latest_generated = str(metadata.get("last_generated_image_path", "")).strip()
    if latest_generated:
        path = Path(latest_generated).expanduser()
        if path.is_file():
            seen_paths.add(str(path))
            recovered.append(str(path))
            if len(recovered) >= limit:
                return recovered

    for msg in reversed(session.messages):
        if msg.role not in {"user", "assistant"} or not msg.content:
            continue
        markdown_paths = [match.group(2).strip() for match in _IMG_MD_RE.finditer(msg.content)]
        plain_paths = [match.group(1).strip() for match in _ABS_IMAGE_PATH_RE.finditer(msg.content)]
        for raw_path in [*markdown_paths, *plain_paths]:
            path = Path(raw_path).expanduser()
            resolved = str(path)
            if resolved in seen_paths or not path.is_file():
                continue
            seen_paths.add(resolved)
            recovered.append(resolved)
            if len(recovered) >= limit:
                return recovered
    return recovered


def _resolve_relative_image_reference_index(message: str) -> int | None:
    for index, pattern in _PREVIOUS_IMAGE_REF_PATTERNS:
        if pattern.search(message):
            return index
    return None


def _number_token_to_index(token: str) -> int | None:
    normalized = token.strip()
    if not normalized:
        return None
    if normalized.isdigit():
        value = int(normalized)
        return value - 1 if value > 0 else None
    zh_map = {
        "一": 0,
        "二": 1,
        "两": 1,
        "三": 2,
        "四": 3,
        "五": 4,
        "六": 5,
        "七": 6,
        "八": 7,
        "九": 8,
    }
    return zh_map.get(normalized)


def _extract_numbered_image_reference_indexes(message: str) -> list[int]:
    indexes: list[int] = []
    seen: set[int] = set()
    for match in _NUMBERED_IMAGE_TOKEN_RE.finditer(message):
        token = match.group(1) or match.group(2) or ""
        index = _number_token_to_index(token)
        if index is None or index in seen:
            continue
        seen.add(index)
        indexes.append(index)
    return indexes


def _resolve_numbered_input_image_paths(
    session: Session | None,
    message: str,
) -> list[str]:
    indexes = _extract_numbered_image_reference_indexes(message)
    if not indexes:
        return []
    candidates = _recover_last_input_image_paths(session)
    if not candidates:
        return []
    resolved: list[str] = []
    for index in indexes:
        if 0 <= index < len(candidates):
            resolved.append(candidates[index])
    return resolved


def _resolve_relative_image_reference_path(
    session: Session | None,
    message: str,
) -> str:
    ref_index = _resolve_relative_image_reference_index(message)
    if ref_index is None:
        return ""
    history = _recover_recent_session_image_paths(session, limit=max(4, ref_index + 1))
    if ref_index >= len(history):
        return ""
    return history[ref_index]


def _is_image_reference_lookup_message(message: str) -> bool:
    if _is_nano_banana_execution_request(message=message, has_new_input_images=False):
        return False
    if _resolve_relative_image_reference_index(message) is None:
        return False
    return any(pattern.search(message) for pattern in _IMAGE_LOOKUP_PATTERNS)


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


def _is_numbered_image_reference_edit_message(text: str) -> bool:
    """Return whether the message is describing an edit across numbered image refs."""
    if not _NUMBERED_IMAGE_REFERENCE_RE.search(text):
        return False
    return _message_requests_image_edit(text)


def _lockable_skill_ids(skills: list[Skill]) -> list[str]:
    return _normalize_skill_ids([
        skill for skill in skills
        if skill.lock_session and skill.is_user_installed
    ])


def _looks_like_nano_banana_prompt_text(message: str) -> bool:
    stripped = message.strip()
    if not stripped:
        return False
    if any(pattern.search(stripped) for pattern in _NANO_BANANA_PROMPT_QUESTION_PATTERNS):
        return False
    if any(pattern.search(stripped) for pattern in _NANO_BANANA_NON_EXECUTION_PATTERNS):
        return False
    compact = re.sub(r"[\s，,。.!！;；:：、】【（）()\"'“”‘’]+", "", stripped)
    if len(compact) < 6:
        return False
    return not bool(
        re.search(r"(?:技能|skill|模型|api|key|路径|网址|链接|PPT|ppt|文档|表格)", stripped)
    )


def _extract_input_image_paths_from_text(
    text: str,
    *,
    limit: int = 8,
) -> list[str]:
    """Extract unique local image paths from the current user message text."""
    extracted: list[str] = []
    seen_paths: set[str] = set()
    markdown_paths = [match.group(2).strip() for match in _IMG_MD_RE.finditer(text)]
    plain_paths = [match.group(1).strip() for match in _ABS_IMAGE_PATH_RE.finditer(text)]
    for raw_path in [*markdown_paths, *plain_paths]:
        path = Path(raw_path).expanduser()
        resolved = str(path)
        if resolved in seen_paths or not path.is_file():
            continue
        seen_paths.add(resolved)
        extracted.append(resolved)
        if len(extracted) >= limit:
            break
    return extracted


def _strip_inline_image_markdown(text: str) -> str:
    """Remove appended local image markdown from a user prompt string."""
    stripped = _IMG_MD_RE.sub("", text)
    stripped = _ABS_IMAGE_PATH_RE.sub("", stripped)
    stripped = stripped.replace("(用户发送了图片)", "")
    return stripped.strip()


def _clean_nano_banana_prompt_delta(message: str) -> str:
    cleaned = _strip_inline_image_markdown(message).strip()
    if not cleaned:
        return ""
    cleaned = _NANO_BANANA_MODEL_PREFIX_RE.sub("", cleaned)
    cleaned = _NANO_BANANA_REGENERATE_PREFIX_RE.sub("", cleaned)
    cleaned = _NANO_BANANA_RATIO_CLAUSE_RE.sub("", cleaned)
    cleaned = re.sub(r"^[，,、；;：:\s]+", "", cleaned)
    cleaned = re.sub(r"[，,、；;：:\s]+$", "", cleaned)
    return cleaned.strip()


def _merge_nano_banana_prompt(
    *,
    previous_prompt: str,
    message: str,
    regenerate: bool,
    image_edit: bool,
) -> str:
    previous = _strip_inline_image_markdown(previous_prompt).strip()
    delta = _clean_nano_banana_prompt_delta(message)
    if not previous:
        return delta
    if not delta:
        return previous
    if delta in previous:
        return previous
    if regenerate or image_edit:
        return (
            f"{previous}\n\n"
            "在保留原始主题、主体和核心场景设定的前提下，按以下要求重新生成或修改："
            f"{delta}"
        )
    return delta


def _recover_latest_generated_image_path(session: Session | None) -> str:
    if session is None:
        return ""
    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    latest_generated = str(metadata.get("last_generated_image_path", "")).strip()
    path = Path(latest_generated).expanduser()
    if latest_generated and path.is_file():
        return str(path)
    return ""


def _recover_last_input_image_paths(session: Session | None) -> list[str]:
    if session is None:
        return []
    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    raw_paths = metadata.get("last_input_image_paths", [])
    if not isinstance(raw_paths, list):
        return []
    output: list[str] = []
    for item in raw_paths:
        path = Path(str(item).strip()).expanduser()
        if path.is_file():
            output.append(str(path))
    return output


def _recover_last_nano_banana_mode(session: Session | None) -> str:
    if session is None:
        return ""
    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    raw_mode = str(metadata.get("last_nano_banana_mode", "")).strip().lower()
    if raw_mode in {"text", "edit"}:
        return raw_mode
    state_map_raw = metadata.get("skill_param_state", {})
    if not isinstance(state_map_raw, dict):
        return ""
    nano_state_raw = state_map_raw.get("nano-banana-image-t8", {})
    if not isinstance(nano_state_raw, dict):
        return ""
    state_mode = str(nano_state_raw.get("__last_mode__", "")).strip().lower()
    if state_mode in {"text", "edit"}:
        return state_mode
    return ""


def _resolve_nano_banana_model_display(session: Session | None) -> str:
    default_model = _load_saved_nano_banana_model_display()
    if session is None:
        return default_model
    metadata = session.metadata if isinstance(session.metadata, dict) else {}
    recent_model = str(metadata.get("last_nano_banana_model_display", "")).strip()
    if recent_model in {"香蕉2", "香蕉pro"}:
        return recent_model
    state_map_raw = metadata.get("skill_param_state", {})
    if not isinstance(state_map_raw, dict):
        return default_model
    nano_state_raw = state_map_raw.get("nano-banana-image-t8", {})
    if not isinstance(nano_state_raw, dict):
        return default_model
    state_model = str(nano_state_raw.get("__model_display__", "")).strip()
    if state_model in {"香蕉2", "香蕉pro"}:
        return state_model
    return default_model


def _resolve_nano_banana_input_paths(
    llm_message: str,
    session: Session | None,
) -> list[str]:
    """Resolve the input image paths for a nano-banana execution turn."""
    explicit_paths = _extract_input_image_paths_from_text(llm_message)
    if explicit_paths:
        return explicit_paths
    # 用户明确要求文生图时，直接返回空（text 模式），
    # 不让后面的 edit 误判覆盖意图。
    if any(p.search(llm_message) for p in _NANO_BANANA_TEXT_TO_IMAGE_PATTERNS):
        return []
    numbered_paths = _resolve_numbered_input_image_paths(session, llm_message)
    if numbered_paths:
        return numbered_paths
    relative_path = _resolve_relative_image_reference_path(session, llm_message)
    if relative_path:
        return [relative_path]
    if _message_requests_image_regenerate(llm_message):
        if _recover_last_nano_banana_mode(session) == "text":
            return []
        return _recover_last_input_image_paths(session)
    if _message_requests_image_edit(llm_message):
        latest_generated = _recover_latest_generated_image_path(session)
        if latest_generated:
            return [latest_generated]
        return _recover_last_input_image_paths(session)
    return []


def _is_nano_banana_execution_request(
    *,
    message: str,
    has_new_input_images: bool,
) -> bool:
    stripped = message.strip()
    if not stripped:
        return has_new_input_images
    if has_new_input_images:
        return True
    if _is_nano_banana_control_message(stripped):
        return False
    if _is_task_done_confirmation(stripped, task_done_patterns=_TASK_DONE_PATTERNS):
        return False
    if any(pattern.search(stripped) for pattern in _NANO_BANANA_NON_EXECUTION_PATTERNS):
        return False
    if (
        any(pattern.search(stripped) for pattern in _NANO_BANANA_PROMPT_QUESTION_PATTERNS)
        and not _message_requests_image_regenerate(stripped)
        and not any(pattern.search(stripped) for pattern in _NANO_BANANA_TEXT_TO_IMAGE_PATTERNS)
        and not _IMAGE_EDIT_FOLLOWUP_RE.search(stripped)
    ):
        return False
    if _message_requests_image_regenerate(stripped):
        return True
    if _message_requests_image_edit(stripped):
        return True
    if any(pattern.search(stripped) for pattern in _NANO_BANANA_TEXT_TO_IMAGE_PATTERNS):
        return True
    if _is_image_generation_request(stripped):
        return True
    if _looks_like_nano_banana_prompt_text(stripped):
        return True
    return True


def _build_nano_banana_command(
    *,
    mode: str,
    model_display: str,
    prompt: str,
    input_paths: list[str],
    ratio: str,
) -> str:
    """Build the fixed nano-banana script command."""
    script_path = (
        Path.home()
        / ".whaleclaw"
        / "workspace"
        / "skills"
        / "nano-banana-image-t8"
        / "scripts"
        / "test_nano_banana_2.py"
    )
    parts = [
        "./python/bin/python3.12",
        shlex.quote(str(script_path)),
        "--mode",
        shlex.quote(mode),
        "--model",
        shlex.quote(model_display),
        "--edit-model",
        shlex.quote(model_display),
        "--prompt",
        shlex.quote(prompt),
        "--aspect-ratio",
        shlex.quote(ratio or "auto"),
    ]
    if model_display == "香蕉pro":
        parts.extend(["--image-size", "2K"])
    for path in input_paths:
        parts.extend(["--input-image", shlex.quote(path)])
    return " ".join(parts)


def _parse_nano_banana_result_path(output: str) -> str:
    for pattern in (
        r"图生图成功:\s*(/[^\s]+)",
        r"文生图成功:\s*(/[^\s]+)",
        r"(/[^\s]+\.(?:png|jpg|jpeg|webp|gif))",
    ):
        match = re.search(pattern, output)
        if match:
            return match.group(1).strip()
    return ""


def _format_nano_banana_success_reply(
    *,
    model_display: str,
    image_path: str,
    regenerate: bool,
) -> str:
    lead = "重新生成好了：" if regenerate else "改好了："
    lines = [f"当前使用模型：{model_display}", lead]
    if image_path:
        lines.append(f"![结果图]({image_path})")
        lines.append(f"文件路径：{image_path}")
    return "\n".join(lines)


def _multi_agent_cfg(config: WhaleclawConfig) -> dict[str, object]:
    plugins_raw = config.plugins
    plugins: dict[str, object] = plugins_raw if isinstance(plugins_raw, dict) else {}  # pyright: ignore[reportAssignmentType, reportUnnecessaryIsInstance]
    raw_ma = plugins.get("multi_agent", {})
    if not isinstance(raw_ma, dict):
        return {"enabled": False, "mode": "parallel", "max_rounds": 1, "roles": []}

    raw: dict[str, object] = raw_ma  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    enabled = bool(raw.get("enabled", False))
    mode_raw = str(raw.get("mode", "parallel")).strip().lower()
    mode = mode_raw if mode_raw in {"parallel", "serial"} else "parallel"

    try:
        max_rounds = int(raw.get("max_rounds", 1))  # pyright: ignore[reportArgumentType]
    except Exception:
        max_rounds = 1
    max_rounds = max(1, min(max_rounds, 10))

    roles_raw = raw.get("roles")
    roles: list[dict[str, object]] = []
    if isinstance(roles_raw, list):
        for idx, item in enumerate(roles_raw[:20], start=1):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            if not isinstance(item, dict):
                continue
            role_item: dict[str, object] = item  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
            rid = str(role_item.get("id", f"role_{idx}")).strip().lower()
            rid = "".join(ch for ch in rid if ch.isalnum() or ch in {"_", "-"})
            if not rid:
                rid = f"role_{idx}"
            name = str(role_item.get("name", f"角色{idx}")).strip() or f"角色{idx}"
            model = str(role_item.get("model", "")).strip()
            system_prompt = str(role_item.get("system_prompt", "")).strip()
            roles.append(
                {
                    "id": rid[:64],
                    "name": name[:50],
                    "enabled": bool(role_item.get("enabled", True)),
                    "model": model[:100],
                    "system_prompt": system_prompt[:3000],
                }
            )
    return {
        "enabled": enabled,
        "mode": mode,
        "max_rounds": max_rounds,
        "roles": roles,
}


def _scenario_discuss_focus(scenario: str) -> str:
    if scenario == "product_design":
        return (
            "重点和用户确认：产品目标、目标用户、核心场景、关键流程、约束条件、"
            "以及交付物类型（如 PRD 文档、流程图图片、原型说明、里程碑计划）。"
        )
    if scenario == "content_creation":
        return (
            "重点和用户确认：受众人群、内容主题、语气风格、发布渠道、篇幅限制、"
            "素材来源与交付物类型（文章/脚本/海报文案/配图说明）。"
        )
    if scenario == "software_development":
        return (
            "重点和用户确认：功能目标、技术栈、运行环境、改动范围、验收标准、"
            "交付物类型（代码补丁、命令步骤、测试报告、部署说明）。"
        )
    if scenario == "data_analysis_decision":
        return (
            "重点和用户确认：决策问题、指标口径、数据来源、时间窗口、可信度要求、"
            "交付物类型（分析报告、图表、结论摘要、决策建议表）。"
        )
    if scenario == "scientific_research":
        return (
            "重点和用户确认：研究问题、假设、实验条件、对照设计、评估指标、"
            "交付物类型（研究提纲、实验方案、结果解读、论文结构草稿）。"
        )
    if scenario == "intelligent_assistant":
        return (
            "重点和用户确认：任务目标、时效要求、可调用工具、执行边界、"
            "交付物类型（行动计划、提醒清单、消息草稿、执行结果汇总）。"
        )
    if scenario == "workflow_automation":
        return (
            "重点和用户确认：触发条件、上下游系统、字段映射、失败重试、监控告警、"
            "交付物类型（流程设计文档、自动化脚本、运行手册、告警规则清单）。"
        )
    return "重点和用户确认目标、约束、验收标准与最终交付物类型。"


def _scenario_delivery_focus(scenario: str) -> str:  # pyright: ignore[reportUnusedFunction]
    if scenario == "product_design":
        return (
            "最终答复必须包含：\n"
            "1) 产品方案摘要\n"
            "2) 结构化交付物清单（文档/图片/表格）\n"
            "3) 每个交付物的建议文件名与路径（例如 /tmp/product_prd.md）\n"
            "4) 执行优先级与里程碑\n"
            "5) 风险与备选方案"
        )
    if scenario == "content_creation":
        return (
            "最终答复必须包含：\n"
            "1) 内容策略与目标受众\n"
            "2) 成品文案或脚本草案\n"
            "3) 渠道适配版本（至少 2 个）\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/content_plan.md）\n"
            "5) 发布节奏与复盘指标"
        )
    if scenario == "software_development":
        return (
            "最终答复必须包含：\n"
            "1) 技术方案与实现路径\n"
            "2) 关键代码改动点或命令步骤\n"
            "3) 测试与回归计划\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/impl_plan.md）\n"
            "5) 风险、回滚与上线注意项"
        )
    if scenario == "data_analysis_decision":
        return (
            "最终答复必须包含：\n"
            "1) 数据结论与关键洞察\n"
            "2) 指标口径说明与分析过程摘要\n"
            "3) 决策选项对比与推荐方案\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/analysis_report.md）\n"
            "5) 风险假设与后续验证计划"
        )
    if scenario == "scientific_research":
        return (
            "最终答复必须包含：\n"
            "1) 研究目标与假设\n"
            "2) 方法设计与实验步骤\n"
            "3) 结果解读框架与可信度边界\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/research_plan.md）\n"
            "5) 下一步实验与论文化建议"
        )
    if scenario == "intelligent_assistant":
        return (
            "最终答复必须包含：\n"
            "1) 任务拆解与优先级\n"
            "2) 可执行动作清单\n"
            "3) 关键提醒与时间节点\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/assistant_actions.md）\n"
            "5) 异常处理与后续跟进建议"
        )
    if scenario == "workflow_automation":
        return (
            "最终答复必须包含：\n"
            "1) 自动化流程设计（触发-处理-输出）\n"
            "2) 集成接口与字段映射说明\n"
            "3) 失败重试与告警策略\n"
            "4) 交付物清单与建议文件名/路径（如 /tmp/workflow_spec.md）\n"
            "5) 上线运行与运维检查清单"
        )
    return (
        "最终答复必须包含：结论、可执行清单、交付物清单（含建议文件名/路径）、"
        "风险与回滚建议。"
    )


def _resolve_multi_agent_cfg(
    config: WhaleclawConfig,
    session: Session | None,
) -> dict[str, object]:
    """Build effective multi-agent config with optional session overrides."""
    cfg = _multi_agent_cfg(config)
    plugins_raw2 = config.plugins
    plugins2: dict[str, object] = plugins_raw2 if isinstance(plugins_raw2, dict) else {}  # pyright: ignore[reportAssignmentType, reportUnnecessaryIsInstance]
    raw = plugins2.get("multi_agent", {})
    if isinstance(raw, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        raw_typed: dict[str, object] = raw  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
        scenario = str(raw_typed.get("scenario", "software_development")).strip()
        cfg["scenario"] = scenario or "software_development"
    else:
        cfg["scenario"] = "software_development"
    if session is None or not isinstance(session.metadata, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return cfg

    metadata = session.metadata
    if isinstance(metadata.get("multi_agent_enabled"), bool):
        cfg["enabled"] = bool(metadata["multi_agent_enabled"])

    mode_raw = str(metadata.get("multi_agent_mode", "")).strip().lower()
    if mode_raw in {"parallel", "serial"}:
        cfg["mode"] = mode_raw

    rounds_raw = metadata.get("multi_agent_max_rounds")
    if isinstance(rounds_raw, int):
        cfg["max_rounds"] = max(1, min(rounds_raw, 10))

    return cfg


def _is_multi_agent_confirm(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _MULTI_AGENT_CONFIRM_PATTERNS)


def _is_multi_agent_cancel(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _MULTI_AGENT_CANCEL_PATTERNS)


def _extract_multi_agent_rounds(text: str) -> int | None:
    t = text.strip()
    if not t:
        return None
    for p in _MULTI_AGENT_ROUNDS_PATTERNS:
        m = p.search(t)
        if not m:
            continue
        try:
            value = int(m.group(1))
        except Exception:
            return None
        if 1 <= value <= 10:
            return value
    return None


def _attach_rounds_marker(topic: str, rounds: int) -> str:
    clean = re.sub(r"\[MA_ROUNDS=\d{1,2}\]\s*", "", topic).strip()
    return f"[MA_ROUNDS={rounds}] {clean}".strip()


def _extract_rounds_marker(topic: str) -> tuple[str, int | None]:
    m = re.search(r"\[MA_ROUNDS=(\d{1,2})\]", topic)
    if not m:
        return (topic, None)
    try:
        value = int(m.group(1))
    except Exception:
        value = None
    clean = re.sub(r"\[MA_ROUNDS=\d{1,2}\]\s*", "", topic).strip()
    if value is None or not (1 <= value <= 10):
        return (clean, None)
    return (clean, value)


def _is_multi_agent_discuss_done(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    return any(p.search(t) for p in _MULTI_AGENT_DISCUSS_DONE_PATTERNS)


def _format_multi_agent_preflight_text(  # pyright: ignore[reportUnusedFunction]
    *,
    cfg: dict[str, object],
    topic: str,
) -> str:
    roles = [
        role
        for role in cast(list[dict[str, object]], cfg["roles"])
        if bool(role.get("enabled", True))
    ]
    mode = cast(str, cfg["mode"])
    rounds = cast(int, cfg["max_rounds"])
    mode_cn = "并行" if mode == "parallel" else "串行"
    lines = [
        "已进入多Agent准备阶段（尚未开始执行）。",
        f"- 当前模式: {mode_cn}（{mode}）",
        f"- 计划回合: {rounds}",
        f"- 角色数量: {len(roles)}",
        "",
        "角色分工:",
    ]
    for role in roles:
        name = str(role.get("name", role.get("id", "角色"))).strip() or "角色"
        duty = _multi_agent_system_prompt(role)
        lines.append(f"- {name}: {_compact_role_output(duty, 120)}")
    lines.extend(
        [
            "",
            "主控建议:",
            "- 先确认目标、交付形式、截止时间与约束。",
            "- 若希望更快出结论，可先改为 2 轮；若任务复杂建议 4 轮以上。",
            "",
            f"当前议题: {topic.strip() or '(未提供)'}",
            "",
            "回复以下任一指令继续:",
            "- 回复“确认开始”：按当前配置启动多Agent执行",
            "- 回复“改为N轮”：修改回合后继续等待确认",
            "- 回复“取消”：退出本次多Agent执行",
        ]
    )
    return "\n".join(lines)


def _multi_agent_module():
    from whaleclaw.agent import multi_agent

    return multi_agent


def multi_agent_system_prompt(role: dict[str, object]) -> str:
    return _multi_agent_module().multi_agent_system_prompt(role)


def compact_role_output(text: str, max_chars: int = 600) -> str:
    return _multi_agent_module().compact_role_output(text, max_chars)


def looks_like_bad_coordinator_output(text: str) -> bool:
    return _multi_agent_module().looks_like_bad_coordinator_output(text)


def looks_like_role_stall_output(text: str) -> bool:
    return _multi_agent_module().looks_like_role_stall_output(text)


def need_image_output(user_message: str) -> bool:
    return _multi_agent_module().need_image_output(user_message)


def extract_requested_deliverables(user_message: str) -> list[str]:
    return _multi_agent_module().extract_requested_deliverables(user_message)


def build_multi_agent_requirement_baseline(
    *,
    message: str,
    scenario: str,
    mode: str,
    max_rounds: int,
    requested_deliverables: list[str],
) -> str:
    return _multi_agent_module().build_multi_agent_requirement_baseline(
        message=message,
        scenario=scenario,
        mode=mode,
        max_rounds=max_rounds,
        requested_deliverables=requested_deliverables,
    )


_multi_agent_system_prompt = multi_agent_system_prompt
_compact_role_output = compact_role_output
_looks_like_bad_coordinator_output = looks_like_bad_coordinator_output
_looks_like_role_stall_output = looks_like_role_stall_output
_need_image_output = need_image_output
_extract_requested_deliverables = extract_requested_deliverables
_build_multi_agent_requirement_baseline = build_multi_agent_requirement_baseline


scenario_discuss_focus = _scenario_discuss_focus
truncate_to_tokens = _truncate_to_tokens
resolve_multi_agent_cfg = _resolve_multi_agent_cfg
is_multi_agent_confirm = _is_multi_agent_confirm
extract_multi_agent_rounds = _extract_multi_agent_rounds
is_multi_agent_discuss_done = _is_multi_agent_discuss_done
select_native_tool_names = _select_native_tool_names
extract_round_delivery_section = _extract_round_delivery_section
extract_delivery_artifact_paths = _extract_delivery_artifact_paths
fix_version_suffix = _fix_version_suffix
snapshot_round_artifacts = _snapshot_round_artifacts
extract_artifact_baseline = _extract_artifact_baseline


async def _persist_session_metadata(
    session: Session | None,
    session_manager: SessionManager | None,
) -> bool:
    if session is None or session_manager is None:
        return False
    try:
        await session_manager.update_metadata(session, session.metadata)
    except Exception:
        return False
    return True


async def _sync_multi_agent_compression_boundary(
    session: Session | None,
    session_manager: SessionManager | None,
    group_compressor: "SessionGroupCompressor | None" = None,
    *,
    ma_enabled: bool,
) -> None:
    """Track MA on/off transition and set compression boundary on MA -> single.

    When MA is enabled, mark current session as MA-active.
    When MA turns off, record a fixed message-index boundary so future
    group-compression only applies to newly produced messages.
    """
    if session is None or not isinstance(session.metadata, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return None
    metadata = session.metadata
    prev_active = bool(metadata.get("multi_agent_active_prev", False))
    changed = False
    if ma_enabled:
        if not prev_active:
            metadata["multi_agent_active_prev"] = True
            changed = True
    else:
        if prev_active:
            metadata["multi_agent_active_prev"] = False
            metadata["compression_resume_message_index"] = len(session.messages)
            changed = True
    if group_compressor is not None:  # session already checked above
        try:
            await group_compressor.set_session_suspended(
                session_id=session.id,
                suspended=ma_enabled,
            )
        except Exception as exc:
            log.debug(
                "agent.multi_agent_compressor_toggle_failed",
                session_id=session.id,
                error=str(exc),
                enabled=ma_enabled,
            )
    if not changed or session_manager is None:
        return None
    try:
        await asyncio.wait_for(
            session_manager.update_metadata(session, session.metadata),
            timeout=1.5,
        )
        return None
    except Exception as exc:
        log.warning(
            "agent.multi_agent_metadata_persist_failed",
            session_id=session.id,
            error=str(exc),
        )
        return None


async def _run_multi_agent_controller_discussion(
    *,
    user_message: str,
    pending_topic: str,
    cfg: dict[str, object],
    session_id: str,
    config: WhaleclawConfig,
    router: ModelRouter,
    registry: ToolRegistry,
    extra_memory: str,
    trigger_event_id: str,
    trigger_text_preview: str,
    include_intro: bool,
) -> str:
    from whaleclaw.agent import multi_agent as _multi_agent

    return await _multi_agent.run_multi_agent_controller_discussion(
        user_message=user_message,
        pending_topic=pending_topic,
        cfg=cfg,
        session_id=session_id,
        config=config,
        router=router,
        registry=registry,
        extra_memory=extra_memory,
        trigger_event_id=trigger_event_id,
        trigger_text_preview=trigger_text_preview,
        include_intro=include_intro,
    )


async def _run_multi_agent_executor(
    *,
    message: str,
    session_id: str,
    config: WhaleclawConfig,
    on_stream: StreamCallback | None,
    router: ModelRouter,
    registry: ToolRegistry,
    images: list[ImageContent] | None,
    extra_memory: str,
    trigger_event_id: str,
    trigger_text_preview: str,
    ma_cfg: dict[str, object],
    on_round_result: OnRoundResult | None = None,
) -> str:
    from whaleclaw.agent import multi_agent as _multi_agent

    return await _multi_agent.run_multi_agent_executor(
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
    nano_banana_recommended_command: str = ""
    routed_skills: list[Skill] = []
    routed_skill_ids: list[str] = []
    if not multi_agent_internal:
        if session is not None:
            raw_queue = session.metadata.get("skill_queue")
            if isinstance(raw_queue, list) and raw_queue:
                skill_queue = [
                    {str(k): str(v) for k, v in item.items() if isinstance(k, str)}
                    for item in raw_queue
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

        if (
            lock_is_explicit
            and "nano-banana-image-t8" in locked_skill_ids
            and _is_nano_banana_activation_message(message)
        ):
            model_display = _resolve_nano_banana_model_display(session)
            return (
                "当前会话仍在香蕉生图技能里。"
                f"当前模型：{model_display}。\n"
                "如果要继续生图，请直接发送提示词或图片；"
                "如果本轮已结束，请回复“任务完成”解除技能锁定。"
            )
        if (
            lock_is_explicit
            and "nano-banana-image-t8" in locked_skill_ids
            and _is_image_reference_lookup_message(llm_message)
        ):
            ref_path = _resolve_relative_image_reference_path(session, llm_message)
            if ref_path:
                return f"你说的这张是：\n路径：{ref_path}\n\n![历史图片]({ref_path})"

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
                    else:
                        session.metadata.pop("locked_skill_ids", None)
                        session.metadata.pop("skill_lock_waiting_done", None)
                        session.metadata.pop("skill_lock_announce_pending", None)
                        session.metadata.pop("skill_queue", None)
                        session.metadata.pop("skill_queue_index", None)
                        session.metadata.pop("skill_queue_original_message", None)
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
        elif lock_waiting_done:
            lock_waiting_done = False
            skill_lock_resumed_from_waiting = True
            if session is not None:
                session.metadata["skill_lock_waiting_done"] = False
                metadata_dirty = True

        routed_skills = _assembler.route_skills(llm_message)
        routed_skill_ids = _normalize_skill_ids(routed_skills)
        lockable_routed_skill_ids = _lockable_skill_ids(routed_skills)

        _is_compound_with_other_steps = (
            routed_skill_ids
            and any(_skill_trigger_mentioned(skill, message) for skill in routed_skills)
            and not _looks_like_skill_activation_message(
                message, skill_activation_patterns=_SKILL_ACTIVATION_PATTERNS,
            )
            and _NON_SKILL_STEP_RE.search(message) is not None
        )

        if (
            not locked_skill_ids
            and routed_skill_ids
            and not _is_compound_with_other_steps
            and (
                _looks_like_skill_activation_message(
                    message,
                    skill_activation_patterns=_SKILL_ACTIVATION_PATTERNS,
                )
                or any(_skill_trigger_mentioned(skill, message) for skill in routed_skills)
            )
        ):
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
                    metadata_dirty = True
            else:
                locked_skill_ids = lockable_routed_skill_ids
                lock_is_explicit = bool(lockable_routed_skill_ids)
                lock_waiting_done = False
                skill_announce_pending = bool(lockable_routed_skill_ids)
                if session is not None and lockable_routed_skill_ids:
                    session.metadata["locked_skill_ids"] = locked_skill_ids
                    session.metadata["skill_lock_waiting_done"] = False
                    session.metadata["skill_lock_announce_pending"] = True
                    metadata_dirty = True
        elif not locked_skill_ids and lockable_routed_skill_ids:
            pending_lock_skill_ids = lockable_routed_skill_ids

        if (
            lock_is_explicit
            and locked_skill_ids
            and routed_skill_ids
            and routed_skill_ids != locked_skill_ids
        ):
            if (
                "nano-banana-image-t8" in locked_skill_ids
                and (
                    _is_numbered_image_reference_edit_message(message)
                    or _message_requests_image_regenerate(message)
                )
            ):
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
                    return (
                        f"当前会话仍锁定在 {current_names} 技能。"
                        f"如果你确实要切换到 {requested_names}，请先回复“任务完成”，"
                        "之后再重新输入目标命令。"
                    )
                routed_skill_ids = []
                routed_skills = []

        active_skills_for_images = routed_skills
        if lock_is_explicit and locked_skill_ids:
            active_skills_for_images = _assembler.route_skills(
                llm_message, forced_skill_ids=locked_skill_ids
            )
        if not images:
            recovered_images: list[ImageContent] = []
            reuse_reason = ""
            skill_needs_images = _skill_requires_images(active_skills_for_images)
            if _message_requests_image_regenerate(llm_message):
                if (
                    "nano-banana-image-t8" in locked_skill_ids
                    and _recover_last_nano_banana_mode(session) == "text"
                ):
                    recovered_images = []
                else:
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
        nano_banana_control_only = (
            lock_is_explicit
            and "nano-banana-image-t8" in locked_skill_ids
            and _is_nano_banana_control_message(llm_message)
        )
        nano_banana_activation_only = (
            lock_is_explicit
            and "nano-banana-image-t8" in locked_skill_ids
            and _is_nano_banana_activation_message(llm_message)
        )
        nano_banana_execution_request = _is_nano_banana_execution_request(
            message=llm_message,
            has_new_input_images=has_new_input_images,
        )
        if (
            lock_is_explicit
            and "nano-banana-image-t8" in locked_skill_ids
            and not nano_banana_control_only
            and not nano_banana_execution_request
        ):
            active_skills_for_images = []
        skip_nano_banana_guard = (
            lock_is_explicit
            and "nano-banana-image-t8" in locked_skill_ids
            and not nano_banana_control_only
            and not nano_banana_execution_request
        )

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

        if lock_is_explicit and locked_skill_ids and not lock_waiting_done and session is not None:
            forced_skill_ids = locked_skill_ids
            if skip_nano_banana_guard:
                forced_skill_ids = [
                    skill_id for skill_id in locked_skill_ids if skill_id != "nano-banana-image-t8"
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
                            str(k): v for k, v in value.items() if isinstance(k, str)
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
                    updated, missing = _update_guard_state(
                        guard.params, skill_state, llm_message, images
                    )
                    if skill.id == "nano-banana-image-t8":
                        if skip_nano_banana_guard:
                            state_map[skill.id] = skill_state
                            continue
                        previous_prompt = str(skill_state.get("prompt", "")).strip()
                        control_message_only = nano_banana_control_only
                        if control_message_only and "prompt" in updated:
                            updated["prompt"] = skill_state.get("prompt")
                        elif has_new_input_images:
                            cleaned_prompt = _clean_nano_banana_prompt_delta(llm_message)
                            if cleaned_prompt:
                                updated["prompt"] = cleaned_prompt
                        else:
                            # 用户明确说"文生图"时，视为全新请求，
                            # 不当作 edit/regenerate 来合并旧 prompt。
                            _is_explicit_t2i = any(
                                p.search(llm_message)
                                for p in _NANO_BANANA_TEXT_TO_IMAGE_PATTERNS
                            )
                            merged_prompt = _merge_nano_banana_prompt(
                                previous_prompt=previous_prompt,
                                message=llm_message,
                                regenerate=(
                                    _message_requests_image_regenerate(llm_message)
                                    and not _is_explicit_t2i
                                ),
                                image_edit=(
                                    _message_requests_image_edit(llm_message)
                                    and not _is_explicit_t2i
                                ),
                            )
                            if merged_prompt:
                                updated["prompt"] = merged_prompt
                        previous_model = str(
                            skill_state.get(
                                "__model_display__",
                                _load_saved_nano_banana_model_display(),
                            )
                        )
                        updated["__model_display__"] = _detect_nano_banana_model_display(
                            llm_message,
                            previous=previous_model,
                        )
                        session.metadata["last_nano_banana_model_display"] = str(
                            updated["__model_display__"]
                        ).strip() or previous_model
                        missing = _nano_banana_missing_required(
                            updated,
                            control_message_only=control_message_only,
                        )
                    state_map[skill.id] = updated
                    missing_any = missing_any or missing
                session.metadata["skill_param_state"] = state_map
                metadata_dirty = True
                if (
                    "nano-banana-image-t8" in forced_skill_ids
                    and nano_banana_control_only
                    and not nano_banana_activation_only
                ):
                    if session_manager is not None:
                        await session_manager.update_metadata(session, session.metadata)
                    model_display = str(
                        state_map.get("nano-banana-image-t8", {}).get(
                            "__model_display__",
                            _resolve_nano_banana_model_display(session),
                        )
                    ).strip() or _resolve_nano_banana_model_display(session)
                    return (
                        f"已切换本次生图模型为：{model_display}。\n"
                        "如需继续生图，请直接发送提示词或图片；"
                        "如本轮任务已结束，请回复“任务完成”解除技能锁定。"
                    )
                if missing_any:
                    if session_manager is not None:
                        await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            session.id,
                            metadata=session.metadata,
                        )
                    blocks = [
                        _build_skill_param_guard_reply(
                            s.id, s.param_guard.params, state_map.get(s.id, {})  # pyright: ignore[reportUnknownArgumentType]
                        )
                        for s in guards
                        if s.param_guard is not None
                    ]
                    return "\n\n".join(blocks)
                if (
                    "nano-banana-image-t8" in locked_skill_ids
                    and nano_banana_execution_request
                    and registry.get("bash") is not None
                ):
                    nano_state = state_map.get("nano-banana-image-t8", {})
                    prompt = _strip_inline_image_markdown(
                        str(nano_state.get("prompt", "")).strip()
                    )
                    model_display = str(
                        nano_state.get(
                            "__model_display__",
                            _resolve_nano_banana_model_display(session),
                        )
                    ).strip() or _resolve_nano_banana_model_display(session)
                    ratio = str(nano_state.get("ratio") or "auto").strip() or "auto"
                    input_paths = _resolve_nano_banana_input_paths(llm_message, session)
                    mode = "edit" if input_paths else "text"
                    if input_paths:
                        session.metadata["last_input_image_paths"] = input_paths
                        _append_image_reference_history(session.metadata, input_paths)
                    session.metadata["last_nano_banana_mode"] = mode
                    session.metadata["last_nano_banana_model_display"] = model_display
                    if isinstance(nano_state, dict):
                        nano_state["__last_mode__"] = mode
                    nano_banana_recommended_command = _build_nano_banana_command(
                        mode=mode,
                        model_display=model_display,
                        prompt=prompt,
                        input_paths=input_paths,
                        ratio=ratio,
                    )
                    metadata_dirty = True

    if session is not None:
        pending_raw = session.metadata.get("evomap_pending_choices")
        if isinstance(pending_raw, dict):
            options_raw: object = pending_raw.get("options")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            options: list[dict[str, str]] = []
            if isinstance(options_raw, list):
                for item in options_raw:  # pyright: ignore[reportUnknownVariableType]
                    if not isinstance(item, dict):
                        continue
                    opt: dict[str, object] = item  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
                    aid = str(opt.get("asset_id", "")).strip()
                    summary = str(opt.get("summary", "")).strip()
                    if not summary and not aid:
                        continue
                    options.append({"asset_id": aid, "summary": summary})
            choice_idx = _extract_evomap_choice_index(message, len(options))
            if choice_idx is not None and options:
                selected = options[choice_idx]
                selected_hint = (
                    f"【EvoMap 已选方案】\n- {selected['asset_id']}: {selected['summary']}"
                )
                extra_memory = (
                    f"{selected_hint}\n{extra_memory}" if extra_memory.strip() else selected_hint
                )
                origin_message = str(pending_raw.get("origin_message", "")).strip()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                if origin_message:
                    llm_message = (
                        f"{origin_message}\n"
                        f"用户已选择方案：{selected['summary']}。\n"
                        "请按该方案执行。"
                    )
                session.metadata.pop("evomap_pending_choices", None)
                if session_manager is not None:
                    await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                        session.id,
                        metadata=session.metadata,
                    )

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
        if nano_banana_recommended_command:
            for _nb_tool in ("bash", "reminder", "cron"):
                if _nb_tool in available_tool_names:
                    selected_tool_names.add(_nb_tool)
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
    fallback_text = (
        "" if native_tools else registry.to_prompt_fallback(include_names=fallback_names)
    )

    effective_skill_ids = locked_skill_ids or pending_lock_skill_ids or None
    system_messages = _assembler.build(
        config,
        llm_message,
        tool_fallback_text=fallback_text,
        assistant_name=assistant_name,
        forced_skill_ids=effective_skill_ids,
    )
    if lock_is_explicit and locked_skill_ids:
        system_messages.append(_build_skill_lock_system_message(locked_skill_ids))
        if skill_lock_resumed_from_waiting:
            joined_ids = ", ".join(locked_skill_ids)
            system_messages.append(Message(
                role="system",
                content=(
                    f"用户刚刚发送了新的任务请求，当前仍锁定在 {joined_ids} 技能。\n"
                    "请立即按照技能规则执行用户的新请求，直接调用工具完成任务。\n"
                    "禁止输出使用说明、操作指南或功能介绍，直接动手做。"
                ),
            ))
        if "nano-banana-image-t8" in locked_skill_ids:
            current_model = _resolve_nano_banana_model_display(session)
            system_messages.append(
                _build_nano_banana_execution_system_message(
                    current_model,
                    _recover_recent_session_image_paths(session),
                )
            )
    if nano_banana_recommended_command:
        system_messages.append(Message(
            role="system",
            content=(
                "nano-banana 生图参数已齐，以下是命令模板（timeout 设为 330）：\n"
                f"```\n{nano_banana_recommended_command}\n```\n"
                "使用规则：\n"
                "1. 如果用户只是要求生图，直接调用 bash 执行上述命令。\n"
                "2. 如果用户的请求是复合任务（如先写文档再生图），"
                "请按正确顺序编排：先完成其他步骤，在需要生图时再调用 bash 执行。\n"
                "3. 你可以根据用户意图调整 --prompt 参数内容"
                "（只放与生图相关的描述，去掉写文档等无关部分）。\n"
                "4. 如果用户要多张图，可以多次调用 bash，"
                "每次修改 --prompt 以生成不同内容。\n"
                "5. 不可修改的参数：--mode、--model、--edit-model、脚本路径。\n"
                "6. 如果用户要求延迟执行（如 N分钟后生图），"
                "请调用 reminder 工具，设置 action='agent'，"
                "message 填写完整的生图指令（如 用香蕉生图画一只波斯猫），"
                "这样到时间后会自动触发 agent 执行该任务。\n"
                "7. 禁止输出命令本身给用户看，直接调用 bash 工具执行。"
            ),
        ))
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
    conversation.append(Message(role="user", content=llm_message, images=images))
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
    db_summaries = []
    pending_office_paths: list[str] = []
    if session_store and summarizer_cfg.enabled:
        try:
            db_summaries = await session_store.get_summaries(session_id)
        except Exception as exc:
            log.debug("agent.summaries_load_failed", error=str(exc))

    guard_state = ToolGuardState()
    invalid_tool_rounds = 0
    empty_reply_rounds = 0
    office_block_bash_probe = False
    office_loop_guard_enabled = False
    office_block_message = ""
    office_edit_only = False
    office_edit_path = ""
    if session is not None:
        is_office_request = _is_office_edit_request(llm_message) or (
            _is_followup_edit_message(llm_message) and _has_any_last_office_path(session.metadata)
        )
        office_loop_guard_enabled = is_office_request
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
    browser_locked_by_evomap = evo_first_mode and not evomap_hint_hit
    while total_output < _MAX_OUTPUT_TOKENS:
        round_idx += 1
        if round_idx >= max_tool_rounds:
            final_text_parts.append(
                f"工具调用轮次已达上限（{max_tool_rounds} 轮），"
                "为避免长时间卡住已暂停。请让我改用更直接的编辑方式继续。"
            )
            break
        if db_summaries:
            all_messages = _context_window.trim_with_summaries(
                [*system_messages, *conversation],
                model_short,
                db_summaries,
            )
        else:
            all_messages = _context_window.trim(
                [*system_messages, *conversation],
                model_short,
            )

        _llm_t0 = _time.monotonic()
        _llm_retries = 0
        _llm_last_err: Exception | None = None
        while _llm_retries <= 2:
            try:
                response = await router.chat(
                    model_id,
                    all_messages,
                    tools=tool_schemas or None,
                    on_stream=on_stream,
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
                    raise
                log.warning(
                    "agent.llm_retry",
                    attempt=_llm_retries,
                    error=str(exc)[:200],
                    model=model_id,
                    session_id=session_id,
                )
                await asyncio.sleep(1.5 * _llm_retries)
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
                if invalid_tool_rounds >= 2:
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
                if invalid_tool_rounds >= 2:
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
            await _persist_message(session_manager, session, "assistant", assistant_persist)
            if assistant_content:
                for office_path in _extract_office_paths(assistant_content):
                    if _remember_office_path(session.metadata, office_path):
                        metadata_dirty = True

        for _tname in ("reminder", "cron"):
            _tool = registry.get(_tname)
            if _tool is not None and hasattr(_tool, "current_session_id"):
                _tool.current_session_id = session_id  # type: ignore[union-attr]

        stop_for_evomap_choice = False
        stop_for_probe_loop = False
        for tc in tool_calls:
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

            if tc.id in _evomap_preflight_results:
                tc_id, result = _evomap_preflight_results[tc.id]
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
                    top3 = _pick_top_evomap_candidates(llm_message, candidates, limit=3)
                    session.metadata["evomap_pending_choices"] = {
                        "origin_message": llm_message,
                        "options": [{"asset_id": aid, "summary": summary} for aid, summary in top3],
                    }
                    if session_manager is not None:
                        await session_manager._store.update_session_field(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                            session.id,
                            metadata=session.metadata,
                        )
                    final_text_parts.append(_build_evomap_choice_prompt(top3))
                    stop_for_evomap_choice = True
                    break
                browser_locked_by_evomap = not _is_no_match_evomap_output(result)
            guard_update = apply_tool_result_guards(
                guard_state,
                tc,
                result,
                office_loop_guard_enabled=office_loop_guard_enabled,
                image_api_probe_guard_enabled=image_api_probe_guard_enabled,
                session_id=session_id,
            )
            for event in guard_update.log_events:
                getattr(log, event.level)(event.event, **event.fields)
            for prompt in guard_update.conversation_messages:
                conversation.append(Message(role="user", content=prompt))
            final_text_parts.extend(guard_update.final_texts)
            if guard_update.stop_for_probe_loop:
                stop_for_probe_loop = True
                break

            if result.success and result.output:
                successful_tool_calls += 1
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

            if (
                session is not None
                and tc.name == "bash"
                and result.success
                and "nano-banana-image-t8" in locked_skill_ids
            ):
                _nb_output = result.output or ""
                _nb_image_path = _parse_nano_banana_result_path(_nb_output)
                if _nb_image_path and Path(_nb_image_path).expanduser().is_file():
                    session.metadata["last_generated_image_path"] = _nb_image_path
                    _append_image_reference_history(
                        session.metadata, [_nb_image_path]
                    )
                    metadata_dirty = True

            if session is not None and tc.name in {"ppt_edit", "docx_edit", "xlsx_edit"}:
                arg_path = tc.arguments.get("path")
                if (
                    isinstance(arg_path, str)
                    and arg_path.strip()
                    and _remember_office_path(session.metadata, arg_path.strip())
                ):
                    metadata_dirty = True

            tool_output = _format_tool_output(result)

            if not result.success:
                _dedup_consecutive_errors(conversation, tc.name, tool_output)

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
                snippet = tool_output[:500] if len(tool_output) > 500 else tool_output
                await _persist_message(
                    session_manager,
                    session,
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

        if stop_for_evomap_choice:
            break
        if stop_for_probe_loop:
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
            await session_manager.update_metadata(session, session.metadata)
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

    if lock_is_explicit and locked_skill_ids and skill_announce_pending:
        announce = _skill_announcement(locked_skill_ids, previous_locked_skill_ids)
        final_text = f"{announce}\n\n{final_text}" if final_text else announce
        skill_announce_pending = False
        if session is not None:
            session.metadata["skill_lock_announce_pending"] = False
            metadata_dirty = True

    _lock_confirm_tip = (
        "如果本轮任务已完成，请回复“任务完成”以解除技能锁定；"
        "若需继续修改，请直接继续说需求。"
    )

    # Deferred lock: first run completed with auto-routed skills -> lock them now.
    if (
        not lock_is_explicit
        and pending_lock_skill_ids
        and session is not None
        and successful_tool_calls > 0
    ):
        locked_skill_ids = pending_lock_skill_ids
        lock_is_explicit = True
        session.metadata["locked_skill_ids"] = locked_skill_ids
        session.metadata["skill_lock_waiting_done"] = True
        metadata_dirty = True
        final_text = f"{final_text}\n\n{_lock_confirm_tip}" if final_text else _lock_confirm_tip

    # Explicit lock: require user confirmation to release after successful tool use.
    elif (
        lock_is_explicit
        and locked_skill_ids
        and session is not None
        and successful_tool_calls > 0
        and not lock_waiting_done
    ):
        session.metadata["locked_skill_ids"] = locked_skill_ids
        session.metadata["skill_lock_waiting_done"] = True
        metadata_dirty = True
        if has_skill_queue and _skill_queue_has_next(skill_queue, skill_queue_index):
            if skill_queue_index < len(skill_queue):
                skill_queue[skill_queue_index]["status"] = "done"
                session.metadata["skill_queue"] = skill_queue
            _q_advance_tip = _build_skill_queue_advance_message(
                skill_queue, skill_queue_index,
            )
            final_text = f"{final_text}\n\n{_q_advance_tip}" if final_text else _q_advance_tip
        else:
            final_text = f"{final_text}\n\n{_lock_confirm_tip}" if final_text else _lock_confirm_tip

    if metadata_dirty and session is not None and session_manager is not None:
        await session_manager.update_metadata(session, session.metadata)

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

    log.info(
        "agent.done",
        model=model_id,
        llm_rounds=max(0, round_idx + 1),
        input_tokens=total_input,
        output_tokens=total_output,
        session_id=session_id,
        trigger_event_id=trigger_event_id,
        trigger_preview=trigger_preview,
    )

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

    return final_text
