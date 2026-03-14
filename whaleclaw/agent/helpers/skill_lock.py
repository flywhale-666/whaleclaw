"""Skill-lock and assistant-name helpers for the single-agent runtime."""

from __future__ import annotations

import os
import re
from pathlib import Path

from whaleclaw.providers.base import ImageContent, Message
from whaleclaw.skills.parser import Skill, SkillParamItem
from whaleclaw.tools.base import ToolDefinition
from whaleclaw.tools.registry import ToolRegistry

CORE_NATIVE_TOOLS = {
    "browser",
    "bash",
    "file_read",
    "file_write",
    "file_edit",
    "patch_apply",
    "ppt_edit",
    "docx_edit",
    "xlsx_edit",
}
_NANO_BANANA_DEFAULT_MODEL_FILE = (
    Path.home() / ".whaleclaw" / "credentials" / "nano_banana_default_model.txt"
)
MAX_NATIVE_TOOLS = 12
TOOL_POLICY_KEYWORDS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("桌面截图", "截图桌面", "截屏", "屏幕截图", "desktop screenshot"): ("desktop_capture",),
    ("ppt", "pptx", "幻灯片", "演示文稿"): ("ppt_edit", "file_edit", "patch_apply"),
    ("word", "docx", "文档"): ("docx_edit", "file_edit"),
    ("excel", "xlsx", "表格", "单元格"): ("xlsx_edit", "file_edit"),
    ("网页", "网站", "页面", "链接", "url"): ("browser", "web_fetch"),
    ("代码", "脚本", "终端", "命令", "日志"): ("bash", "file_read", "file_write"),
    ("进程", "后台", "卡住", "kill", "日志"): ("process", "bash"),
}


def preview_text(text: str, limit: int = 80) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def sanitize_assistant_name(raw: str) -> str:
    name = raw.strip().strip("\"'“”‘’「」[]()（）")
    if not name:
        return ""
    if any(ch in name for ch in ("?", "？", "!", "！", "吗")):
        return ""
    low = name.lower()
    if low in {"什么", "啥", "name", "名字"}:
        return ""
    if "什么" in name:
        return ""
    if len(name) > 24:
        return ""
    if not re.fullmatch(r"[\w\u4e00-\u9fff·\-.]{1,24}", name):
        return ""
    return name


def detect_assistant_name_update(
    message: str,
    *,
    reset_patterns: tuple[re.Pattern[str], ...],
    set_patterns: tuple[re.Pattern[str], ...],
) -> tuple[str, str]:
    text = message.strip()
    if not text:
        return ("none", "")
    for pattern in reset_patterns:
        if pattern.search(text):
            return ("reset", "")
    for pattern in set_patterns:
        match = pattern.search(text)
        if not match:
            continue
        name = sanitize_assistant_name(match.group(1))
        if name:
            return ("set", name)
    return ("none", "")


def normalize_for_match(text: str) -> str:
    return " ".join(text.lower().split())


def parse_use_command(
    text: str,
    *,
    use_cmd_re: re.Pattern[str],
) -> tuple[list[str], str] | None:
    match = use_cmd_re.match(text)
    if not match:
        return None
    token = match.group(1).strip().lower()
    skill_ids = [item.strip() for item in token.split(",") if item.strip()]
    remainder = match.group(2).strip()
    return (skill_ids, remainder)


def is_task_done_confirmation(
    text: str,
    *,
    task_done_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in task_done_patterns)


def build_skill_lock_system_message(skill_ids: list[str]) -> Message:
    joined = ", ".join(skill_ids)
    return Message(
        role="system",
        content=(
            f"当前会话已锁定技能：{joined}。\n"
            "核心任务必须优先使用锁定技能的方式完成，不要偏移到无关方案。\n"
            "但如果用户请求中包含后续步骤（如把结果放到文档/PPT/表格中），"
            "在锁定技能部分完成后，可以继续使用通用工具（bash、file_write、"
            "ppt_edit、docx_edit 等）完成后续步骤，不需要先解锁。\n"
            "若用户明确回复“任务完成”，再解除该锁定。"
        ),
    )


def build_nano_banana_execution_system_message(
    current_model: str,
    recent_image_paths: list[str],
) -> Message:
    if recent_image_paths:
        alias_lines: list[str] = []
        for index, path in enumerate(recent_image_paths):
            if index == 0:
                alias = "当前这张/最新一张"
            elif index == 1:
                alias = "上一张/前一张"
            elif index == 2:
                alias = "再上一张/上上张"
            else:
                alias = f"倒数第{index + 1}张"
            alias_lines.append(f"- {alias}: {path}")
        image_lines = "\n".join(alias_lines)
    else:
        image_lines = "- 当前没有可复用的历史图片路径"
    return Message(
        role="system",
        content=(
            "当前正在执行 nano-banana-image-t8 技能。\n"
            "执行约束：\n"
            f"1) 当前本轮模型是：{current_model}。若调用脚本，"
            f"必须把 `--model` 和 `--edit-model` 都设置为 `{current_model}`，"
            "不要继续沿用其它模型。\n"
            "2) 对外回复只使用展示名“香蕉2”或“香蕉pro”，"
            "除非用户明确追问，不要暴露底层模型标识。\n"
            "3) 模式选择规则（非常重要）：\n"
            "   - 文生图（--mode text）：用户描述了一个全新画面/题材，"
            "不要加 --input-image，直接用 prompt 生成。\n"
            "   - 图生图（--mode edit + --input-image）：仅当用户明确说"
            "“处理这张图”“修改上一张”“重试”"
            "“继续处理”等续跑/编辑语义时才使用。\n"
            "   - 判断标准：如果用户的描述主题与历史图片无关，就是文生图。\n"
            "4) 图片引用规则：当前这张=最新一张；上一张=倒数第二张；再上一张=倒数第三张。\n"
            "5) 仅在图生图模式下才需要以下历史图片路径，文生图模式请忽略：\n"
            f"{image_lines}"
        ),
    )


def looks_like_skill_activation_message(
    text: str,
    *,
    skill_activation_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.lower().startswith("/use "):
        return True
    return any(pattern.search(stripped) for pattern in skill_activation_patterns)


def is_skill_switch_consent(
    text: str,
    *,
    skill_switch_consent_patterns: tuple[re.Pattern[str], ...],
) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.search(stripped) for pattern in skill_switch_consent_patterns)


def normalize_skill_ids(skills: list[Skill]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for skill in skills:
        skill_id = skill.id.strip().lower()
        if not skill_id or skill_id in seen:
            continue
        seen.add(skill_id)
        output.append(skill_id)
    return output


def skill_announcement(skill_ids: list[str], previous_skill_ids: list[str]) -> str:
    joined = "、".join(skill_ids)
    if not previous_skill_ids:
        return f"我将使用 {joined} 技能继续完成任务。"
    if previous_skill_ids != skill_ids:
        return f"已按你的要求切换为 {joined} 技能，我继续处理。"
    return f"我会继续使用 {joined} 技能推进当前任务。"


def skill_token_mentioned(token: str, text: str) -> bool:
    lower = text.lower()
    msg_norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lower)

    base = token.strip().lower().replace("_", "-")
    if not base:
        return False

    candidates: set[str] = {base, base.replace("-", "")}
    parts = [part for part in base.split("-") if part]
    if len(parts) >= 2:
        short = "-".join(parts[:2])
        candidates.add(short)
        candidates.add(short.replace("-", ""))

    for candidate in candidates:
        if len(candidate) < 4:
            continue
        if candidate in lower or candidate in msg_norm:
            return True
    return False


def skill_explicitly_mentioned(skill: Skill, text: str) -> bool:
    return skill_token_mentioned(skill.id, text) or skill_token_mentioned(skill.name, text)


def skill_trigger_mentioned(skill: Skill, text: str) -> bool:
    """Return whether the message includes any sufficiently specific trigger."""
    lower = text.lower()
    msg_norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", lower)
    for raw in skill.triggers:
        trigger = raw.strip().lower()
        if len(trigger) < 4:
            continue
        if trigger in lower:
            return True
        norm = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", trigger)
        if len(norm) >= 4 and norm in msg_norm:
            return True
    return False


def extract_ratio_or_size(text: str) -> str:
    ratio = re.search(r"(?<!\d)(\d{1,2}\s*:\s*\d{1,2})(?!\d)", text)
    if ratio:
        return ratio.group(1).replace(" ", "")
    size = re.search(r"(?<!\d)(\d{3,5}\s*x\s*\d{3,5})(?!\d)", text, re.IGNORECASE)
    if size:
        return size.group(1).replace(" ", "").lower()
    return ""


def extract_value_by_aliases(text: str, aliases: list[str]) -> str:
    for alias in aliases:
        pattern = rf"{re.escape(alias)}\s*(?:是|为|:|：)\s*(.+)$"
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return ""


def has_param_secret_source(param: SkillParamItem) -> bool:
    for env_name in param.env_vars:
        val = os.getenv(env_name, "").strip()
        if val:
            return True
    if param.saved_file:
        path = Path(param.saved_file).expanduser()
        try:
            return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
        except Exception:
            return False
    return False


def capture_param_value(
    param: SkillParamItem,
    text: str,
    images: list[ImageContent] | None,
    previous: object,
) -> object:
    param_type = param.type.strip().lower()
    aliases = [param.key, *param.aliases]
    if param_type == "images":
        prev_count = int(previous) if isinstance(previous, int) else 0
        return max(prev_count, len(images or []))
    if param_type in {"ratio", "size"}:
        value = extract_ratio_or_size(text) or extract_value_by_aliases(text, aliases)
        return value or previous
    if param_type == "api_key":
        if re.search(r"\bsk-[A-Za-z0-9_-]{12,}\b", text):
            return "__present__"
        alias_val = extract_value_by_aliases(text, aliases)
        if alias_val and "sk-" in alias_val.lower():
            return "__present__"
        if has_param_secret_source(param):
            return "__present__"
        return previous
    alias_val = extract_value_by_aliases(text, aliases)
    if alias_val:
        return alias_val
    if param_type == "text":
        stripped = text.strip()
        if (
            stripped
            and len(stripped) >= 6
            and not stripped.startswith("/use ")
            and "技能" not in stripped
        ):
            if any(token in stripped for token in ("api key", "apikey", "尺寸", "比例")):
                return previous
            return stripped
    return previous


def param_satisfied(param: SkillParamItem, value: object) -> bool:
    param_type = param.type.strip().lower()
    if param_type == "images":
        count = int(value) if isinstance(value, int) else 0
        return count >= max(1, int(param.min_count))
    if value is None:
        return False
    return bool(str(value).strip())


def format_param_status(param: SkillParamItem, value: object) -> str:
    label = param.label or param.key
    param_type = param.type.strip().lower()
    if param_type == "images":
        count = int(value) if isinstance(value, int) else 0
        need = max(1, int(param.min_count))
        return f"{label}：已收到 {count} 张（至少 {need} 张）"
    if param_satisfied(param, value):
        if param_type == "api_key":
            return f"{label}：已就绪"
        return f"{label}：已收到"
    return f"{label}：未提供"


def detect_nano_banana_model_display(text: str, previous: str = "香蕉2") -> str:
    """Infer the selected Nano Banana model display name from user text."""
    normalized = text.strip().lower()
    if not normalized:
        return previous
    if any(token in normalized for token in ("香蕉pro", "nano-banana-2", "banana pro")):
        return "香蕉pro"
    if any(
        token in normalized
        for token in ("香蕉2", "banana生图", "香蕉生图", "gemini-3.1-flash-image-preview")
    ):
        return "香蕉2"
    return previous


_NANO_BANANA_CONTROL_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:请)?(?:我想|我要|我想要)?(?:用|使用)?\s*"
        r"(?:香蕉生图|香蕉文生图|香蕉图生图|banana生图|banana文生图|banana图生图|"
        r"nanobanana|nano\s*banana|nano-banana-2)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:切换|切换到|改用|换成|默认模型)\s*(?:香蕉2|香蕉pro)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:模型)?(?:切换|改成|换成)\s*(?:香蕉2|香蕉pro)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:切换|改用|换成)\s*模型\s*(?:香蕉2|香蕉pro)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:(?:不对[呀啊]?[,，\s]*)?(?:我的|当前)?(?:默认)?模型(?:应该是|改成|换成|切换到)?\s*"
        r"(?:香蕉2|香蕉pro)(?:[呀啊哦呢吧嘛]*)?)\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:用|改用|切换到)?\s*(?:香蕉2|香蕉pro)\s*(?:重试|再试一次|继续|继续跑)?\s*$",
        re.IGNORECASE,
    ),
)

_NANO_BANANA_ACTIVATION_MESSAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^(?:请)?(?:我想|我要|我想要)?(?:用|使用)?\s*"
        r"(?:香蕉生图|香蕉文生图|香蕉图生图|banana生图|banana文生图|banana图生图|"
        r"nanobanana|nano\s*banana|nano-banana-2)\s*$",
        re.IGNORECASE,
    ),
)


def is_nano_banana_control_message(text: str) -> bool:
    """Return whether the message is only a Nano Banana control/activation command."""
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in _NANO_BANANA_CONTROL_MESSAGE_PATTERNS)


def is_nano_banana_activation_message(text: str) -> bool:
    """Return whether the message is a pure Nano Banana activation command."""
    stripped = text.strip()
    if not stripped:
        return False
    return any(pattern.fullmatch(stripped) for pattern in _NANO_BANANA_ACTIVATION_MESSAGE_PATTERNS)


def load_saved_nano_banana_model_display() -> str:
    """Load persisted Nano Banana default model display name."""
    try:
        raw = _NANO_BANANA_DEFAULT_MODEL_FILE.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return "香蕉2"
    if raw == "nano-banana-2":
        return "香蕉pro"
    return "香蕉2"


def _build_nano_banana_param_guard_reply(state: dict[str, object]) -> str:
    """Render the dedicated parameter guard copy for Nano Banana."""
    current_model = str(
        state.get("__model_display__", load_saved_nano_banana_model_display())
    ).strip() or "香蕉2"
    if current_model == "香蕉pro":
        model_line = "2) 当前模型：香蕉pro（0.2元）可切换模型香蕉2（0.1元）"
    else:
        model_line = "2) 当前模型：香蕉2（0.1元）可切换模型香蕉pro（0.2元）"

    prompt_status = "3) 提示词：已收到" if param_satisfied(
        SkillParamItem(key="prompt", type="text"),
        state.get("prompt"),
    ) else "3) 提示词：未提供"
    image_status = format_param_status(
        SkillParamItem(
            key="images",
            label="图生图图片",
            type="images",
            required=False,
            min_count=1,
        ),
        state.get("images"),
    )
    lines = [
        "我将使用 nano-banana-image-t8 技能继续完成任务。",
        "",
        "我先确认参数（缺啥补啥）：",
        (
            "1) API Key：已就绪"
            if param_satisfied(
                SkillParamItem(key="api_key", type="api_key"),
                state.get("api_key"),
            )
            else "1) API Key：未提供"
        ),
        model_line,
        prompt_status,
        f"4) {image_status}",
        "5) 切换本次模型：切换香蕉2（pro）。设置默认模型：默认模型香蕉2（pro）",
    ]
    missing_prompts: list[str] = []
    if not param_satisfied(SkillParamItem(key="api_key", type="api_key"), state.get("api_key")):
        missing_prompts.append("请提供 Nano Banana API Key")
    if not param_satisfied(SkillParamItem(key="prompt", type="text"), state.get("prompt")):
        missing_prompts.append("请提供提示词")
    raw_images = state.get("images", 0)
    image_count = int(raw_images) if isinstance(raw_images, int) else 0
    if image_count < 1:
        missing_prompts.append("请上传图片")
    if missing_prompts:
        lines.append("请补充：" + "；".join(missing_prompts) + "。")
    else:
        lines.append("参数已齐，我现在开始执行。")
    return "\n".join(lines)


def nano_banana_missing_required(
    state: dict[str, object],
    *,
    control_message_only: bool,
) -> bool:
    """Decide whether Nano Banana should stay in the fixed parameter-guard stage."""
    has_key = param_satisfied(SkillParamItem(key="api_key", type="api_key"), state.get("api_key"))
    has_prompt = param_satisfied(SkillParamItem(key="prompt", type="text"), state.get("prompt"))
    raw_images = state.get("images", 0)
    image_count = int(raw_images) if isinstance(raw_images, int) else 0
    if not has_key or not has_prompt:
        return True
    return bool(image_count < 1 and control_message_only)


def build_skill_param_guard_reply(
    skill_id: str,
    params: list[SkillParamItem],
    state: dict[str, object],
) -> str:
    if skill_id == "nano-banana-image-t8":
        return _build_nano_banana_param_guard_reply(state)

    lines = [f"我将使用 {skill_id} 技能继续完成任务。", "", "我先确认参数（缺啥补啥）："]
    missing_prompts: list[str] = []
    for idx, param in enumerate(params, start=1):
        value = state.get(param.key)
        lines.append(f"{idx}) {format_param_status(param, value)}")
        if param.required and not param_satisfied(param, value):
            missing_prompts.append(param.prompt.strip() or (param.label or param.key))
    lines.append("")
    if missing_prompts:
        lines.append("请补充：" + "；".join(missing_prompts) + "。")
    else:
        lines.append("参数已齐，我现在开始执行。")
    return "\n".join(lines)


def update_guard_state(
    params: list[SkillParamItem],
    state: dict[str, object],
    message: str,
    images: list[ImageContent] | None,
) -> tuple[dict[str, object], bool]:
    new_state = dict(state)
    missing_required = False
    for param in params:
        prev = new_state.get(param.key)
        captured = capture_param_value(param, message, images, prev)
        new_state[param.key] = captured
        if param.required and not param_satisfied(param, captured):
            missing_required = True
    return new_state, missing_required


def guarded_skills(skills: list[Skill]) -> list[Skill]:
    output: list[Skill] = []
    for skill in skills:
        if (
            skill.param_guard is None
            or not skill.param_guard.enabled
            or not skill.param_guard.params
        ):
            continue
        output.append(skill)
    return output


def score_tool_relevance(user_message: str, tool: ToolDefinition) -> int:
    query = normalize_for_match(user_message)
    if not query:
        return 0

    score = 0
    corpus_parts = [tool.name, tool.description]
    corpus_parts.extend(param.name for param in tool.parameters)
    corpus_parts.extend(param.description for param in tool.parameters)
    corpus = normalize_for_match(" ".join(corpus_parts))

    keywords = {word for word in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}
    for keyword in keywords:
        if keyword and keyword in corpus:
            score += 1

    if tool.name == "desktop_capture" and (
        ("桌面" in query and ("截图" in query or "截屏" in query))
        or "desktop screenshot" in query
        or "screenshot" in query
    ):
        score += 6

    if ("图片" in query or "photo" in query or "image" in query) and tool.name in {
        "browser",
        "desktop_capture",
    }:
        score += 2

    if ("evomap" in query or "evo map" in query) and tool.name.startswith("evomap_"):
        score += 4

    return score


def select_native_tool_names(
    registry: ToolRegistry,
    user_message: str,
) -> set[str]:
    definitions = registry.list_tools()
    available = {definition.name for definition in definitions}
    selected = {name for name in CORE_NATIVE_TOOLS if name in available}
    query = normalize_for_match(user_message)

    for keywords, names in TOOL_POLICY_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            for name in names:
                if name in available:
                    selected.add(name)

    if len(selected) >= MAX_NATIVE_TOOLS:
        return selected

    scored: list[tuple[int, str]] = []
    for definition in definitions:
        if definition.name in selected:
            continue
        scored.append((score_tool_relevance(user_message, definition), definition.name))

    scored.sort(key=lambda item: (-item[0], item[1]))
    for score, name in scored:
        if score <= 0:
            break
        selected.add(name)
        if len(selected) >= MAX_NATIVE_TOOLS:
            break

    return selected


_QUEUE_ADVANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*继续\s*$"),
    re.compile(r"^\s*下一步\s*$"),
    re.compile(r"^\s*下一个\s*$"),
    re.compile(r"^\s*(?:继续|下一步|下一个)\s*(?:吧|呗|啊|嘛|了)?\s*$"),
    re.compile(r"^\s*(?:go|next|continue)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:执行|开始)\s*(?:下一步|下一个)\s*$"),
    re.compile(r"^\s*(?:好的?\s*)?(?:继续|下一步)\s*$"),
)


def is_queue_advance_confirmation(text: str) -> bool:
    """用户是否回复了"继续"/"下一步"等推进确认。"""
    stripped = text.strip()
    if not stripped:
        return False
    return any(p.search(stripped) for p in _QUEUE_ADVANCE_PATTERNS)


def build_skill_queue_plan_messages(
    skill_ids: list[str],
    user_message: str,
) -> list[Message]:
    """构建让 LLM 做任务拆解的 prompt，控制在 ~200 tokens。"""
    ids_str = ", ".join(skill_ids)
    return [
        Message(
            role="system",
            content=(
                "你是一个任务规划器。用户消息涉及多个技能，请按执行顺序拆解为子任务。\n"
                "输出严格 JSON 数组，每项包含 skill_id 和 task 字段。\n"
                "skill_id 必须从给定列表中选择。task 用一句话描述该步要做什么。\n"
                "如果某步的产出是下一步的输入，在 task 中说明。\n"
                "只输出 JSON，不要其它文字。"
            ),
        ),
        Message(
            role="user",
            content=f"可用技能: [{ids_str}]\n用户消息: {user_message}",
        ),
    ]


def parse_skill_queue_plan(
    llm_response: str,
    skill_ids: list[str],
) -> list[dict[str, str]]:
    """解析 LLM 返回的 JSON 队列。解析失败返回空列表。"""
    import json

    text = llm_response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        raw = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    allowed = set(skill_ids)
    result: list[dict[str, str]] = []
    for entry in raw:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("skill_id", "")).strip()  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        task = str(entry.get("task", "")).strip()  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        if sid not in allowed or not task:
            continue
        result.append({"skill_id": sid, "task": task, "status": "pending", "output_hint": ""})
    return result


def build_skill_queue_status_message(
    queue: list[dict[str, str]],
    current_index: int,
) -> str:
    """构建队列整体状态文案。"""
    lines: list[str] = ["收到复合任务，我将分步执行："]
    for i, item in enumerate(queue):
        sid = item.get("skill_id", "")
        task = item.get("task", "")
        if i < current_index:
            tag = "已完成"
        elif i == current_index:
            tag = "进行中"
        else:
            tag = "待执行"
        lines.append(f"  {i + 1}) [{tag}] {sid}: {task}")
    lines.append(f"\n我先执行第 {current_index + 1} 步...")
    return "\n".join(lines)


def build_skill_queue_advance_message(
    queue: list[dict[str, str]],
    finished_index: int,
) -> str:
    """构建"当前步完成，下一步预告"的播报文案。"""
    finished = queue[finished_index] if finished_index < len(queue) else None
    next_index = finished_index + 1
    has_next = next_index < len(queue)
    lines: list[str] = []
    if finished:
        hint = finished.get("output_hint", "")
        hint_text = f"（{hint}）" if hint else ""
        lines.append(f"第 {finished_index + 1} 步完成！{hint_text}")
    if has_next:
        next_item = queue[next_index]
        next_step = next_index + 1
        next_task = next_item.get("task", "")
        lines.append(
            f'回复\u201c继续\u201d推进到第 {next_step} 步: '
            f'{next_task}\uff1b'
            '回复\u201c任务完成\u201d结束全部任务。'
        )
    else:
        lines.append('全部子任务已完成！回复\u201c任务完成\u201d解除技能锁定。')
    return "\n".join(lines)


def skill_queue_has_next(
    queue: list[dict[str, str]],
    current_index: int,
) -> bool:
    """队列是否还有下一个待执行的步骤。"""
    return current_index + 1 < len(queue)


__all__ = [
    "build_skill_lock_system_message",
    "build_skill_queue_advance_message",
    "build_skill_queue_plan_messages",
    "build_skill_queue_status_message",
    "build_skill_param_guard_reply",
    "capture_param_value",
    "detect_nano_banana_model_display",
    "detect_assistant_name_update",
    "extract_ratio_or_size",
    "extract_value_by_aliases",
    "format_param_status",
    "guarded_skills",
    "has_param_secret_source",
    "is_nano_banana_activation_message",
    "is_nano_banana_control_message",
    "is_skill_switch_consent",
    "is_task_done_confirmation",
    "looks_like_skill_activation_message",
    "nano_banana_missing_required",
    "normalize_for_match",
    "normalize_skill_ids",
    "param_satisfied",
    "parse_use_command",
    "preview_text",
    "sanitize_assistant_name",
    "score_tool_relevance",
    "select_native_tool_names",
    "skill_announcement",
    "skill_explicitly_mentioned",
    "skill_trigger_mentioned",
    "skill_token_mentioned",
    "is_queue_advance_confirmation",
    "parse_skill_queue_plan",
    "skill_queue_has_next",
    "update_guard_state",
]
