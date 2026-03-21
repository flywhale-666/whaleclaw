# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAssignmentType=false, reportUnusedFunction=false
"""EvoMap 相关辅助函数 — 从 single_agent.py 提取。

包含 EvoMap 启用检测、候选方案解析、选择提示构建等逻辑。
"""

import re

from whaleclaw.agent.helpers.skill_helpers import normalize_for_match
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.providers.base import Message
from whaleclaw.tools.base import ToolResult

__all__ = [
    "EVOMAP_LINE_RE",
    "EVOMAP_CHOICE_PATTERNS",
    "is_evomap_enabled",
    "is_evomap_status_question",
    "is_tasky_message_for_evomap",
    "infer_task_kind",
    "extract_topic_terms",
    "recommended_evomap_signals",
    "extra_memory_has_evomap_hint",
    "is_no_match_evomap_output",
    "build_evomap_first_system_message",
    "extract_evomap_choice_index",
    "parse_evomap_fetch_candidates",
    "pick_top_evomap_candidates",
    "build_evomap_choice_prompt",
]

# ---------------------------------------------------------------------------
# 正则常量
# ---------------------------------------------------------------------------

EVOMAP_LINE_RE = re.compile(r"^\s*-\s*([^:]+):\s*(.+?)\s*$")

EVOMAP_CHOICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*(?:选|选择)?\s*([ABCabc])\s*$"),
    re.compile(r"^\s*(?:选|选择)?\s*([123])\s*$"),
)

# ---------------------------------------------------------------------------
# 公开函数
# ---------------------------------------------------------------------------


def is_evomap_enabled(config: WhaleclawConfig) -> bool:
    plugins_cfg = getattr(config, "plugins", None)
    if not isinstance(plugins_cfg, dict):
        return False
    evomap_cfg_raw: object = plugins_cfg.get("evomap", None)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if not isinstance(evomap_cfg_raw, dict):
        return False
    evomap_cfg: dict[str, object] = evomap_cfg_raw  # pyright: ignore[reportAssignmentType, reportUnknownVariableType]
    return bool(evomap_cfg.get("enabled", False))


def is_evomap_status_question(text: str) -> bool:
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


def is_tasky_message_for_evomap(text: str) -> bool:
    low = normalize_for_match(text)
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


def infer_task_kind(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("ppt", "幻灯片", "演示文稿", "slides", "deck")):
        return "ppt"
    if any(k in low for k in ("网页", "网站", "html", "web", "landing page", "前端")):
        return "web"
    if any(k in low for k in ("检索", "汇总", "调研", "信息", "collect", "research", "summarize")):
        return "research"
    return "general"


def extract_topic_terms(text: str, *, limit: int = 2) -> list[str]:
    low = normalize_for_match(text)
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


def recommended_evomap_signals(text: str) -> str:  # pyright: ignore[reportUnusedFunction]
    kind = infer_task_kind(text)
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
    return ",".join(base + extract_topic_terms(text, limit=2))


def extra_memory_has_evomap_hint(extra_memory: str) -> bool:
    text = extra_memory.strip()
    if not text:
        return False
    return "EvoMap 协作经验候选" in text


def is_no_match_evomap_output(result: ToolResult) -> bool:
    if not result.success:
        return False
    out = (result.output or "").strip()
    if not out:
        return True
    hints = ("未找到匹配方案", "暂无可用任务", "无已认领任务")
    return any(h in out for h in hints)


def build_evomap_first_system_message() -> Message:
    return Message(
        role="system",
        content=(
            "执行策略：本轮是流程任务，优先复用 EvoMap 成功经验。\n"
            "1) 必须先调用 evomap_fetch 获取经验候选；\n"
            "2) 只有当 evomap_fetch 无命中或失败时，才可调用 browser；\n"
            "3) 若 evomap_fetch 命中，请先按命中方案执行。"
        ),
    )


def extract_evomap_choice_index(text: str, options_count: int) -> int | None:
    if options_count <= 0:
        return None
    raw = text.strip()
    if not raw:
        return None
    for p in EVOMAP_CHOICE_PATTERNS:
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


def parse_evomap_fetch_candidates(output: str) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    items: list[tuple[str, str]] = []
    for ln in lines:
        m = EVOMAP_LINE_RE.match(ln)
        if not m:
            continue
        aid = m.group(1).strip()
        summary = m.group(2).strip()
        if not aid and not summary:
            continue
        items.append((aid, summary))
    return items


def pick_top_evomap_candidates(
    user_message: str,
    candidates: list[tuple[str, str]],
    *,
    limit: int = 3,
) -> list[tuple[str, str]]:
    query = normalize_for_match(user_message)
    terms = {t for t in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}

    scored: list[tuple[int, tuple[str, str]]] = []
    for item in candidates:
        aid, summary = item
        hay = normalize_for_match(f"{aid} {summary}")
        score = 0
        for t in terms:
            if t in hay:
                score += 1
        scored.append((score, item))

    scored.sort(key=lambda x: (-x[0], x[1][0]))
    return [item for _score, item in scored[: max(1, limit)]]


def build_evomap_choice_prompt(candidates: list[tuple[str, str]]) -> str:
    labels = ("A", "B", "C")
    lines = ["EvoMap 命中了多条可用方案，请先选一个我再执行："]
    for idx, item in enumerate(candidates[:3]):
        aid, summary = item
        label = labels[idx]
        lines.append(f"{label}. {aid} — {summary}")
    lines.append("请直接回复：选A / 选B / 选C")
    return "\n".join(lines)
