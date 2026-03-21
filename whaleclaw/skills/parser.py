"""Parse SKILL.md files with YAML frontmatter."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field


_BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "bundled"


class Skill(BaseModel):
    """Skill parsed from SKILL.md."""

    model_config = {"arbitrary_types_allowed": True}

    id: str
    name: str
    triggers: list[str] = Field(default_factory=list)
    trigger_description: str = ""
    instructions: str
    tools: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    max_tokens: int = 6000
    lock_session: bool = False
    is_user_installed: bool = False
    param_guard: SkillParamGuard | None = None
    source_path: Path
    hooks: Any = None


class SkillParamItem(BaseModel):
    """Skill parameter requirement metadata."""

    key: str
    label: str = ""
    type: str = "text"
    required: bool = True
    prompt: str = ""
    aliases: list[str] = Field(default_factory=list)
    min_count: int = 1
    env_vars: list[str] = Field(default_factory=list)
    saved_file: str = ""


class SkillParamGuard(BaseModel):
    """Parameter guard config parsed from frontmatter."""

    enabled: bool = False
    params: list[SkillParamItem] = Field(default_factory=list)


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL | re.MULTILINE)


def _extract_section(content: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$\s*\n(.*?)(?=^##\s|\Z)"
    m = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else ""


def _first_paragraph(text: str) -> str:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras[0] if paras else ""


def _make_param_item_from_keyword(keyword: str, *, required: bool) -> SkillParamItem | None:
    k = keyword.strip().lower()
    if not k:
        return None
    if any(x in k for x in ("api key", "apikey", "key", "令牌", "token")):
        return SkillParamItem(
            key="api_key",
            label="API Key",
            type="api_key",
            required=required,
            prompt="请提供 API Key",
            aliases=["apikey", "api key", "key"],
        )
    if any(x in k for x in ("提示词", "prompt")):
        return SkillParamItem(
            key="prompt",
            label="提示词",
            type="text",
            required=required,
            prompt="请提供提示词",
            aliases=["提示词", "prompt"],
        )
    if any(x in k for x in ("图片", "image", "img")):
        return SkillParamItem(
            key="images",
            label="图片",
            type="images",
            required=required,
            prompt="请上传图片",
            min_count=1,
        )
    if any(x in k for x in ("比例", "尺寸", "size", "aspect")):
        return SkillParamItem(
            key="ratio",
            label="尺寸/比例",
            type="ratio",
            required=required,
            prompt="可选填写比例或尺寸",
            aliases=["比例", "尺寸", "size"],
        )
    return None


_PARAM_HINT_KEYWORDS = ("最小必填", "必填", "确认参数", "缺啥补啥", "缺少参数", "可选")
_PARAM_HINT_EXCLUDE = ("必须遵守", "必须使用", "必须包含", "必须保持", "必须在", "命令行参数")


def _infer_param_guard_from_instructions(instructions: str) -> SkillParamGuard | None:
    lines = [ln.strip() for ln in instructions.splitlines() if ln.strip()]
    if not lines:
        return None
    params: list[SkillParamItem] = []
    seen: set[str] = set()
    for line in lines:
        line_l = line.lower()
        if any(ex in line_l for ex in _PARAM_HINT_EXCLUDE):
            continue
        is_hint_line = any(x in line_l for x in _PARAM_HINT_KEYWORDS)
        if not is_hint_line:
            continue
        required = not any(x in line_l for x in ("可选", "非必填", "选填"))
        chunks = re.split(r"[：:；;，,+/\s]+", line)
        for chunk in chunks:
            item = _make_param_item_from_keyword(chunk, required=required)
            if item is None or item.key in seen:
                continue
            seen.add(item.key)
            params.append(item)
    if not params:
        return None
    return SkillParamGuard(enabled=True, params=params)


class SkillParser:
    """Parse SKILL.md files with YAML frontmatter."""

    def parse(self, path: Path) -> Skill:
        """Parse SKILL.md file into Skill model."""
        raw = path.read_text(encoding="utf-8")
        body = raw
        frontmatter: dict[str, Any] = {}

        fm_match = _FRONTMATTER_RE.match(raw)
        if fm_match:
            with contextlib.suppress(yaml.YAMLError):
                loaded: Any = yaml.safe_load(fm_match.group(1))
                if isinstance(loaded, dict):
                    frontmatter = cast(dict[str, Any], loaded)
            body = raw[fm_match.end() :]

        raw_triggers: Any = frontmatter.get("triggers") or []
        if isinstance(raw_triggers, str):
            triggers: list[str] = [raw_triggers]
        elif isinstance(raw_triggers, list):
            triggers = [str(t) for t in cast(list[Any], raw_triggers)]
        else:
            triggers = []
        max_tokens = int(frontmatter.get("max_tokens") or 6000)
        lock_session = bool(
            frontmatter.get("lock_session", frontmatter.get("conversation_lock", False))
        )
        param_guard: SkillParamGuard | None = None
        guard_raw: Any = frontmatter.get("param_guard")
        if isinstance(guard_raw, dict):
            typed_guard = cast(dict[str, Any], guard_raw)
            params_raw: Any = typed_guard.get("params")
            items: list[SkillParamItem] = []
            if isinstance(params_raw, list):
                for p in cast(list[Any], params_raw):
                    if isinstance(p, dict) and cast(dict[str, Any], p).get("key"):
                        with contextlib.suppress(Exception):
                            items.append(SkillParamItem.model_validate(p))
            if items:
                param_guard = SkillParamGuard(
                    enabled=bool(typed_guard.get("enabled", True)),
                    params=items,
                )

        heading_match = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
        name = heading_match.group(1).strip() if heading_match else path.stem

        trigger_desc = _extract_section(body, "触发条件")
        trigger_description = _first_paragraph(trigger_desc)

        instructions = _extract_section(body, "指令")
        if not instructions:
            heading_end = heading_match.end() if heading_match else 0
            instructions = body[heading_end:].strip()
        if param_guard is None:
            param_guard = _infer_param_guard_from_instructions(instructions)

        tools_text = _extract_section(body, "工具")
        tools = [t.strip().lstrip("-* ").strip() for t in tools_text.splitlines() if t.strip()]

        examples_text = _extract_section(body, "示例")
        examples = [e.strip() for e in examples_text.splitlines() if e.strip()]

        skill_id = path.parent.name
        resolved = path.resolve()
        bundled_resolved = _BUNDLED_SKILLS_DIR.resolve()
        is_user = True
        try:
            resolved.relative_to(bundled_resolved)
            is_user = False
        except ValueError:
            pass

        return Skill(
            id=skill_id,
            name=name,
            triggers=triggers,
            trigger_description=trigger_description,
            instructions=instructions,
            tools=tools,
            examples=examples,
            max_tokens=max_tokens,
            lock_session=lock_session,
            is_user_installed=is_user,
            param_guard=param_guard,
            source_path=path,
        )
