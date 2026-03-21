# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportAssignmentType=false, reportUnusedFunction=false, reportUnnecessaryCast=false, reportUnnecessaryIsInstance=false, reportGeneralTypeIssues=false
"""图片引用相关的辅助函数 — 从 single_agent.py 提取。

包含图片路径修复、图片引用检测、图片历史管理等功能。
"""

import base64
import re
from pathlib import Path
from typing import cast

from whaleclaw.providers.base import ImageContent, Message
from whaleclaw.sessions.manager import Session
from whaleclaw.skills.hooks import SkillHooks
from whaleclaw.skills.parser import Skill
from whaleclaw.utils.log import get_logger

log = get_logger(__name__)

# ── 正则常量 ──────────────────────────────────────────────────

IMG_MD_RE = re.compile(r"!\[([^\]]*)\]\((/[^)]+)\)")
ABS_IMAGE_PATH_RE = re.compile(
    r"(/[^\s\"')]+?\.(?:png|jpg|jpeg|gif|webp))(?=[\s\"')]|$)",
    re.IGNORECASE,
)
IMAGE_REFERENCE_RE = re.compile(
    r"(这张图|这张图片|这幅图|这幅图片|图里|图中|参考图|按这张|基于这张|用这张)",
    re.IGNORECASE,
)
NUMBERED_IMAGE_REFERENCE_RE = re.compile(
    r"(?:图\s*[1-9一二三四五六七八九两]|第[一二三四五六七八九两123456789]+张图)",
    re.IGNORECASE,
)
IMAGE_EDIT_FOLLOWUP_RE = re.compile(
    r"(改(?:成|下|一下)?|修改|调整|优化|增强|变得|变成|换成|把.+(?:改|变|改成|变成|换成)|"
    r"让.+(?:改成|变成|换成)|"
    r"加上|加个|加一(?:个|只)?|加只|添加|增加|补上|再来(?:个|只)?|放一(?:个|只)?)",
    re.IGNORECASE,
)
IMAGE_EDIT_SUBJECT_CONTINUATION_RE = re.compile(
    r"^\s*(?:请)?(?:帮我)?(?:让|给)\s*(?:这|那|它|他|她)"
    r"(?:只|个|头|名|位|条|匹|张|幅)?",
    re.IGNORECASE,
)
IMAGE_REGENERATE_RE = re.compile(
    r"(重试|重做|重新生成|重生成|重新来|再来一版|再生成一次|再试一次|重画|这图不好看)",
    re.IGNORECASE,
)
PREVIOUS_IMAGE_REF_PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
    (2, re.compile(r"(?:再上一张|上上张|前前一张)图?", re.IGNORECASE)),
    (1, re.compile(r"(?:上一张|前一张)图?", re.IGNORECASE)),
    (0, re.compile(r"(?:当前这张|最新一张|这张图|这幅图)")),
)
NUMBERED_IMAGE_TOKEN_RE = re.compile(
    r"(?:图\s*([1-9])|第([123456789一二三四五六七八九两])张图)",
    re.IGNORECASE,
)
IMAGE_LOOKUP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:长啥样|什么样|是哪张|发(?:来)?看|看一下|看看|给我看)"),
    re.compile(r"^(?:那)?(?:再)?(?:上[一二两]?张|前[一二两]?张|这张)呢[？?]?$"),
)
# 用户直接要求看图但没有指定"这张"/"上一张"等引用，默认查看最新图片
IMAGE_SEND_REQUEST_RE = re.compile(
    r"(?:图发(?:我|来)|发(?:图|来)?(?:给我)?(?:看|瞅)|没看到|没收到图|图呢|配图呢|图片呢|"
    r"发一下图|看下图|我要看图|图片发一下|把图发)",
    re.IGNORECASE,
)
MAX_IMAGE_REFERENCE_HISTORY = 6

__all__ = [
    "IMG_MD_RE",
    "ABS_IMAGE_PATH_RE",
    "IMAGE_REFERENCE_RE",
    "NUMBERED_IMAGE_REFERENCE_RE",
    "IMAGE_EDIT_FOLLOWUP_RE",
    "IMAGE_EDIT_SUBJECT_CONTINUATION_RE",
    "IMAGE_REGENERATE_RE",
    "PREVIOUS_IMAGE_REF_PATTERNS",
    "NUMBERED_IMAGE_TOKEN_RE",
    "IMAGE_LOOKUP_PATTERNS",
    "MAX_IMAGE_REFERENCE_HISTORY",
    "fix_image_paths",
    "message_may_need_prior_images",
    "message_requests_image_edit",
    "message_requests_image_regenerate",
    "skill_requires_images",
    "mime_from_image_path",
    "load_images_from_paths",
    "recover_latest_generated_image",
    "append_image_reference_history",
    "recover_last_input_images",
    "recover_last_input_image_paths",
    "recover_recent_session_image_paths",
    "resolve_relative_image_reference_index",
    "number_token_to_index",
    "extract_numbered_image_reference_indexes",
    "resolve_numbered_input_image_paths",
    "resolve_relative_image_reference_path",
    "is_image_reference_lookup_message",
    "is_numbered_image_reference_edit_message",
    "extract_input_image_paths_from_text",
    "strip_inline_image_markdown",
]


# ── 函数 ──────────────────────────────────────────────────────


def fix_image_paths(text: str, known_paths: list[str] | None = None) -> str:
    """Validate image paths in markdown; fix fabricated paths using known real ones."""
    from pathlib import Path

    unused_real = list(known_paths or [])

    def _replace(m: re.Match[str]) -> str:
        alt, raw_path = m.group(1), m.group(2)
        fp = Path(raw_path)
        if fp.is_file():
            return m.group(0)

        for i, real in enumerate(unused_real):
            rp = Path(real)
            if rp.is_file():
                unused_real.pop(i)
                log.info("fix_image_path.known", original=raw_path, found=real)
                return f"![{alt}]({real})"

        stem = fp.stem
        hash_m = re.search(r"_([0-9a-f]{6,8})$", stem)
        if hash_m and fp.parent.is_dir():
            suffix = hash_m.group(0) + fp.suffix
            for candidate in fp.parent.iterdir():
                if candidate.name.endswith(suffix) and candidate.is_file():
                    log.info("fix_image_path.fuzzy", original=raw_path, found=str(candidate))
                    return f"![{alt}]({candidate})"

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

    return IMG_MD_RE.sub(_replace, text)


def message_may_need_prior_images(message: str) -> bool:
    """Detect whether the user is referring to a previously uploaded image."""
    return bool(IMAGE_REFERENCE_RE.search(message))


def message_requests_image_edit(message: str) -> bool:
    """Detect edit follow-ups that should continue from the latest output image."""
    if message_may_need_prior_images(message):
        return True
    if NUMBERED_IMAGE_REFERENCE_RE.search(message):
        return True
    if IMAGE_EDIT_SUBJECT_CONTINUATION_RE.search(message):
        return True
    return bool(IMAGE_EDIT_FOLLOWUP_RE.search(message))


def message_requests_image_regenerate(message: str) -> bool:
    """Detect reruns that should go back to the original input image set."""
    return bool(IMAGE_REGENERATE_RE.search(message))


def skill_requires_images(skills: list[Skill]) -> bool:
    """Return whether any active skill explicitly requires image inputs."""
    for skill in skills:
        guard = skill.param_guard
        if guard is None or not guard.enabled:
            continue
        for param in guard.params:
            if param.type.strip().lower() == "images":
                return True
    return False


def mime_from_image_path(path: Path) -> str:
    """Infer an image mime type from the local file suffix."""
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    return "image/jpeg"


def load_images_from_paths(paths: list[str]) -> list[ImageContent]:
    """Read local image paths into inline message payloads."""
    recovered: list[ImageContent] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            data = path.read_bytes()
        except OSError:
            continue
        recovered.append(ImageContent(
            mime=mime_from_image_path(path),
            data=base64.b64encode(data).decode("ascii"),
        ))
    return recovered


def recover_latest_generated_image(session: Session | None) -> list[ImageContent]:
    """Return only the latest generated image for edit-followup turns."""
    if session is None:
        return []
    metadata = session.metadata
    latest_generated = str(metadata.get("last_generated_image_path", "")).strip()
    if not latest_generated:
        return []
    return load_images_from_paths([latest_generated])


def append_image_reference_history(metadata: dict[str, object], paths: list[str]) -> bool:
    raw_history = metadata.get("image_reference_history", [])
    history: list[str] = []
    if isinstance(raw_history, list):
        history_items = cast(list[object], raw_history)
        history = [item.strip() for item in history_items if isinstance(item, str) and item.strip()]
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
    if len(history) > MAX_IMAGE_REFERENCE_HISTORY:
        history = history[:MAX_IMAGE_REFERENCE_HISTORY]
        changed = True
    if changed:
        metadata["image_reference_history"] = history
    return changed


def recover_last_input_images(session: Session | None) -> list[ImageContent]:
    """Return the last explicit input image set for regenerate-followup turns."""
    if session is None:
        return []
    metadata = session.metadata
    raw_paths = metadata.get("last_input_image_paths", [])
    if not isinstance(raw_paths, list):
        return []
    path_items = cast(list[object], raw_paths)
    paths = [item.strip() for item in path_items if isinstance(item, str) and item.strip()]
    return load_images_from_paths(paths)


def recover_last_input_image_paths(session: Session | None) -> list[str]:
    """Return the last explicit input image path strings for numbered-reference resolution."""
    if session is None:
        return []
    raw_paths = session.metadata.get("last_input_image_paths", [])
    if not isinstance(raw_paths, list):
        return []
    path_items = cast(list[object], raw_paths)
    return [item.strip() for item in path_items if isinstance(item, str) and item.strip()]


def recover_recent_session_image_paths(
    session: Session | None,
    *,
    limit: int = 4,
) -> list[str]:
    """Collect recent valid local image paths, preferring latest generated outputs."""
    if session is None:
        return []

    recovered: list[str] = []
    seen_paths: set[str] = set()

    metadata = session.metadata
    history_raw = metadata.get("image_reference_history", [])
    if isinstance(history_raw, list):
        for item in cast(list[object], history_raw):
            if not isinstance(item, str) or not item.strip():
                continue
            path = Path(item.strip()).expanduser()
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
        markdown_paths = [match.group(2).strip() for match in IMG_MD_RE.finditer(msg.content)]
        plain_paths = [match.group(1).strip() for match in ABS_IMAGE_PATH_RE.finditer(msg.content)]
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


def resolve_relative_image_reference_index(message: str) -> int | None:
    for index, pattern in PREVIOUS_IMAGE_REF_PATTERNS:
        if pattern.search(message):
            return index
    return None


def number_token_to_index(token: str) -> int | None:
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


def extract_numbered_image_reference_indexes(message: str) -> list[int]:
    indexes: list[int] = []
    seen: set[int] = set()
    for match in NUMBERED_IMAGE_TOKEN_RE.finditer(message):
        token = match.group(1) or match.group(2) or ""
        index = number_token_to_index(token)
        if index is None or index in seen:
            continue
        seen.add(index)
        indexes.append(index)
    return indexes


def resolve_numbered_input_image_paths(
    session: Session | None,
    message: str,
) -> list[str]:
    indexes = extract_numbered_image_reference_indexes(message)
    if not indexes:
        return []
    candidates = recover_last_input_image_paths(session)
    if not candidates:
        return []
    resolved: list[str] = []
    for index in indexes:
        if 0 <= index < len(candidates):
            resolved.append(candidates[index])
    return resolved


def resolve_relative_image_reference_path(
    session: Session | None,
    message: str,
) -> str:
    ref_index = resolve_relative_image_reference_index(message)
    if ref_index is None:
        return ""
    history = recover_recent_session_image_paths(session, limit=max(4, ref_index + 1))
    if ref_index >= len(history):
        return ""
    return history[ref_index]


def is_image_reference_lookup_message(
    message: str,
    *,
    locked_hooks: list[SkillHooks] | None = None,
    session: Session | None = None,
) -> bool:
    if locked_hooks:
        for _h in locked_hooks:
            exec_req = _h.is_execution_request(
                message, has_new_input_images=False, session=session,
            )
            if exec_req:
                return False
    if resolve_relative_image_reference_index(message) is not None:
        if any(pattern.search(message) for pattern in IMAGE_LOOKUP_PATTERNS):
            return True
    # 兜底：用户直接要求发图（"图发我看一下"/"没看到"等），默认查看最新图
    if IMAGE_SEND_REQUEST_RE.search(message):
        return True
    return False


def is_numbered_image_reference_edit_message(text: str) -> bool:
    """Return whether the message is describing an edit across numbered image refs."""
    if not NUMBERED_IMAGE_REFERENCE_RE.search(text):
        return False
    return message_requests_image_edit(text)


def extract_input_image_paths_from_text(
    text: str,
    *,
    limit: int = 8,
) -> list[str]:
    """Extract unique local image paths from the current user message text."""
    extracted: list[str] = []
    seen_paths: set[str] = set()
    markdown_paths = [match.group(2).strip() for match in IMG_MD_RE.finditer(text)]
    plain_paths = [match.group(1).strip() for match in ABS_IMAGE_PATH_RE.finditer(text)]
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


def strip_inline_image_markdown(text: str) -> str:
    """Remove appended local image markdown from a user prompt string."""
    stripped = IMG_MD_RE.sub("", text)
    stripped = ABS_IMAGE_PATH_RE.sub("", stripped)
    stripped = stripped.replace("(用户发送了图片)", "")
    return stripped.strip()
