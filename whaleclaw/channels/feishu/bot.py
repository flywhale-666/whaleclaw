"""Feishu bot — core message handling logic."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from whaleclaw.channels.feishu.allowlist import FeishuAllowList
from whaleclaw.channels.feishu.card import FeishuCard
from whaleclaw.channels.feishu.config import FeishuConfig
from whaleclaw.channels.feishu.dedup import MessageDedup
from whaleclaw.channels.feishu.mention import is_bot_mentioned, strip_bot_mention
from whaleclaw.config.loader import set_default_agent_model
from whaleclaw.config.paths import WHALECLAW_HOME
from whaleclaw.media.image_resize import resize_image_long_edge
from whaleclaw.plugins.evomap.bridge import build_memory_hint_from_hook_data
from whaleclaw.plugins.hooks import HookContext, HookManager, HookPoint
from whaleclaw.providers.base import ImageContent
from whaleclaw.providers.nvidia import NvidiaProvider
from whaleclaw.utils.log import get_logger

_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_FILE_RE = re.compile(r"(?<!!)\[([^\]]+)\]\((/[^)]+)\)")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
_SUBMIT_RE = re.compile(r"(?:^|[\s，。,\.!！?？])提交\s*$", re.IGNORECASE)
_FILE_EXTS = {
    ".txt",
    ".md",
    ".json",
    ".log",
    ".pptx",
    ".ppt",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".csv",
    ".zip",
    ".tar",
    ".gz",
    ".mp3",
    ".wav",
    ".aif",
    ".aiff",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".flac",
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}
_BOLD_PATH_RE = re.compile(
    r"\*\*(/[^*\s]+\.(?:"
    + "|".join(ext.lstrip(".") for ext in sorted(_FILE_EXTS))
    + r"))\*\*",
    re.MULTILINE,
)
_BARE_PATH_RE = re.compile(
    r"(?:^|[\s`：:])(/[^`\s]+\.(?:"
    + "|".join(ext.lstrip(".") for ext in sorted(_FILE_EXTS))
    + r"))(?:[\s`]|$)",
    re.MULTILINE,
)

if TYPE_CHECKING:
    from whaleclaw.config.schema import WhaleclawConfig
    from whaleclaw.memory.manager import MemoryManager
    from whaleclaw.sessions.group_compressor import SessionGroupCompressor
    from whaleclaw.sessions.manager import SessionManager
    from whaleclaw.tools.registry import ToolRegistry

log = get_logger(__name__)
_FEISHU_MEDIA_DIR = WHALECLAW_HOME / "media" / "feishu"
_PENDING_IMAGE_PATHS_KEY = "feishu_pending_image_paths"
_PENDING_PROMPT_KEY = "feishu_pending_prompt"
_PROCESSING_REACTION_EMOJIS = ("THINKING", "HOURGLASS")
def _image_buffer_skill_ids() -> frozenset[str]:
    """从已加载技能的 hooks 动态获取需要图片缓冲的技能 ID。"""
    from whaleclaw.skills.hooks import get_skill_hooks
    from whaleclaw.skills.manager import SkillManager
    ids: set[str] = set()
    for skill in SkillManager().discover():
        hooks = get_skill_hooks(skill)
        if hooks is not None and hooks.image_buffer_enabled:
            ids.add(skill.id)
    return frozenset(ids)


@dataclass(slots=True)
class _InboundImage:
    """Normalized inbound Feishu image for both agent input and local replay."""

    content: ImageContent
    path: str


def _format_exception_text(exc: Exception) -> str:
    """Return a readable exception text even when ``str(exc)`` is empty."""
    msg = str(exc).strip()
    return msg if msg else exc.__class__.__name__


def format_exception_text(exc: Exception) -> str:
    """Public wrapper for exception text formatting."""
    return _format_exception_text(exc)


def processing_reaction_emojis() -> tuple[str, ...]:
    """Public accessor for reaction fallback order."""
    return _PROCESSING_REACTION_EMOJIS


def _to_object_dict(value: object) -> dict[str, object] | None:
    """Normalize loose JSON/mapping payloads into a string-keyed dict."""
    if not isinstance(value, dict):
        return None
    raw_dict = cast(dict[object, object], value)
    return {str(key): item for key, item in raw_dict.items()}


def _parse_json_object(content_str: object) -> dict[str, object] | None:
    """Parse a JSON string and return a normalized object dict."""
    if not isinstance(content_str, str):
        return None
    try:
        parsed = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return None
    return _to_object_dict(parsed)


def _iter_post_elements(content: Mapping[str, object]) -> list[dict[str, object]]:
    """Flatten Feishu post blocks into a list of normalized element dicts."""
    raw_blocks = content.get("content")
    if not isinstance(raw_blocks, list):
        return []
    elements: list[dict[str, object]] = []
    for raw_line in cast(list[object], raw_blocks):
        if not isinstance(raw_line, list):
            continue
        for raw_elem in cast(list[object], raw_line):
            elem = _to_object_dict(raw_elem)
            if elem is not None:
                elements.append(elem)
    return elements


def _copy_metadata(value: object) -> dict[str, Any]:
    """Create a typed metadata copy safe for session updates."""
    if not isinstance(value, dict):
        return {}
    raw_dict = cast(dict[object, Any], value)
    return {str(key): item for key, item in raw_dict.items()}


def _string_list(value: object) -> list[str]:
    """Coerce a JSON-ish list into a filtered string list."""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in cast(list[object], value):
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _int_or_none(value: object) -> int | None:
    """Parse an integer from loose JSON values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


class _FeishuClientLike(Protocol):
    """Minimal client protocol used by the bot and tests."""

    async def reply_message(
        self, message_id: str, msg_type: str, content: str
    ) -> dict[str, Any]: ...

    async def create_message_reaction(
        self,
        message_id: str,
        emoji_type: str,
    ) -> dict[str, Any]: ...

    async def delete_message_reaction(
        self,
        message_id: str,
        reaction_id: str,
    ) -> dict[str, Any]: ...

    async def download_resource(
        self, message_id: str, file_key: str, *, resource_type: str = "file"
    ) -> bytes: ...

    async def upload_image(self, image: bytes, image_type: str = "message") -> str: ...

    async def upload_file(self, file: bytes, filename: str, file_type: str) -> str: ...

    async def send_message(
        self,
        receive_id: str,
        msg_type: str,
        content: str,
        receive_id_type: str = "open_id",
    ) -> dict[str, Any]: ...


class FeishuBot:
    """Process incoming Feishu messages and route to the Agent."""

    def __init__(
        self,
        client: _FeishuClientLike,
        config: FeishuConfig,
        allowlist: FeishuAllowList | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._dedup = MessageDedup()
        self._allowlist = allowlist or FeishuAllowList()
        self._pairing_codes: dict[str, str] = {}
        self._bot_open_id = ""
        self._whaleclaw_config: WhaleclawConfig | None = None
        self._session_manager: SessionManager | None = None
        self._tool_registry: ToolRegistry | None = None
        self._memory_manager: MemoryManager | None = None
        self._hook_manager: HookManager | None = None
        self._group_compressor: SessionGroupCompressor | None = None
        self._compression_ready_fn: Callable[[], bool] | None = None

    def set_bot_open_id(self, bot_open_id: str) -> None:
        self._bot_open_id = bot_open_id

    def bind_agent(
        self,
        config: WhaleclawConfig,
        session_manager: SessionManager,
        registry: ToolRegistry,
        memory_manager: MemoryManager | None = None,
        hook_manager: HookManager | None = None,
        group_compressor: SessionGroupCompressor | None = None,
        compression_ready_fn: Callable[[], bool] | None = None,
    ) -> None:
        """Inject Agent dependencies so handle_message can run the full loop."""
        self._whaleclaw_config = config
        self._session_manager = session_manager
        self._tool_registry = registry
        self._memory_manager = memory_manager
        self._hook_manager = hook_manager
        self._group_compressor = group_compressor
        self._compression_ready_fn = compression_ready_fn

    async def handle_event(
        self, event_type: str, body: dict[str, Any]
    ) -> None:
        """Dispatch an event to the appropriate handler."""
        if event_type == "im.message.receive_v1":
            event = body.get("event", {})
            await self.handle_message(event)

    async def handle_message(self, event: dict[str, Any]) -> None:
        """Process a received message event."""
        message = event.get("message", {})
        msg_id = message.get("message_id", "")
        msg_type = message.get("message_type", "text")
        chat_type = message.get("chat_type", "")
        sender = event.get("sender", {}).get("sender_id", {})
        open_id = sender.get("open_id", "")

        if self._dedup.is_duplicate(msg_id):
            return
        self._dedup.mark(msg_id)

        raw_text = self.extract_text(message).strip()
        inbound_images = (
            await self._extract_images(message)
            if msg_type in ("image", "post")
            else []
        )
        images = [item.content for item in inbound_images]
        image_markdown = self._build_image_markdown(inbound_images)
        file_path = await self._extract_file(message) if msg_type == "file" else None
        if not raw_text and not images and not file_path:
            return
        if file_path:
            if raw_text:
                raw_text = f"{raw_text}\n\n📎 用户发送了文件:\n{file_path}"
            else:
                raw_text = f"(用户发送了文件)\n{file_path}"
        if images:
            if raw_text:
                text = f"{raw_text}\n\n{image_markdown}"
            elif image_markdown:
                text = image_markdown
            else:
                text = "(用户发送了一张图片)"
        else:
            text = raw_text

        if chat_type == "group":
            group_cfg = self._config.groups.get(
                message.get("chat_id", "")
            )
            need_mention = (
                group_cfg.require_mention if group_cfg else True
            )
            if need_mention and not is_bot_mentioned(message, self._bot_open_id):
                return
            text = strip_bot_mention(text, "")
            raw_text = strip_bot_mention(raw_text, "").strip()

        not_allowed = (
            chat_type == "p2p"
            and self._config.dm_policy != "open"
            and not self._allowlist.is_allowed(open_id)
        )
        if not_allowed:
            if self._config.dm_policy == "closed":
                return
            await self._send_pairing_prompt(open_id, msg_id)
            return

        log.info(
            "feishu.message",
            message_id=msg_id,
            chat_type=chat_type,
            open_id=open_id,
            msg_type=msg_type,
            text_len=len(text),
            text_preview=(" ".join(text.split())[:80]),
            images=len(images),
            has_file=bool(file_path),
        )

        session = None
        if self._session_manager is not None:
            session = await self._session_manager.get_or_create("feishu", open_id)
        if session is not None and not file_path and not raw_text.startswith("/"):
            buffer_result = await self._handle_pending_image_submission(
                session,
                raw_text=raw_text,
                inbound_images=inbound_images,
            )
            if buffer_result is not None:
                reply_text, run_text, run_images = buffer_result
                if reply_text:
                    await self._client.reply_message(
                        msg_id,
                        "text",
                        json.dumps({"text": reply_text}, ensure_ascii=False),
                    )
                if run_text is None:
                    return
                text = run_text
                images = run_images

        reaction_state = await self._add_processing_reaction(msg_id)
        try:
            await self._run_agent_and_reply(text, open_id, msg_id, images=images)
        finally:
            await self._remove_processing_reaction(msg_id, reaction_state)

    async def _handle_pending_image_submission(
        self,
        session: Any,
        *,
        raw_text: str,
        inbound_images: Sequence[_InboundImage],
    ) -> tuple[str | None, str | None, list[ImageContent] | None] | None:
        """Buffer Feishu images until the user explicitly submits the request.

        Only activates when the session has a skill that requires image
        buffering (e.g. xiaohongshu_publish) or there are already buffered
        images from a previous turn.
        """
        if self._session_manager is None:
            return None

        metadata = _copy_metadata(session.metadata)
        pending_paths = _string_list(metadata.get(_PENDING_IMAGE_PATHS_KEY))

        if not pending_paths:
            locked = _string_list(metadata.get("locked_skill_ids"))
            if not _image_buffer_skill_ids().intersection(s.lower() for s in locked):
                return None
        pending_prompt = str(metadata.get(_PENDING_PROMPT_KEY, "")).strip()
        stripped_text, submitted = self._strip_submit_suffix(raw_text)

        if inbound_images:
            for item in inbound_images:
                if item.path not in pending_paths:
                    pending_paths.append(item.path)

        if inbound_images and stripped_text and not submitted:
            buffered = self._load_buffered_images(pending_paths)
            metadata.pop(_PENDING_IMAGE_PATHS_KEY, None)
            metadata.pop(_PENDING_PROMPT_KEY, None)
            await self._session_manager.update_metadata(session, metadata)
            markdown = self._build_image_markdown(buffered)
            full_text = f"{stripped_text}\n\n{markdown}" if markdown else stripped_text
            return (None, full_text, [item.content for item in buffered])

        if inbound_images and not stripped_text and not submitted:
            metadata[_PENDING_IMAGE_PATHS_KEY] = pending_paths
            metadata.pop(_PENDING_PROMPT_KEY, None)
            await self._session_manager.update_metadata(session, metadata)
            already = self._count_skill_images(metadata)
            start = already + len(pending_paths) - len(inbound_images) + 1
            labels = self._format_image_range(start, already + len(pending_paths))
            locked = _string_list(metadata.get("locked_skill_ids"))
            hint = self._build_image_buffer_hint(locked, labels)
            return (hint, None, None)

        if pending_paths and stripped_text and not submitted:
            # "任务完成"等解锁指令不要合并 pending 图片，直接透传给 agent
            if self._is_unlock_command(stripped_text):
                metadata.pop(_PENDING_IMAGE_PATHS_KEY, None)
                metadata.pop(_PENDING_PROMPT_KEY, None)
                await self._session_manager.update_metadata(session, metadata)
                return None
            buffered = self._load_buffered_images(pending_paths)
            metadata.pop(_PENDING_IMAGE_PATHS_KEY, None)
            metadata.pop(_PENDING_PROMPT_KEY, None)
            await self._session_manager.update_metadata(session, metadata)
            markdown = self._build_image_markdown(buffered)
            full_text = f"{stripped_text}\n\n{markdown}" if markdown else stripped_text
            return (None, full_text, [item.content for item in buffered])

        if pending_paths and submitted:
            prompt = stripped_text or pending_prompt
            if not prompt:
                metadata[_PENDING_IMAGE_PATHS_KEY] = pending_paths
                metadata.pop(_PENDING_PROMPT_KEY, None)
                await self._session_manager.update_metadata(session, metadata)
                return ("已收到图片。请先发送提示词，再回复“提交”。", None, None)
            buffered = self._load_buffered_images(pending_paths)
            metadata.pop(_PENDING_IMAGE_PATHS_KEY, None)
            metadata.pop(_PENDING_PROMPT_KEY, None)
            await self._session_manager.update_metadata(session, metadata)
            markdown = self._build_image_markdown(buffered)
            full_text = f"{prompt}\n\n{markdown}" if markdown else prompt
            return (None, full_text, [item.content for item in buffered])

        return None

    _UNLOCK_RE = re.compile(
        r"^\s*(?:任务完成|完成任务|任务结束|结束任务|完成了?|结束了?|可以了?|取消|解锁)\s*[!！。.]*$",
        re.IGNORECASE,
    )

    @classmethod
    def _is_unlock_command(cls, text: str) -> bool:
        """判断文字是否是技能解锁指令，不应与 pending 图片合并。"""
        return bool(cls._UNLOCK_RE.search(text.strip()))

    @staticmethod
    def _strip_submit_suffix(text: str) -> tuple[str, bool]:
        stripped = text.strip()
        if not stripped:
            return ("", False)
        if not _SUBMIT_RE.search(stripped):
            return (stripped, False)
        cleaned = _SUBMIT_RE.sub("", stripped).strip("，。, .!！?？")
        return (cleaned.strip(), True)

    def _load_buffered_images(self, image_paths: list[str]) -> list[_InboundImage]:
        """Rehydrate buffered image paths into agent-ready image payloads."""
        loaded: list[_InboundImage] = []
        for raw_path in image_paths:
            path = Path(raw_path).expanduser()
            if not path.is_file():
                log.warning(
                    "feishu.buffered_image_missing",
                    path=raw_path,
                    resolved=str(path),
                )
                continue
            try:
                data = path.read_bytes()
            except OSError:
                log.warning("feishu.buffered_image_read_failed", path=str(path))
                continue
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            loaded.append(_InboundImage(
                ImageContent(
                    mime=mime,
                    data=base64.b64encode(data).decode("ascii"),
                ),
                str(path),
            ))
        log.info(
            "feishu.buffered_images_loaded",
            requested=len(image_paths),
            loaded=len(loaded),
            paths=image_paths,
        )
        return loaded

    @staticmethod
    def _count_skill_images(metadata: dict[str, Any]) -> int:
        """统计 skill_param_state 中已确认的图片数（不含当前 pending buffer）。"""
        sps: object = metadata.get("skill_param_state")
        if not isinstance(sps, dict):
            return 0
        typed_sps = cast(dict[str, object], sps)
        for state_val in typed_sps.values():
            if not isinstance(state_val, dict):
                continue
            typed_state = cast(dict[str, object], state_val)
            raw_img: object = typed_state.get("images")
            if isinstance(raw_img, int) and raw_img > 0:
                return raw_img
        return 0

    @staticmethod
    def _format_image_range(start: int, end: int) -> str:
        if end <= start:
            return f"图{start}"
        return f"图{start}到图{end}"

    async def _add_processing_reaction(self, message_id: str) -> tuple[str, str] | None:
        """Add a lightweight processing reaction to the user message."""
        for emoji_type in _PROCESSING_REACTION_EMOJIS:
            try:
                resp = await self._client.create_message_reaction(message_id, emoji_type)
                reaction_id = str(resp.get("data", {}).get("reaction_id", "")).strip()
                if reaction_id:
                    return (reaction_id, emoji_type)
            except Exception as exc:
                log.debug(
                    "feishu.reaction_create_failed",
                    message_id=message_id,
                    emoji_type=emoji_type,
                    error=_format_exception_text(exc),
                )
        return None

    async def _remove_processing_reaction(
        self,
        message_id: str,
        reaction_state: tuple[str, str] | None,
    ) -> None:
        """Remove the temporary processing reaction after final reply."""
        if not reaction_state:
            return
        reaction_id, emoji_type = reaction_state
        try:
            await self._client.delete_message_reaction(message_id, reaction_id)
        except Exception as exc:
            log.debug(
                "feishu.reaction_delete_failed",
                message_id=message_id,
                reaction_id=reaction_id,
                emoji_type=emoji_type,
                error=_format_exception_text(exc),
            )

    async def _run_agent_and_reply(
        self,
        text: str,
        peer_id: str,
        reply_to_msg_id: str,
        *,
        images: list[ImageContent] | None = None,
    ) -> None:
        """Run Agent and send plain text/image/file replies to Feishu."""
        if not self._whaleclaw_config or not self._session_manager or not self._tool_registry:
            await self._client.reply_message(
                reply_to_msg_id, "text", json.dumps({"text": text}, ensure_ascii=False)
            )
            return

        from whaleclaw.agent.loop import run_agent
        from whaleclaw.gateway.protocol import make_message, make_status
        from whaleclaw.gateway.ws import broadcast_all
        from whaleclaw.providers.router import ModelRouter

        if self._compression_ready_fn is not None and not self._compression_ready_fn():
            await self._client.reply_message(
                reply_to_msg_id,
                "text",
                json.dumps({"text": "会话压缩中，请稍后再试。"}, ensure_ascii=False),
            )
            return

        session = await self._session_manager.get_or_create("feishu", peer_id)
        cmd_reply = await self._handle_command(text, session)
        if cmd_reply is not None:
            await self._client.reply_message(
                reply_to_msg_id,
                "text",
                json.dumps({"text": cmd_reply}, ensure_ascii=False),
            )
            status_msg = make_status(session.id, cmd_reply)
            status_msg.payload["model"] = session.model
            await broadcast_all(status_msg)
            return

        await self._session_manager.add_message(session, "user", text)

        await broadcast_all(make_message(session.id, f"📨 **飞书** `{peer_id[:8]}…`:\n{text}"))

        router = ModelRouter(self._whaleclaw_config.models)
        extra_memory = ""
        if self._hook_manager is not None:
            try:
                hook_out = await self._hook_manager.run(
                    HookPoint.BEFORE_MESSAGE,
                    HookContext(
                        hook=HookPoint.BEFORE_MESSAGE,
                        session_id=session.id,
                        data={"message": text, "channel": "feishu", "peer_id": peer_id},
                    ),
                )
                if hook_out.proceed:
                    extra_memory = build_memory_hint_from_hook_data(hook_out.data)
            except Exception:
                pass

        try:
            reply = await run_agent(
                message=text,
                session_id=session.id,
                config=self._whaleclaw_config,
                session=session,
                router=router,
                registry=self._tool_registry,
                images=images or None,
                session_manager=self._session_manager,
                session_store=self._session_manager.store,
                memory_manager=self._memory_manager,
                extra_memory=extra_memory,
                trigger_event_id=reply_to_msg_id,
                trigger_text_preview=text,
                group_compressor=self._group_compressor,
            )
            log.info("feishu.agent_reply", reply_len=len(reply), preview=reply[:200])
        except Exception as exc:
            if self._hook_manager is not None:
                with suppress(Exception):
                    await self._hook_manager.run(
                        HookPoint.ON_ERROR,
                        HookContext(
                            hook=HookPoint.ON_ERROR,
                            session_id=session.id,
                            data={"error": str(exc), "message": text, "channel": "feishu"},
                        ),
                    )
            error_text = _format_exception_text(exc)
            log.exception("feishu.agent_error", error=error_text, model=session.model)
            await self._client.reply_message(
                reply_to_msg_id,
                "text",
                json.dumps({"text": f"处理失败: {error_text}"}, ensure_ascii=False),
            )
            await broadcast_all(make_message(session.id, f"❌ **飞书处理失败**: {error_text}"))
            return

        if not reply.strip():
            await self._client.reply_message(
                reply_to_msg_id,
                "text",
                json.dumps(
                    {"text": "任务执行中但未返回结果，请稍后重试或查看 WebChat。"},
                    ensure_ascii=False,
                ),
            )
            return

        try:
            await self._session_manager.add_message(session, "assistant", reply)
            text_content, image_paths, file_paths = self._prepare_reply_payload(reply)
            if text_content:
                await self._client.reply_message(
                    reply_to_msg_id,
                    "text",
                    json.dumps({"text": text_content}, ensure_ascii=False),
                )
            for image_path in image_paths:
                await self._send_image_to_peer(peer_id, image_path)
            for fp in file_paths:
                await self._send_file_to_peer(peer_id, fp)
            await broadcast_all(make_message(session.id, f"🤖 **飞书回复**:\n{reply}"))
        except Exception as exc:
            error_text = _format_exception_text(exc)
            log.exception("feishu.reply_failed", error=error_text)
            with suppress(Exception):
                await self._client.reply_message(
                    reply_to_msg_id,
                    "text",
                    json.dumps(
                        {"text": reply or f"回复发送失败: {error_text}"},
                        ensure_ascii=False,
                    ),
                )

    async def _handle_command(self, text: str, session: Any) -> str | None:
        """Handle Feishu slash commands for model switching."""
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/help", "/h"}:
            return (
                "可用命令:\n"
                "/models - 查看可切换模型\n"
                "/model <序号|provider/model> - 切换模型\n"
                "/model - 查看当前模型\n"
                "/multi status - 查看多Agent状态\n"
                "/multi on|off - 会话级启用/禁用多Agent\n"
                "/multi mode parallel|serial - 设置会话协作模式\n"
                "/multi rounds <1-10> - 设置会话最大回合\n"
                "/think 已禁用（按你的配置不启用）"
            )

        if cmd in {"/think", "/thinking"}:
            return "当前通道已禁用 think 模式切换。"

        models = self._list_selectable_models()

        if cmd in {"/models", "/lsmodels"}:
            if not models:
                return "当前没有可切换模型，请先在配置中启用并验证模型。"
            lines = ["可切换模型:"]
            for i, mid in enumerate(models, start=1):
                marker = " (当前)" if mid == session.model else ""
                lines.append(f"{i}. {mid}{marker}")
            lines.append("发送 /model <序号> 或 /model <provider/model> 切换。")
            return "\n".join(lines)

        if cmd == "/model":
            if not arg:
                return "当前模型: " + session.model + "\n发送 /models 查看可选模型。"

            target = arg
            if arg.isdigit():
                idx = int(arg)
                if idx < 1 or idx > len(models):
                    return f"序号无效: {arg}\n发送 /models 查看可选模型。"
                target = models[idx - 1]

            if target not in models:
                return f"模型不可用: {target}\n发送 /models 查看可选模型。"

            if self._session_manager is None:
                return "模型切换不可用：会话管理器未初始化。"
            await self._session_manager.update_model(session, target)
            if self._whaleclaw_config is not None:
                self._whaleclaw_config.agent.model = target
            try:
                set_default_agent_model(target)
            except Exception as exc:
                err = _format_exception_text(exc)
                log.warning(
                    "feishu.default_model_persist_failed",
                    model=target,
                    error=err,
                )
                return f"已切换模型到: {target}\n默认模型保存失败: {err}"
            return f"已切换模型到: {target}"

        if cmd == "/multi":
            if self._session_manager is None:
                return "多Agent命令不可用：会话管理器未初始化。"

            raw_parts = arg.split() if arg else []
            action = raw_parts[0].lower() if raw_parts else "status"

            if action in {"status", "st", "s"}:
                return self._format_multi_agent_status(session)

            metadata = _copy_metadata(session.metadata)

            if action in {"on", "enable"}:
                metadata["multi_agent_enabled"] = True
                await self._session_manager.update_metadata(session, metadata)
                return "已开启本会话多Agent。发送 /multi status 查看当前状态。"

            if action in {"off", "disable"}:
                metadata["multi_agent_enabled"] = False
                await self._session_manager.update_metadata(session, metadata)
                return "已关闭本会话多Agent。发送 /multi status 查看当前状态。"

            if action == "mode":
                if len(raw_parts) < 2:
                    return "用法: /multi mode parallel|serial"
                mode = raw_parts[1].strip().lower()
                if mode not in {"parallel", "serial"}:
                    return "模式无效，仅支持 parallel 或 serial。"
                metadata["multi_agent_mode"] = mode
                await self._session_manager.update_metadata(session, metadata)
                mode_cn = "并行" if mode == "parallel" else "串行"
                return f"已设置本会话多Agent模式: {mode_cn}（{mode}）。"

            if action in {"round", "rounds"}:
                if len(raw_parts) < 2:
                    return "用法: /multi rounds <1-10>"
                value = raw_parts[1].strip()
                if not value.isdigit():
                    return f"回合数无效: {value}，请输入 1-10 的整数。"
                rounds = int(value)
                if rounds < 1 or rounds > 10:
                    return f"回合数超出范围: {rounds}，允许范围为 1-10。"
                metadata["multi_agent_max_rounds"] = rounds
                await self._session_manager.update_metadata(session, metadata)
                return f"已设置本会话多Agent最大回合: {rounds}。"

            return (
                "用法:\n"
                "/multi status\n"
                "/multi on\n"
                "/multi off\n"
                "/multi mode parallel|serial\n"
                "/multi rounds <1-10>"
            )

        return None

    def _format_multi_agent_status(self, session: Any) -> str:
        """Format global/session/effective multi-agent status for command reply."""
        global_enabled = False
        global_mode = "parallel"
        global_rounds = 1
        if self._whaleclaw_config is not None:
            raw = self._whaleclaw_config.plugins.get("multi_agent", {})
            global_enabled = bool(raw.get("enabled", False))
            mode_raw = str(raw.get("mode", "parallel")).strip().lower()
            global_mode = mode_raw if mode_raw in {"parallel", "serial"} else "parallel"
            global_rounds = _int_or_none(raw.get("max_rounds")) or 1
        global_rounds = max(1, min(global_rounds, 10))

        metadata = _copy_metadata(session.metadata)
        enabled_override = metadata.get("multi_agent_enabled")
        mode_override = str(metadata.get("multi_agent_mode", "")).strip().lower()
        rounds_override = _int_or_none(metadata.get("multi_agent_max_rounds"))
        has_enabled_override = isinstance(enabled_override, bool)
        has_mode_override = mode_override in {
            "parallel",
            "serial",
        }
        has_rounds_override = rounds_override is not None

        effective_enabled = (
            enabled_override
            if has_enabled_override
            else global_enabled
        )
        effective_mode = (
            mode_override
            if has_mode_override
            else global_mode
        )
        effective_rounds = global_rounds
        if has_rounds_override:
            effective_rounds = rounds_override
        effective_rounds = max(1, min(effective_rounds, 10))

        mode_cn = "并行" if effective_mode == "parallel" else "串行"
        global_line = (
            f"- 全局: {'开启' if global_enabled else '关闭'}"
            f" | 模式={global_mode} | 回合={global_rounds}"
        )
        session_line = (
            f"- 会话覆盖: enabled={metadata.get('multi_agent_enabled', '(未设置)')}, "
            f"mode={metadata.get('multi_agent_mode', '(未设置)')}, "
            f"rounds={metadata.get('multi_agent_max_rounds', '(未设置)')}"
        )
        effective_line = (
            f"- 当前生效: {'开启' if effective_enabled else '关闭'} | "
            f"模式={mode_cn}（{effective_mode}） | 回合={effective_rounds}"
        )
        return (
            "多Agent状态:\n"
            f"{global_line}\n"
            f"{session_line}\n"
            f"{effective_line}"
        )

    def _list_selectable_models(self) -> list[str]:
        """Return verified and configured model IDs."""
        if self._whaleclaw_config is None:
            return []

        providers_cfg = self._whaleclaw_config.models
        result: list[str] = []
        for pname in providers_cfg.all_provider_names():
            pcfg = providers_cfg.get_provider(pname)
            has_auth = bool(pcfg.api_key) or (
                getattr(pcfg, "auth_mode", "api_key") == "oauth" and bool(pcfg.oauth_access)
            )
            if not has_auth:
                continue
            for cm in pcfg.configured_models:
                if not cm.verified:
                    continue
                if (
                    pname == "openai"
                    and pcfg.auth_mode == "oauth"
                    and not cm.id.lower().startswith("gpt-5")
                ):
                    continue
                if pname == "nvidia" and not NvidiaProvider.model_supports_tools(cm.id):
                    continue
                result.append(f"{pname}/{cm.id}")
        return result

    def _prepare_reply_payload(self, reply: str) -> tuple[str, list[Path], list[Path]]:
        """Extract text/image/file payloads from agent reply."""
        image_paths: list[Path] = []
        seen_image_paths: set[str] = set()
        for match in _IMG_RE.finditer(reply):
            path = match.group(2)
            local = Path(path)
            if local.is_file() and local.suffix.lower() in _IMAGE_EXTS:
                image_paths.append(local)
                seen_image_paths.add(path)

        file_paths: list[Path] = []
        file_replacements: list[tuple[str, str]] = []
        seen_paths: set[str] = set()

        for match in _FILE_RE.finditer(reply):
            name, path = match.group(1), match.group(2)
            local = Path(path)
            if not local.is_file() or path in seen_paths:
                continue
            ext = local.suffix.lower()
            if ext in _IMAGE_EXTS and path not in seen_image_paths:
                seen_image_paths.add(path)
                image_paths.append(local)
                file_replacements.append((match.group(0), ""))
            elif ext in _FILE_EXTS:
                seen_paths.add(path)
                file_paths.append(local)
                file_replacements.append((match.group(0), f"📎 {name}"))

        for match in _BOLD_PATH_RE.finditer(reply):
            path = match.group(1)
            local = Path(path)
            if local.is_file() and local.suffix.lower() in _FILE_EXTS and path not in seen_paths:
                seen_paths.add(path)
                file_paths.append(local)
                file_replacements.append((match.group(0), f"📎 {local.name}"))

        for match in _BARE_PATH_RE.finditer(reply):
            path = match.group(1)
            local = Path(path)
            if local.is_file() and path not in seen_paths:
                seen_paths.add(path)
                file_paths.append(local)
                file_replacements.append((path, f"📎 {local.name}"))

        log.info("feishu.reply_files", count=len(file_paths), paths=[str(p) for p in file_paths])

        clean_text = reply
        for match in _IMG_RE.finditer(reply):
            clean_text = clean_text.replace(match.group(0), "")
        for md_str, label in file_replacements:
            clean_text = clean_text.replace(md_str, label)
        clean_text = clean_text.replace("\r\n", "\n").replace("\r", "\n")
        clean_text = re.sub(r"(?m)^\s*\d+\.\s*$\n?", "", clean_text)
        clean_text = re.sub(r"\n\s*\n+", "\n", clean_text)
        return clean_text.strip(), image_paths, file_paths

    def prepare_reply_payload(self, reply: str) -> tuple[str, list[Path], list[Path]]:
        """Public wrapper for reply payload extraction."""
        return self._prepare_reply_payload(reply)

    async def _send_image_to_peer(self, peer_id: str, image_path: Path) -> None:
        """Upload a local image to Feishu and send as image message."""
        try:
            data = image_path.read_bytes()
            image_key = await self._client.upload_image(data)
            if image_key:
                await self._client.send_message(
                    peer_id,
                    "image",
                    json.dumps({"image_key": image_key}, ensure_ascii=False),
                )
                log.info("feishu.image_sent", name=image_path.name)
            else:
                log.warning("feishu.image_upload_no_key", path=str(image_path))
        except Exception:
            log.exception("feishu.image_send_failed", path=str(image_path))

    async def _send_file_to_peer(self, peer_id: str, file_path: Path) -> None:
        """Upload a local file to Feishu and send as a file message."""
        try:
            log.info("feishu.file_uploading", name=file_path.name, size=file_path.stat().st_size)
            data = file_path.read_bytes()
            file_key = await self._client.upload_file(data, file_path.name, "stream")
            log.info("feishu.file_uploaded", file_key=file_key)
            if file_key:
                content = json.dumps({"file_key": file_key})
                await self._client.send_message(peer_id, "file", content)
                log.info("feishu.file_sent", name=file_path.name)
            else:
                log.warning("feishu.file_upload_no_key", name=file_path.name)
        except Exception:
            log.exception("feishu.file_send_failed", path=str(file_path))

    @staticmethod
    def extract_text(message: dict[str, Any]) -> str:
        """Extract plain text from a Feishu message."""
        msg_type = message.get("message_type", "text")
        content = _parse_json_object(message.get("content", "{}"))
        if content is None:
            return ""

        if msg_type == "text":
            text = content.get("text", "")
            return text if isinstance(text, str) else ""
        if msg_type == "post":
            parts: list[str] = []
            for elem in _iter_post_elements(content):
                if elem.get("tag") == "text":
                    text = elem.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return ""

    async def _extract_images(self, message: dict[str, Any]) -> list[_InboundImage]:
        """Download incoming Feishu image(s) and convert to ImageContent.

        Supports both pure image messages (msg_type=image) and rich-text
        posts (msg_type=post) that embed ``img`` elements with image_key.
        """
        msg_id = message.get("message_id", "")
        msg_type = message.get("message_type", "")
        content = _parse_json_object(message.get("content", "{}"))
        if content is None:
            return []

        image_keys: list[str] = []
        if msg_type == "image":
            key = content.get("image_key")
            if isinstance(key, str) and key:
                image_keys.append(key)
        elif msg_type == "post":
            for elem in _iter_post_elements(content):
                image_key = elem.get("image_key")
                if elem.get("tag") == "img" and isinstance(image_key, str) and image_key:
                    image_keys.append(image_key)
            raw_keys = content.get("image_keys")
            if isinstance(raw_keys, list):
                for raw_key in cast(list[object], raw_keys):
                    if isinstance(raw_key, str) and raw_key and raw_key not in image_keys:
                        image_keys.append(raw_key)

        if not image_keys:
            return []

        results: list[_InboundImage] = []
        for key in image_keys[:4]:
            try:
                data = await self._client.download_resource(
                    msg_id, key, resource_type="image"
                )
            except Exception:
                log.exception("feishu.image_download_failed", message_id=msg_id, image_key=key)
                continue
            if data:
                resized = resize_image_long_edge(data, mime=None, max_long_edge=1536)
                if resized.resized:
                    log.info(
                        "feishu.image_resized",
                        message_id=msg_id,
                        image_key=key,
                        width=resized.width,
                        height=resized.height,
                    )
                mime = resized.mime or "image/png"
                image_path = self._save_inbound_image(
                    message_id=msg_id,
                    image_key=key,
                    data=resized.data,
                    mime=mime,
                )
                results.append(_InboundImage(
                    ImageContent(
                        mime=mime,
                        data=base64.b64encode(resized.data).decode("ascii"),
                    ),
                    image_path,
                ))
        return results

    @staticmethod
    def _build_image_buffer_hint(locked: list[str], labels: str) -> str:
        """通过 hooks 获取图片缓冲提示语，无匹配则返回默认提示。"""
        from whaleclaw.skills.hooks import get_skill_hooks
        from whaleclaw.skills.manager import SkillManager
        locked_lower = {s.lower() for s in locked}
        for skill in SkillManager().discover():
            if skill.id.lower() not in locked_lower:
                continue
            hooks = get_skill_hooks(skill)
            if hooks is not None and hooks.image_buffer_enabled:
                custom = hooks.image_buffer_hint(labels)
                if custom is not None:
                    return custom
        return f"已收到{labels}。继续上传图片，或发送提示词开始执行。"

    @staticmethod
    def _build_image_markdown(images: Sequence[_InboundImage]) -> str:
        """Render local image paths into markdown for WebChat/history replay."""
        if not images:
            return ""
        lines = ["(用户发送了图片)"]
        for idx, image in enumerate(images, start=1):
            lines.append(f"![飞书图片{idx}]({image.path})")
        return "\n".join(lines)

    def _save_inbound_image(
        self,
        *,
        message_id: str,
        image_key: str,
        data: bytes,
        mime: str,
    ) -> str:
        """Persist inbound Feishu image so WebChat/session history can reference it."""
        suffix = self._image_suffix_for_mime(mime)
        _FEISHU_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        dest = _FEISHU_MEDIA_DIR / f"{message_id[:8]}_{image_key}{suffix}"
        dest.write_bytes(data)
        return str(dest.resolve())

    @staticmethod
    def _image_suffix_for_mime(mime: str) -> str:
        """Map normalized image mime to a stable file suffix."""
        normalized = mime.strip().lower()
        if normalized == "image/png":
            return ".png"
        if normalized == "image/webp":
            return ".webp"
        if normalized == "image/gif":
            return ".gif"
        return ".jpg"

    async def _extract_file(self, message: dict[str, Any]) -> str | None:
        """Download incoming Feishu file message and return local absolute path."""
        msg_id = message.get("message_id", "")
        content = _parse_json_object(message.get("content", "{}"))
        if content is None:
            return None

        file_key = str(content.get("file_key", "")).strip()
        raw_name = str(content.get("file_name", "")).strip()
        if not file_key:
            return None

        filename = Path(raw_name).name if raw_name else f"{file_key}.bin"
        dest = _FEISHU_MEDIA_DIR / f"{msg_id[:8]}_{filename}"
        _FEISHU_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        try:
            data = await self._client.download_resource(msg_id, file_key, resource_type="file")
        except Exception:
            log.exception("feishu.file_download_failed", message_id=msg_id, file_key=file_key)
            return None
        if not data:
            return None
        try:
            dest.write_bytes(data)
        except OSError:
            log.exception("feishu.file_save_failed", path=str(dest))
            return None
        return str(dest.resolve())

    async def _send_pairing_prompt(
        self, open_id: str, msg_id: str
    ) -> None:
        import random
        import string

        code = "".join(random.choices(string.digits, k=6))  # noqa: S311
        self._pairing_codes[code] = open_id
        card = FeishuCard.text_card(
            f"请将此配对码发送给管理员进行验证:\n\n**{code}**",
            title="配对验证",
        )
        await self._client.reply_message(msg_id, "interactive", card)

    def approve_pairing(self, code: str) -> str | None:
        """Approve a pairing code and add the user to the allowlist."""
        open_id = self._pairing_codes.pop(code, None)
        if open_id:
            self._allowlist.add(open_id)
        return open_id
