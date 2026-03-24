"""WebSocket connection handler with session and tool support."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from whaleclaw.agent.commands import ChatCommand
from whaleclaw.agent.helpers.tool_execution import (
    is_transient_cli_usage_error as _is_transient_cli_usage_error,
)
from whaleclaw.agent.single_agent import (
    AgentDoneInfo,
    is_multi_agent_confirm,
    is_multi_agent_discuss_done,
    run_agent,
)
from whaleclaw.config.paths import WHALECLAW_HOME
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.gateway.middleware import verify_jwt
from whaleclaw.gateway.protocol import (
    MessageType,
    WSMessage,
    make_agent_done,
    make_error,
    make_message,
    make_pong,
    make_status,
    make_stream,
    make_tool_call,
    make_tool_result,
)
from whaleclaw.gateway.task_registry import task_registry
from whaleclaw.media.image_resize import resize_image_long_edge
from whaleclaw.plugins.evomap.bridge import build_memory_hint_from_hook_data
from whaleclaw.plugins.hooks import HookContext, HookManager, HookPoint
from whaleclaw.providers.base import ImageContent
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.tools.base import ToolResult
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.memory.manager import MemoryManager
    from whaleclaw.sessions.group_compressor import SessionGroupCompressor

log = get_logger(__name__)

_active_connections: dict[str, WebSocket] = {}
_SELF_HEAL_MARKER = "[system_self_heal]"
_SELF_HEAL_RETRY_KEY = "self_heal_retries"
_WS_UPLOAD_DIR = WHALECLAW_HOME / "uploads"


def _image_suffix_for_mime(mime: str) -> str:
    normalized = mime.strip().lower()
    if normalized == "image/png":
        return ".png"
    if normalized == "image/webp":
        return ".webp"
    if normalized == "image/gif":
        return ".gif"
    return ".jpg"


def _persist_ws_images(session_id: str, images: list[ImageContent]) -> tuple[list[str], str]:
    """Persist inbound WebChat images so downstream skills can reference local paths."""
    _WS_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    markdown_lines = ["(用户发送了图片)"]
    for index, image in enumerate(images, start=1):
        suffix = _image_suffix_for_mime(image.mime or "image/png")
        path = _WS_UPLOAD_DIR / f"ws_{session_id}_{uuid4().hex[:10]}{suffix}"
        path.write_bytes(base64.b64decode(image.data))
        normalized = str(path)
        saved_paths.append(normalized)
        markdown_lines.append(f"![WebChat图片{index}]({normalized})")
    return saved_paths, "\n".join(markdown_lines)


def _is_multi_agent_effective_for_session(
    config: WhaleclawConfig,
    session: Session,
) -> bool:
    global_enabled = False
    raw = config.plugins.get("multi_agent", {})
    global_enabled = bool(raw.get("enabled", False))
    metadata = session.metadata
    if isinstance(metadata.get("multi_agent_enabled"), bool):
        return bool(metadata["multi_agent_enabled"])
    return global_enabled


def _should_run_multi_agent_in_background(
    config: WhaleclawConfig,
    session: Session,
    user_message: str,
) -> bool:
    """True only when multi-Agent is enabled and message triggers execution."""
    if not _is_multi_agent_effective_for_session(config, session):
        return False
    metadata = session.metadata
    state = str(metadata.get("multi_agent_state", "")).strip().lower()
    waiting = state == "confirm" or (
        state == "" and bool(metadata.get("multi_agent_waiting_confirm", False))
    )
    msg = user_message.strip()
    if state == "discuss" and is_multi_agent_discuss_done(msg):
        return True
    return bool(waiting and is_multi_agent_confirm(msg))


async def push_to_session(session_id: str, msg: WSMessage) -> bool:
    """Push a message to a connected WebSocket by session ID.

    Returns True if sent successfully, False if session not connected.
    """
    ws = _active_connections.get(session_id)
    if ws is None:
        return False
    return await _safe_send(ws, msg)


async def broadcast_all(msg: WSMessage) -> int:
    """Broadcast a message to ALL active WebSocket connections (deduplicated)."""
    sent = 0
    seen: set[int] = set()
    for ws in list(_active_connections.values()):
        ws_id = id(ws)
        if ws_id in seen:
            continue
        seen.add(ws_id)
        if await _safe_send(ws, msg):
            sent += 1
    return sent


async def _safe_send(ws: WebSocket, msg: WSMessage) -> bool:
    """Send a WS message, returning False if connection is dead."""
    try:
        await ws.send_text(msg.model_dump_json())
        return True
    except Exception:
        return False


async def _resolve_session(
    session_id: str | None,
    current: Session | None,
    manager: SessionManager,
) -> Session:
    """Resolve session: reuse existing, load from DB, or create new."""
    target_id = session_id or (current.id if current else None)
    if target_id:
        if current and current.id == target_id:
            return current
        loaded = await manager.get(target_id)
        if loaded:
            return loaded
    return await manager.create("webchat", "ws-anonymous")


async def websocket_handler(
    websocket: WebSocket,
    config: WhaleclawConfig,
    session_manager: SessionManager,
    registry: ToolRegistry,
    memory_manager: MemoryManager | None = None,
    hook_manager: HookManager | None = None,
    group_compressor: SessionGroupCompressor | None = None,
    compression_ready_fn: Callable[[], bool] | None = None,
) -> None:
    """Handle a single WebSocket connection lifecycle."""
    auth_cfg = config.gateway.auth
    if auth_cfg.mode == "token":
        ws_token = websocket.query_params.get("token", "")
        if ws_token != auth_cfg.token:
            await websocket.close(code=4401, reason="认证失败")
            return
    elif auth_cfg.mode == "password":
        ws_token = websocket.query_params.get("token", "")
        if not verify_jwt(auth_cfg, ws_token):
            await websocket.close(code=4401, reason="认证失败")
            return

    await websocket.accept()

    conn_key = f"_conn:{uuid4().hex[:12]}"
    _active_connections[conn_key] = websocket

    session: Session | None = None
    router = ModelRouter(config.models)
    from whaleclaw.sessions.compressor import ContextCompressor

    session_store = session_manager.store
    _compressor = ContextCompressor()
    chat_cmd = ChatCommand(
        session_manager,
        config=config,
        router=router,
        compressor=_compressor,
        session_store=session_store,
    )
    ws_alive = True
    inbox: asyncio.Queue[WSMessage] = asyncio.Queue()

    log.debug("ws.connected", conn_key=conn_key)

    # ---- helpers to build a WS sink from the current connection ----

    def _make_sink(ws: WebSocket) -> Callable[[WSMessage], Awaitable[bool]]:
        async def _sink(msg: WSMessage) -> bool:
            return await _safe_send(ws, msg)
        return _sink

    # ---- reconnection: attach to a running background task ----

    async def _try_attach_running_task(sid: str) -> bool:
        """If a background task is running for *sid*, attach this WS and return True."""
        if not task_registry.has_running(sid):
            entry = task_registry.get(sid)
            if entry is not None and not entry.is_alive and entry.final_result:
                await _safe_send(websocket, make_message(sid, entry.final_result))
                return True
            return False
        attached = await task_registry.attach(sid, _make_sink(websocket))
        if attached:
            log.info("ws.reattached_running_task", session_id=sid)
            await _safe_send(
                websocket,
                make_status(sid, "已重新连接到正在执行的多Agent任务，继续接收结果…"),
            )
        return attached

    async def _reader() -> None:
        """Read WS messages and dispatch: ping handled inline, others queued."""
        nonlocal ws_alive
        try:
            while ws_alive:
                raw = await websocket.receive_text()
                log.debug("ws.recv_raw", size=len(raw), preview=raw[:120])
                try:
                    incoming = WSMessage.model_validate_json(raw)
                except Exception:
                    sid = session.id if session else ""
                    log.warning("ws.invalid_message_json", preview=raw[:200], session_id=sid)
                    await _safe_send(websocket, make_error(sid, "无效的消息格式"))
                    continue
                if incoming.type == MessageType.PING:
                    log.debug("ws.ping", session_id=incoming.session_id)
                    await _safe_send(websocket, make_pong())
                    if incoming.session_id and session is None:
                        loaded = await session_manager.get(incoming.session_id)
                        if loaded:
                            await _try_attach_running_task(loaded.id)
                    continue

                log.debug(
                    "ws.incoming",
                    type=incoming.type,
                    session_id=incoming.session_id,
                    payload_keys=list(incoming.payload.keys()),
                )

                if incoming.type == MessageType.MESSAGE:
                    content = str(incoming.payload.get("content", ""))
                    log.info(
                        "ws.message_enqueued",
                        session_id=incoming.session_id,
                        content_len=len(content),
                        content_preview=(" ".join(content.split())[:80]),
                    )
                    await inbox.put(incoming)
        except WebSocketDisconnect:
            ws_alive = False
            sid = session.id if session else "(none)"
            log.debug("ws.disconnected", session_id=sid)
        except Exception:
            ws_alive = False
            sid = session.id if session else "(none)"
            log.debug("ws.closed_unexpectedly", session_id=sid)

    async def _processor() -> None:
        """Process queued user messages (run agent, send replies)."""
        nonlocal session
        while True:
            try:
                incoming = await asyncio.wait_for(inbox.get(), timeout=1.0)
            except TimeoutError:
                if not ws_alive and inbox.empty():
                    break
                continue

            session = await _resolve_session(
                incoming.session_id, session, session_manager,
            )
            _active_connections[session.id] = websocket

            # If a background task is already running for this session, skip.
            if task_registry.has_running(session.id):
                await _safe_send(
                    websocket,
                    make_status(session.id, "多Agent任务正在执行中，请等待完成…"),
                )
                continue

            if (
                compression_ready_fn is not None
                and not compression_ready_fn()
                and not _is_multi_agent_effective_for_session(config, session)
            ):
                await _safe_send(
                    websocket,
                    make_status(session.id, "会话压缩中，请稍后再试…"),
                )
                continue

            content = incoming.payload.get("content", "")
            raw_images_obj = incoming.payload.get("images", [])
            raw_images = (
                cast(list[object], raw_images_obj)
                if isinstance(raw_images_obj, list)
                else []
            )
            log.debug(
                "ws.message_processing",
                session_id=session.id,
                incoming_session_id=incoming.session_id,
                content_len=len(content) if isinstance(content, str) else 0,
                images_count=len(raw_images),
            )

            images: list[ImageContent] = []
            for img_data in raw_images:
                if not isinstance(img_data, dict):
                    continue
                image_item = cast(dict[object, object], img_data)
                image_data = image_item.get("data")
                if not image_data:
                    continue
                mime = str(image_item.get("mime", "image/png"))
                raw_b64 = str(image_data)
                try:
                    decoded = base64.b64decode(raw_b64, validate=True)
                except Exception:
                    log.warning("ws.image_invalid_base64", session_id=session.id)
                    continue
                resized = resize_image_long_edge(decoded, mime=mime, max_long_edge=1536)
                if resized.resized:
                    log.info(
                        "ws.image_resized",
                        session_id=session.id,
                        width=resized.width,
                        height=resized.height,
                    )
                images.append(ImageContent(
                    mime=resized.mime or mime,
                    data=base64.b64encode(resized.data).decode("ascii"),
                ))

            raw_attachments_obj = incoming.payload.get("attachments", [])
            raw_attachments = (
                cast(list[object], raw_attachments_obj)
                if isinstance(raw_attachments_obj, list)
                else []
            )
            if raw_attachments:
                att_lines: list[str] = []
                for att in raw_attachments:
                    if not isinstance(att, dict):
                        continue
                    att_item = cast(dict[object, object], att)
                    att_name = str(att_item.get("name", "")).strip()
                    att_filename = str(att_item.get("filename", "")).strip()
                    if att_filename:
                        att_path = str(Path.home() / ".whaleclaw" / "uploads" / att_filename)
                        att_lines.append(f"[附件: {att_name}]({att_path})")
                if att_lines:
                    content = (content or "").rstrip()
                    content = f"{content}\n\n用户上传了以下文件：\n" + "\n".join(att_lines)

            if not content and not images:
                await _safe_send(websocket, make_error(session.id, "消息内容为空"))
                continue

            if images:
                log.debug("ws.images", count=len(images), session_id=session.id)
                _saved_paths, image_markdown = _persist_ws_images(session.id, images)
                content = (content or "").strip()
                content = f"{content}\n\n{image_markdown}" if content else image_markdown

            cmd_result = await chat_cmd.handle(content, session) if content else None
            if cmd_result is not None:
                await _safe_send(websocket, make_message(session.id, cmd_result))
                if content.strip().lower() in ("/new", "/reset"):
                    session = await session_manager.get(session.id) or session
                continue

            try:
                await session_manager.add_message(session, "user", content or "(图片)")
            except Exception as exc:
                log.warning("ws.add_message_failed", error=str(exc))
                session = await session_manager.create("webchat", "ws-anonymous")
                await session_manager.add_message(session, "user", content or "(图片)")

            sid = session.id
            current_session = session
            run_in_background = _should_run_multi_agent_in_background(
                config, session, content or "",
            )

            if run_in_background:
                await _run_multi_agent_background(
                    session=current_session,
                    sid=sid,
                    content=content,
                    images=images,
                    incoming=incoming,
                    config=config,
                    router=router,
                    registry=registry,
                    session_manager=session_manager,
                    session_store=session_store,
                    memory_manager=memory_manager,
                    hook_manager=hook_manager,
                    group_compressor=group_compressor,
                    websocket=websocket,
                )
            else:
                await _run_single_agent_inline(
                    session=current_session,
                    sid=sid,
                    content=content,
                    images=images,
                    incoming=incoming,
                    config=config,
                    router=router,
                    registry=registry,
                    session_manager=session_manager,
                    session_store=session_store,
                    memory_manager=memory_manager,
                    hook_manager=hook_manager,
                    group_compressor=group_compressor,
                    websocket=websocket,
                    ws_alive_fn=lambda: ws_alive,
                )

    # ---- inline single-agent execution (original behaviour) ----

    async def _run_single_agent_inline(
        *,
        session: Session,
        sid: str,
        content: str,
        images: list[ImageContent],
        incoming: WSMessage,
        config: WhaleclawConfig,
        router: ModelRouter,
        registry: ToolRegistry,
        session_manager: SessionManager,
        session_store: SessionStore,
        memory_manager: MemoryManager | None,
        hook_manager: HookManager | None,
        group_compressor: SessionGroupCompressor | None,
        websocket: WebSocket,
        ws_alive_fn: Callable[[], bool],
    ) -> None:
        async def on_stream(chunk: str, _sid: str = sid) -> None:
            if ws_alive_fn():
                await _safe_send(websocket, make_stream(_sid, chunk))

        async def on_tool_call_cb(
            name: str,
            arguments: dict[str, object],
            _sid: str = sid,
        ) -> None:
            if ws_alive_fn():
                await _safe_send(websocket, make_tool_call(_sid, name, arguments))

        async def on_tool_result_cb(
            name: str,
            result: ToolResult,
            _sid: str = sid,
        ) -> None:
            if _is_transient_cli_usage_error(result):
                return
            if ws_alive_fn():
                await _safe_send(
                    websocket,
                    make_tool_result(_sid, name, result.output, result.success, result.error),
                )

        async def on_round_result_cb(
            round_no: int,
            content_text: str,
            _sid: str = sid,
            _session: Session = session,
        ) -> None:
            if not ws_alive_fn():
                return
            round_msg = f"第 {round_no} 轮交付\n\n{content_text}".strip()
            await session_manager.add_message(_session, "assistant", round_msg)
            await _safe_send(websocket, make_message(_sid, round_msg))

        async def on_done_cb(
            info: AgentDoneInfo,
            _sid: str = sid,
        ) -> None:
            if ws_alive_fn():
                await _safe_send(
                    websocket,
                    make_agent_done(
                        _sid,
                        model=info.model,
                        input_tokens=info.input_tokens,
                        output_tokens=info.output_tokens,
                        llm_rounds=info.llm_rounds,
                    ),
                )

        async def _clear_self_heal_retries() -> None:
            if _SELF_HEAL_RETRY_KEY not in session.metadata:
                return
            session.metadata.pop(_SELF_HEAL_RETRY_KEY, None)
            await session_manager.update_metadata(session, session.metadata)

        async def _attempt_self_heal_inline(root_error: str, extra_memory: str) -> bool:
            if not config.agent.auto_self_heal_on_error:
                return False
            if content.strip().startswith(_SELF_HEAL_MARKER):
                return False
            retry_used_raw = session.metadata.get(_SELF_HEAL_RETRY_KEY, 0)
            try:
                retry_used = int(retry_used_raw)
            except (TypeError, ValueError):
                retry_used = 0
            if retry_used >= config.agent.self_heal_max_retries:
                return False
            session.metadata[_SELF_HEAL_RETRY_KEY] = retry_used + 1
            await session_manager.update_metadata(session, session.metadata)

            await _safe_send(
                websocket,
                make_status(session.id, "执行中断，正在自动恢复并继续完成任务…"),
            )

            heal_message = (
                f"{_SELF_HEAL_MARKER}\n"
                "上一轮执行出现异常，请立即执行自愈流程：\n"
                "1) 读取 ~/.whaleclaw/logs/gateway.err "
                "与 ~/.whaleclaw/logs/gateway.log 最近 200 行\n"
                "2) 定位错误根因，必要时在当前项目中修复代码（优先用 patch_apply）\n"
                "3) 执行最小回归验证，确认错误消失\n"
                "4) 修复成功后，继续完成用户原始任务\n"
                "5) 对用户的最终回复只输出任务结果，不要暴露内部报错、日志或堆栈\n"
                "6) 仅当你确认仍无法完成任务时，才简要说明错误与建议\n\n"
                f"原始用户请求: {content or '(用户发送了图片)'}\n"
                f"内部参考错误(勿直接输出给用户): {root_error}"
            )
            try:
                heal_reply = await run_agent(
                    message=heal_message,
                    session_id=session.id,
                    config=config,
                    on_stream=on_stream,
                    session=session,
                    router=router,
                    registry=registry,
                    on_tool_call=on_tool_call_cb,
                    on_tool_result=on_tool_result_cb,
                    on_round_result=on_round_result_cb,
                    images=images or None,
                    session_manager=session_manager,
                    session_store=session_store,
                    memory_manager=memory_manager,
                    extra_memory=extra_memory,
                    trigger_event_id=incoming.id,
                    trigger_text_preview="[self-heal]",
                    group_compressor=group_compressor,
                )
                if heal_reply.strip():
                    await session_manager.add_message(session, "assistant", heal_reply)
                    await _safe_send(websocket, make_message(session.id, heal_reply))
                await _clear_self_heal_retries()
                return True
            except Exception as heal_exc:
                log.error(
                    "agent.self_heal_failed",
                    error=str(heal_exc),
                    session_id=session.id,
                )
                return False

        extra_memory = ""
        try:
            if hook_manager is not None:
                hook_out = await hook_manager.run(
                    HookPoint.BEFORE_MESSAGE,
                    HookContext(
                        hook=HookPoint.BEFORE_MESSAGE,
                        session_id=session.id,
                        data={
                            "message": content,
                            "channel": "webchat",
                        },
                    ),
                )
                if not hook_out.proceed:
                    await _safe_send(websocket, make_message(session.id, "消息被策略阻止。"))
                    return
                extra_memory = build_memory_hint_from_hook_data(hook_out.data)
            reply = await run_agent(
                message=content or "(用户发送了图片)",
                session_id=session.id,
                config=config,
                on_stream=on_stream,
                session=session,
                router=router,
                registry=registry,
                on_tool_call=on_tool_call_cb,
                on_tool_result=on_tool_result_cb,
                on_round_result=on_round_result_cb,
                on_done=on_done_cb,
                images=images or None,
                session_manager=session_manager,
                session_store=session_store,
                memory_manager=memory_manager,
                extra_memory=extra_memory,
                trigger_event_id=incoming.id,
                trigger_text_preview=content or "(用户发送了图片)",
                group_compressor=group_compressor,
            )
            if reply.strip():
                await session_manager.add_message(session, "assistant", reply)
                await _safe_send(websocket, make_message(session.id, reply))
            await _clear_self_heal_retries()
        except Exception as exc:
            if hook_manager is not None:
                with suppress(Exception):
                    await hook_manager.run(
                        HookPoint.ON_ERROR,
                        HookContext(
                            hook=HookPoint.ON_ERROR,
                            session_id=session.id,
                            data={
                                "error": str(exc),
                                "message": content,
                                "channel": "webchat",
                            },
                        ),
                    )
            log.error("agent.error", error=str(exc), session_id=session.id)
            healed = await _attempt_self_heal_inline(str(exc), extra_memory=extra_memory)
            if healed:
                return
            await _safe_send(websocket, make_error(session.id, f"Agent 处理失败: {exc}"))

    # ---- background multi-Agent execution ----

    async def _run_multi_agent_background(
        *,
        session: Session,
        sid: str,
        content: str,
        images: list[ImageContent],
        incoming: WSMessage,
        config: WhaleclawConfig,
        router: ModelRouter,
        registry: ToolRegistry,
        session_manager: SessionManager,
        session_store: SessionStore,
        memory_manager: MemoryManager | None,
        hook_manager: HookManager | None,
        group_compressor: SessionGroupCompressor | None,
        websocket: WebSocket,
    ) -> None:
        """Launch the multi-Agent task in the background via TaskRegistry."""

        extra_memory = ""
        if hook_manager is not None:
            try:
                hook_out = await hook_manager.run(
                    HookPoint.BEFORE_MESSAGE,
                    HookContext(
                        hook=HookPoint.BEFORE_MESSAGE,
                        session_id=session.id,
                        data={"message": content, "channel": "webchat"},
                    ),
                )
                if not hook_out.proceed:
                    await _safe_send(websocket, make_message(session.id, "消息被策略阻止。"))
                    return
                extra_memory = build_memory_hint_from_hook_data(hook_out.data)
            except Exception as exc:
                log.warning("ws.hook_before_message_failed", error=str(exc))

        async def _bg_run_agent() -> str:
            """The actual coroutine that runs in the background task."""
            entry = task_registry.get(sid)

            async def on_stream(chunk: str) -> None:
                if entry:
                    await entry.emit(make_stream(sid, chunk))

            async def on_tool_call_bg(name: str, arguments: dict[str, object]) -> None:
                if entry:
                    await entry.emit(make_tool_call(sid, name, arguments))

            async def on_tool_result_bg(name: str, result: ToolResult) -> None:
                if entry:
                    await entry.emit(
                        make_tool_result(sid, name, result.output, result.success, result.error)
                    )

            async def on_round_result_bg(round_no: int, content_text: str) -> None:
                round_msg = f"第 {round_no} 轮交付\n\n{content_text}".strip()
                await session_manager.add_message(session, "assistant", round_msg)
                if entry:
                    await entry.emit(make_message(sid, round_msg))

            try:
                reply = await run_agent(
                    message=content or "(用户发送了图片)",
                    session_id=session.id,
                    config=config,
                    on_stream=on_stream,
                    session=session,
                    router=router,
                    registry=registry,
                    on_tool_call=on_tool_call_bg,
                    on_tool_result=on_tool_result_bg,
                    on_round_result=on_round_result_bg,
                    images=images or None,
                    session_manager=session_manager,
                    session_store=session_store,
                    memory_manager=memory_manager,
                    extra_memory=extra_memory,
                    trigger_event_id=incoming.id,
                    trigger_text_preview=content or "(用户发送了图片)",
                    group_compressor=group_compressor,
                )
                if reply.strip():
                    await session_manager.add_message(session, "assistant", reply)
                    if entry:
                        await entry.emit(make_message(sid, reply))
                return reply
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("agent.bg_error", error=str(exc), session_id=session.id)
                if entry:
                    await entry.emit(
                        make_error(session.id, f"多Agent任务执行失败: {exc}")
                    )
                if hook_manager is not None:
                    with suppress(Exception):
                        await hook_manager.run(
                            HookPoint.ON_ERROR,
                            HookContext(
                                hook=HookPoint.ON_ERROR,
                                session_id=session.id,
                                data={
                                    "error": str(exc),
                                    "message": content,
                                    "channel": "webchat",
                                },
                            ),
                        )
                return f"多Agent任务执行失败: {exc}"

        entry = task_registry.launch(sid, _bg_run_agent())
        await entry.attach(_make_sink(websocket))
        await _safe_send(
            websocket,
            make_status(sid, "多Agent任务执行中…"),
        )

    # ---- lifecycle ----

    reader_task = asyncio.create_task(_reader())
    processor_task = asyncio.create_task(_processor())

    try:
        await reader_task
    except Exception as exc:
        if not isinstance(exc, asyncio.CancelledError):
            log.error("ws.reader_error", error=str(exc))
    finally:
        ws_alive = False

    # Let _processor drain any remaining inbox messages before shutting down.
    try:
        await asyncio.wait_for(processor_task, timeout=300)
    except TimeoutError:
        log.warning("ws.processor_drain_timeout", session_id=session.id if session else "(none)")
        processor_task.cancel()
    except Exception as exc:
        if not isinstance(exc, asyncio.CancelledError):
            log.error("ws.processor_error", error=str(exc))

    # Detach from any running background task (don't cancel it).
    if session is not None:
        await task_registry.detach(session.id)
    _active_connections.pop(conn_key, None)
    if session:
        _active_connections.pop(session.id, None)
