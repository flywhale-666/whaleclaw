"""FastAPI application factory for the Gateway."""

# pyright: reportUnusedFunction=false

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from whaleclaw.agent.helpers.tool_execution import create_default_registry
from whaleclaw.config.paths import CONFIG_FILE, MEMORY_DIR, WHALECLAW_HOME, WORKSPACE_DIR
from whaleclaw.config.schema import WhaleclawConfig
from whaleclaw.cron.scheduler import CronAction, CronScheduler
from whaleclaw.cron.store import CronStore
from whaleclaw.gateway.middleware import AuthMiddleware, create_jwt
from whaleclaw.gateway.protocol import make_message
from whaleclaw.gateway.ws import push_to_session, websocket_handler
from whaleclaw.memory.manager import MemoryManager
from whaleclaw.memory.vector import SimpleMemoryStore
from whaleclaw.plugins.hooks import HookManager
from whaleclaw.plugins.loader import PluginLoader
from whaleclaw.plugins.registry import PluginRegistry
from whaleclaw.providers.router import ModelRouter
from whaleclaw.sessions.group_compressor import SessionGroupCompressor
from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.sessions.store import SessionStore
from whaleclaw.skills.clawhub import (
    ClawHubCliError,
    get_clawhub_auth_status,
    get_clawhub_cli_status,
    install_clawhub_cli,
    is_clawhub_cli_available,
    login_clawhub_cli,
    logout_clawhub_cli,
)
from whaleclaw.skills.clawhub import (
    install_skill as clawhub_install_skill,
)
from whaleclaw.skills.clawhub import (
    publish_installed_skill as clawhub_publish_installed_skill,
)
from whaleclaw.skills.clawhub import (
    search_skills as clawhub_search_skills,
)
from whaleclaw.tools.mcp_manage import (
    aggregate_mcp_servers,
    is_mcporter_available,
    remove_mcporter_server,
)
from whaleclaw.tools.registry import ToolRegistry
from whaleclaw.utils.log import get_logger
from whaleclaw.version import __version__

log = get_logger(__name__)

_UPLOAD_DIR = WHALECLAW_HOME / "uploads"
_CRON_DB_PATH = WHALECLAW_HOME / "cron.db"
_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
_MULTI_AGENT_MODES = {"parallel", "serial"}
_MULTI_AGENT_SCENARIOS = {
    "product_design",
    "content_creation",
    "software_development",
    "data_analysis_decision",
    "scientific_research",
    "intelligent_assistant",
    "workflow_automation",
}
_MULTI_AGENT_DEFAULT_SCENARIO = "software_development"


def _categorize_tool_name(name: str) -> str:
    """Map tool name to UI category."""
    explicit: dict[str, str] = {
        "bash": "system",
        "process": "system",
        "node_invoke": "system",
        "code_sandbox": "system",
        "file_read": "file",
        "file_write": "file",
        "file_edit": "file",
        "patch_apply": "file",
        "ppt_edit": "file",
        "docx_edit": "file",
        "xlsx_edit": "file",
        "web_fetch": "browser",
        "browser": "browser",
        "desktop_capture": "device",
        "canvas": "device",
        "sessions_list": "session",
        "sessions_history": "session",
        "sessions_send": "session",
        "cron": "automation",
        "cron_manage": "automation",
        "reminder": "automation",
        "skill": "skill",
        "memory_search": "memory",
        "memory_add": "memory",
        "memory_list": "memory",
        "mcp_manage": "integration",
    }
    if name in explicit:
        return explicit[name]
    if name.startswith("evomap_"):
        return "integration"
    if name.startswith("feishu_"):
        return "integration"
    if name.endswith("_edit"):
        return "file"
    if name.startswith("file_"):
        return "file"
    if name.startswith("memory_"):
        return "memory"
    if name.startswith("sessions_"):
        return "session"
    return "other"


def _default_multi_agent_roles() -> list[dict[str, object]]:
    return [
        {
            "id": "planner",
            "name": "规划师",
            "enabled": True,
            "model": "",
            "system_prompt": (
                "你是规划师。先澄清目标与约束，再拆解任务边界。"
                "输出：目标定义、范围/非范围、里程碑、优先级与验收标准。"
            ),
        },
        {
            "id": "architect",
            "name": "架构师",
            "enabled": True,
            "model": "",
            "system_prompt": (
                "你是架构师。基于规划给出技术方案与关键决策。"
                "输出：总体架构、模块边界、关键接口、技术选型、风险与降级方案。"
            ),
        },
        {
            "id": "implementer",
            "name": "执行者",
            "enabled": True,
            "model": "",
            "system_prompt": (
                "你是执行者。把方案落地为可执行步骤或代码改动。"
                "输出：分步实施计划、关键实现细节、命令/代码片段与完成定义。"
            ),
        },
        {
            "id": "reviewer",
            "name": "评审者",
            "enabled": True,
            "model": "",
            "system_prompt": (
                "你是评审者。独立审视可行性与质量风险。"
                "输出：问题清单、回归点、测试策略、监控指标与上线检查项。"
            ),
        },
    ]


def _default_multi_agent_config() -> dict[str, object]:
    return {
        "enabled": False,
        "scenario": _MULTI_AGENT_DEFAULT_SCENARIO,
        "custom_scenarios": [],
        "mode": "parallel",
        "max_rounds": 3,
        "roles": _default_multi_agent_roles(),
    }


def _normalize_multi_agent_config(raw: object) -> dict[str, Any]:
    defaults = _default_multi_agent_config()
    if not isinstance(raw, dict):
        return defaults

    raw_dict = cast(dict[object, object], raw)
    default_mode = str(defaults["mode"])
    default_scenario = str(defaults["scenario"])
    default_max_rounds = cast(int, defaults["max_rounds"])

    mode_value = raw_dict.get("mode", default_mode)
    mode_raw = str(mode_value).strip().lower()
    mode = mode_raw if mode_raw in _MULTI_AGENT_MODES else default_mode

    scenario_value = raw_dict.get("scenario", default_scenario)
    scenario_raw = str(scenario_value).strip()
    scenario = scenario_raw
    if scenario not in _MULTI_AGENT_SCENARIOS and not scenario.startswith("custom::"):
        scenario = default_scenario

    custom_scenarios_raw = raw_dict.get("custom_scenarios")
    custom_scenarios: list[str] = []
    if isinstance(custom_scenarios_raw, list):
        raw_custom_scenarios = cast(list[object], custom_scenarios_raw)
        for item in raw_custom_scenarios[:20]:
            cname = str(item).strip()
            if not cname or cname in custom_scenarios:
                continue
            custom_scenarios.append(cname[:40])

    max_rounds_raw = raw_dict.get("max_rounds", default_max_rounds)
    if isinstance(max_rounds_raw, bool):
        max_rounds = int(max_rounds_raw)
    elif isinstance(max_rounds_raw, int):
        max_rounds = max_rounds_raw
    elif isinstance(max_rounds_raw, (float, str)):
        try:
            max_rounds = int(max_rounds_raw)
        except Exception:
            max_rounds = default_max_rounds
    else:
        max_rounds = default_max_rounds
    max_rounds = max(1, min(max_rounds, 10))

    roles_raw = raw_dict.get("roles")
    roles: list[dict[str, Any]] = []
    if isinstance(roles_raw, list):
        raw_roles = cast(list[object], roles_raw)
        for idx, item in enumerate(raw_roles[:20], start=1):
            if not isinstance(item, dict):
                continue
            ritem = cast(dict[object, object], item)
            rid = str(ritem.get("id", f"role_{idx}")).strip().lower()
            rid = "".join(ch for ch in rid if ch.isalnum() or ch in {"_", "-"})
            if not rid:
                rid = f"role_{idx}"
            rname = str(ritem.get("name", f"角色{idx}")).strip() or f"角色{idx}"
            rmodel = str(ritem.get("model", "")).strip()
            prompt = str(ritem.get("system_prompt", "")).strip()
            roles.append(
                {
                    "id": rid[:64],
                    "name": rname[:50],
                    "enabled": bool(ritem.get("enabled", True)),
                    "model": rmodel[:100],
                    "system_prompt": prompt[:3000],
                }
            )
    if not roles:
        roles = _default_multi_agent_roles()

    return {
        "enabled": bool(raw_dict.get("enabled", defaults["enabled"])),
        "scenario": scenario,
        "custom_scenarios": custom_scenarios,
        "mode": mode,
        "max_rounds": max_rounds,
        "roles": roles,
    }


def _is_multi_agent_effective_for_metadata(
    config: WhaleclawConfig,
    metadata: Any,
) -> bool:
    global_enabled = False
    ma_raw = config.plugins.get("multi_agent", {})
    global_enabled = bool(ma_raw.get("enabled", False))
    if isinstance(metadata, dict):
        metadata_map = cast(dict[object, object], metadata)
        enabled = metadata_map.get("multi_agent_enabled")
        if isinstance(enabled, bool):
            return enabled
    return global_enabled


def _read_json_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw  # type: ignore[return-value]


def _write_json_config(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _as_str_object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    raw = cast(dict[object, object], value)
    return {str(key): item for key, item in raw.items() if isinstance(key, str)}


def create_app(config: WhaleclawConfig) -> FastAPI:
    """Create and configure the FastAPI application."""

    store = SessionStore()
    cron_store = CronStore(_CRON_DB_PATH)

    async def _feishu_send_rich_reply(
        channel: Any, peer_id: str, reply: str,
    ) -> None:
        """推送 agent 回复到飞书，正确处理图片和文件。"""
        bot = getattr(channel, "bot", None)
        client = getattr(channel, "client", None)
        if bot is None or client is None:
            return
        text_content, image_paths, file_paths = bot.prepare_reply_payload(reply)
        if text_content:
            await client.send_message(
                peer_id,
                "text",
                json.dumps({"text": text_content}, ensure_ascii=False),
            )
        for img_path in image_paths:
            await bot._send_image_to_peer(peer_id, img_path)  # noqa: SLF001
        for fp in file_paths:
            await bot._send_file_to_peer(peer_id, fp)  # noqa: SLF001

    _DEFERRED_TIME_PREFIX_RE = re.compile(
        r"^\s*"
        r"(?:"
        r"\d+\s*(?:分钟|小时|秒|min|hour|h)\s*(?:后|之后|以后)"
        r"|"
        r"(?:今天|明天|后天)?\s*(?:今晚|明早|早上|上午|中午|下午|晚上|凌晨|傍晚)?"
        r"\s*\d{1,2}\s*[点时:：]\s*(?:半|\d{0,2})"
        r")"
        r"\s*",
    )

    def _strip_deferred_time_prefix(text: str) -> str:
        """去掉 '10分钟后' / '晚上10点半' 等时间前缀，只保留要执行的任务内容。"""
        return _DEFERRED_TIME_PREFIX_RE.sub("", text).strip() or text

    async def _resolve_cron_session(
        target: str,
    ) -> tuple[str, Session | None, SessionManager | None]:
        """将 cron action.target 解析为 (session_id, session, manager)。

        当 target 不是有效 session（如 'user'）时，自动回退到最近活跃的会话。
        """
        mgr = state.get("manager")
        session_manager = mgr if isinstance(mgr, SessionManager) else None
        if session_manager is None:
            return target, None, None

        session = await session_manager.get(target)
        if session is not None:
            return target, session, session_manager

        # target 无效（如 'user'），回退到最近活跃的会话
        sessions = await session_manager.list_sessions()
        if sessions:
            latest = max(sessions, key=lambda s: s.updated_at)
            session = await session_manager.get(latest.id)
            if session is not None:
                return session.id, session, session_manager

        return target, None, session_manager

    async def _on_cron_fire(job_id: str, action: CronAction) -> None:
        from whaleclaw.gateway.ws import broadcast_all

        if action.type == "message":
            text = action.payload.get("text", "")
            if not text:
                return
            content = f"⏰ **提醒**: {text}"

            session_id, session, session_manager = await _resolve_cron_session(action.target)

            sent = await push_to_session(session_id, make_message(session_id, content))
            if not sent:
                await broadcast_all(make_message(session_id, content))

            # 飞书 fallback：优先用关联 session 的 peer_id，否则广播到飞书
            if not sent and feishu_channel is not None and feishu_channel.client:
                if session and session.channel == "feishu":
                    await feishu_channel.client.send_message(
                        session.peer_id,
                        "text",
                        json.dumps({"text": content}, ensure_ascii=False),
                    )
                else:
                    # 没有关联飞书会话时，尝试找任意飞书会话发送
                    if session_manager is not None:
                        all_sessions = await session_manager.list_sessions()
                        for s in sorted(all_sessions, key=lambda x: x.updated_at, reverse=True):
                            if s.channel == "feishu" and s.peer_id:
                                await feishu_channel.client.send_message(
                                    s.peer_id,
                                    "text",
                                    json.dumps({"text": content}, ensure_ascii=False),
                                )
                                break

            if session_manager is not None and session is not None:
                await session_manager.add_message(session, "assistant", content)

        elif action.type == "agent":
            text = str(action.payload.get("text", ""))
            if not text:
                return

            from whaleclaw.agent.single_agent import run_agent

            session_id, session, session_manager = await _resolve_cron_session(action.target)
            if session is None:
                log.warning("cron_fire_no_session", job_id=job_id, target=action.target)
                return

            notice = f"⏰ 定时任务触发：{text}"
            notice_pushed = await push_to_session(session_id, make_message(session_id, notice))
            if not notice_pushed:
                await broadcast_all(make_message(session_id, notice))
            if not notice_pushed and feishu_channel is not None:
                if session.channel == "feishu" and feishu_channel.client:
                    await feishu_channel.client.send_message(
                        session.peer_id,
                        "text",
                        json.dumps({"text": notice}, ensure_ascii=False),
                    )
            if session_manager is not None:
                await session_manager.add_message(session, "user", text)

            registry = state.get("registry")
            memory_manager = state.get("memory_manager")
            group_compressor = state.get("group_compressor")
            cron_router = ModelRouter(config.models)
            task_text = _strip_deferred_time_prefix(text)
            try:
                reply = await run_agent(
                    message=task_text,
                    session_id=session.id,
                    config=config,
                    session=session,
                    router=cron_router,
                    registry=registry,
                    session_manager=session_manager,
                    session_store=store,
                    memory_manager=memory_manager if isinstance(memory_manager, MemoryManager) else None,
                    group_compressor=group_compressor if isinstance(group_compressor, SessionGroupCompressor) else None,
                )
                if reply.strip() and session_manager is not None:
                    await session_manager.add_message(session, "assistant", reply)
                reply_pushed = await push_to_session(session_id, make_message(session_id, reply))
                await push_to_session(session_id, make_message(session_id, ""))
                if not reply_pushed:
                    await broadcast_all(make_message(session_id, reply))
                if not reply_pushed and feishu_channel is not None:
                    if session.channel == "feishu" and feishu_channel.client:
                        await _feishu_send_rich_reply(
                            feishu_channel, session.peer_id, reply,
                        )
            except Exception as exc:
                import structlog
                structlog.get_logger().error(
                    "cron_agent_fire_failed",
                    job_id=job_id,
                    session_id=session_id,
                    error=str(exc),
                )

    cron_scheduler = CronScheduler(on_fire=_on_cron_fire)
    plugin_registry = PluginRegistry()
    hook_manager = HookManager()

    feishu_channel: Any = None

    state: dict[str, Any] = {
        "manager": None,
        "registry": None,
        "memory_manager": None,
        "hook_manager": hook_manager,
        "group_compressor": None,
        "compression_ready": True,
        "compression_running": False,
        "compression_sessions_total": 0,
        "compression_sessions_done": 0,
        "compression_groups_total": 0,
        "compression_groups_done": 0,
        "compression_cache_hits": 0,
        "compression_generated": 0,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await store.open()
        await cron_store.open()

        cron_scheduler.set_persist(
            on_save=cron_store.save_job,
            on_delete=cron_store.delete_job,
        )
        persisted = await cron_store.load_jobs()
        for job in persisted:
            await cron_scheduler.add_job(job, persist=False)

        manager = SessionManager(store, config)
        state["manager"] = manager
        compression_router = ModelRouter(config.models)

        async def _on_message_persisted(session: Session, role: str, _content: str) -> None:
            if role != "assistant" or not isinstance(session, Session):
                return
            group_compressor = state["group_compressor"]
            if (
                not isinstance(group_compressor, SessionGroupCompressor)
                or not config.agent.summarizer.enabled
                or not session.messages
            ):
                return
            model_id = config.agent.summarizer.model.strip()
            if not model_id:
                return

            async def _run_followup() -> None:
                try:
                    await group_compressor.build_window_messages(
                        session_id=session.id,
                        messages=list(session.messages),
                        router=compression_router,
                        model_id=model_id,
                    )
                except Exception as exc:
                    log.debug(
                        "gateway.group_compress_followup_failed",
                        session_id=session.id,
                        error=str(exc),
                    )

            asyncio.create_task(
                _run_followup(),
                name=f"group-compress-followup-{session.id[:8]}",
            )

        manager.set_message_persist_hook(_on_message_persisted)
        memory_store = SimpleMemoryStore(persist_dir=MEMORY_DIR)
        memory_manager = MemoryManager(memory_store)
        state["memory_manager"] = memory_manager

        registry = create_default_registry(
            session_manager=manager,
            cron_scheduler=cron_scheduler,
            memory_manager=memory_manager,
            memory_store=memory_store,
        )

        await _load_plugins(config, registry, plugin_registry, hook_manager)
        state["registry"] = registry

        await plugin_registry.start_all()
        await cron_scheduler.start()

        summarizer_model = config.agent.summarizer.model.strip()
        prewarm_task: asyncio.Task[None] | None = None
        if config.agent.summarizer.enabled and summarizer_model:
            group_compressor = SessionGroupCompressor(store)
            state["group_compressor"] = group_compressor
            state["compression_ready"] = False
            state["compression_running"] = True
            state["compression_sessions_total"] = 0
            state["compression_sessions_done"] = 0
            state["compression_groups_total"] = 0
            state["compression_groups_done"] = 0
            state["compression_cache_hits"] = 0
            state["compression_generated"] = 0

            async def _prewarm_all_sessions() -> None:
                router = ModelRouter(config.models)
                try:
                    sessions = await manager.list_sessions()
                    state["compression_sessions_total"] = len(sessions)
                    log.info("compressor.prewarm_all_start", sessions_total=len(sessions))
                    for s in sessions:
                        loaded = await manager.get(s.id)
                        sessions_done = int(state["compression_sessions_done"])
                        if not loaded:
                            state["compression_sessions_done"] = sessions_done + 1
                            continue
                        if _is_multi_agent_effective_for_metadata(config, loaded.metadata):
                            await group_compressor.set_session_suspended(
                                session_id=loaded.id,
                                suspended=True,
                            )
                            state["compression_sessions_done"] = sessions_done + 1
                            continue
                        if not loaded.messages:
                            state["compression_sessions_done"] = sessions_done + 1
                            continue
                        stats = await group_compressor.prewarm_session(
                            session_id=loaded.id,
                            messages=loaded.messages,
                            router=router,
                            model_id=summarizer_model,
                        )
                        groups_total = int(state["compression_groups_total"])
                        groups_done = int(state["compression_groups_done"])
                        cache_hits = int(state["compression_cache_hits"])
                        generated = int(state["compression_generated"])

                        state["compression_groups_total"] = groups_total + int(
                            stats["total_groups"]
                        )
                        state["compression_groups_done"] = groups_done + int(
                            stats["processed_groups"]
                        )
                        state["compression_cache_hits"] = cache_hits + int(stats["cache_hits"])
                        state["compression_generated"] = generated + int(stats["generated"])
                        state["compression_sessions_done"] = sessions_done + 1
                        log.info(
                            "compressor.prewarm_session_done",
                            session_id=loaded.id,
                            sessions_done=state["compression_sessions_done"],
                            sessions_total=state["compression_sessions_total"],
                            groups_done=state["compression_groups_done"],
                            groups_total=state["compression_groups_total"],
                            cache_hits=state["compression_cache_hits"],
                            generated=state["compression_generated"],
                        )
                    log.info("compressor.prewarm_done", sessions=len(sessions))
                except Exception as exc:
                    log.warning("compressor.prewarm_failed", error=str(exc))
                finally:
                    state["compression_running"] = False
                    state["compression_ready"] = True

            prewarm_task = asyncio.create_task(
                _prewarm_all_sessions(),
                name="session-group-prewarm",
            )
        else:
            state["compression_ready"] = True
            state["compression_running"] = False

        nonlocal feishu_channel
        feishu_cfg = config.channels.feishu
        if feishu_cfg.app_id and feishu_cfg.app_secret:
            from whaleclaw.channels.feishu import FeishuChannel
            from whaleclaw.channels.feishu.config import FeishuConfig

            feishu_channel = FeishuChannel(FeishuConfig(**feishu_cfg.model_dump()))
            await feishu_channel.start()
            if feishu_channel.bot is not None:
                feishu_channel.bot.bind_agent(
                    config,
                    manager,
                    registry,
                    memory_manager=memory_manager,
                    hook_manager=hook_manager,
                    group_compressor=state["group_compressor"],
                    compression_ready_fn=lambda: bool(state["compression_ready"]),
                )

        _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        log.info(
            "gateway.started",
            tools=len(registry.list_tools()),
            plugins=len(plugin_registry.list_plugins()),
        )
        yield

        if feishu_channel is not None:
            await feishu_channel.stop()
        if prewarm_task is not None and not prewarm_task.done():
            prewarm_task.cancel()
        compressor = state["group_compressor"]
        if isinstance(compressor, SessionGroupCompressor):
            await compressor.shutdown()
        await cron_scheduler.stop()
        await plugin_registry.stop_all()
        await cron_store.close()
        await store.close()

    def _mgr() -> SessionManager:
        mgr = state["manager"]
        assert isinstance(mgr, SessionManager), "App not started"
        return mgr

    def _tool_registry() -> ToolRegistry:
        reg = state["registry"]
        assert isinstance(reg, ToolRegistry), "App not started"
        return reg

    def _memory_manager() -> MemoryManager:
        mgr = state["memory_manager"]
        assert isinstance(mgr, MemoryManager), "App not started"
        return mgr

    def _hook_manager() -> HookManager:
        hm = state["hook_manager"]
        assert isinstance(hm, HookManager), "App not started"
        return hm

    app = FastAPI(
        title="WhaleClaw Gateway",
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if config.gateway.auth.mode != "none":
        app.add_middleware(
            AuthMiddleware,
            auth_config=config.gateway.auth,
        )

    # ── Health ──────────────────────────────────────────────

    @app.get("/api/status")
    async def _api_status() -> dict[str, Any]:
        evomap_raw = config.plugins.get("evomap", {})
        evomap_enabled = bool(evomap_raw.get("enabled", False))
        return {
            "status": "ok",
            "version": __version__,
            "compression_ready": bool(state["compression_ready"]),
            "compression_running": bool(state["compression_running"]),
            "compression_progress": {
                "sessions_total": state["compression_sessions_total"],
                "sessions_done": state["compression_sessions_done"],
                "groups_total": state["compression_groups_total"],
                "groups_done": state["compression_groups_done"],
                "cache_hits": state["compression_cache_hits"],
                "generated": state["compression_generated"],
            },
            "gateway": {
                "port": config.gateway.port,
                "bind": config.gateway.bind,
            },
            "agent": {"model": config.agent.model},
            "auth_mode": config.gateway.auth.mode,
            "plugins": {
                "evomap": {"enabled": evomap_enabled},
            },
        }

    # ── Models ───────────────────────────────────────────────

    @app.get("/api/models")
    async def _api_list_models() -> dict[str, Any]:
        """Return all verified configured models grouped by provider.

        Each provider reads from ``configured_models`` (persisted via the
        config script).  Models include ``thinking`` level and ``tools``
        support flag so the frontend can display and auto-apply them.
        """
        from whaleclaw.providers.nvidia import NvidiaProvider

        providers_cfg = config.models
        available: list[dict[str, object]] = []

        for pname in providers_cfg.all_provider_names():
            pcfg = providers_cfg.get_provider(pname)
            has_auth = bool(pcfg.api_key) or (pcfg.auth_mode == "oauth" and pcfg.oauth_access)
            if not has_auth:
                continue

            if pcfg.configured_models:
                for cm in pcfg.configured_models:
                    if not cm.verified:
                        continue
                    if (
                        pname == "openai"
                        and pcfg.auth_mode == "oauth"
                        and not cm.id.lower().startswith("gpt-5")
                    ):
                        continue
                    tools: bool = True
                    if pname == "nvidia":
                        tools = NvidiaProvider.model_supports_tools(cm.id)
                    entry: dict[str, object] = {
                        "id": f"{pname}/{cm.id}",
                        "name": cm.name or cm.id,
                        "provider": pname,
                        "tools": tools,
                        "thinking": cm.thinking,
                    }
                    available.append(entry)

        return {
            "default": config.agent.model,
            "thinking_level": config.agent.thinking_level,
            "models": available,
        }

    # ── Auth ────────────────────────────────────────────────

    @app.post("/api/auth/login")
    async def _api_auth_login(body: dict[str, str]) -> JSONResponse:
        if config.gateway.auth.mode != "password":
            return JSONResponse(
                {"error": "当前认证模式不支持密码登录"},
                status_code=400,
            )
        if body.get("password") != config.gateway.auth.password:
            return JSONResponse({"error": "密码错误"}, status_code=401)
        token = create_jwt(config.gateway.auth)
        return JSONResponse({"token": token})

    @app.get("/api/auth/verify")
    async def _api_auth_verify() -> JSONResponse:
        return JSONResponse({"valid": True})

    # ── Sessions REST ───────────────────────────────────────

    @app.get("/api/sessions")
    async def _api_list_sessions() -> list[dict[str, Any]]:
        mgr = _mgr()
        sessions = await mgr.list_sessions()
        result: list[dict[str, Any]] = []
        for s in sessions:
            usage = await store.get_session_token_usage(s.id)
            result.append(
                {
                    "id": s.id,
                    "channel": s.channel,
                    "peer_id": s.peer_id,
                    "model": s.model,
                    "thinking_level": s.thinking_level,
                    "message_count": s.message_count or len(s.messages),
                    "tokens": usage["input_tokens"] + usage["output_tokens"],
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
            )
        return result

    @app.post("/api/sessions")
    async def _api_create_session() -> dict[str, Any]:
        mgr = _mgr()
        session = await mgr.create("webchat", "web-user")
        return {
            "id": session.id,
            "model": session.model,
            "created_at": session.created_at.isoformat(),
        }

    @app.get("/api/sessions/{session_id}")
    async def _api_get_session(session_id: str) -> JSONResponse:
        mgr = _mgr()
        session = await mgr.get(session_id)
        if not session:
            return JSONResponse({"error": "会话不存在"}, status_code=404)
        return JSONResponse(
            {
                "id": session.id,
                "channel": session.channel,
                "model": session.model,
                "thinking_level": session.thinking_level,
                "messages": [{"role": m.role, "content": m.content} for m in session.messages],
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            }
        )

    @app.delete("/api/sessions/{session_id}")
    async def _api_delete_session(session_id: str) -> JSONResponse:
        mgr = _mgr()
        await mgr.delete(session_id)
        return JSONResponse({"ok": True})

    @app.post("/api/sessions/{session_id}/compact")
    async def _api_compact_session(session_id: str) -> JSONResponse:
        return JSONResponse({"message": "上下文压缩功能将在后续版本实现"})

    @app.get("/api/token-usage")
    async def _api_get_total_token_usage() -> JSONResponse:
        total = await store.get_total_token_usage()
        by_model = await store.get_token_usage_by_model()
        return JSONResponse(
            {
                "total": total,
                "by_model": by_model,
            }
        )

    @app.get("/api/sessions/{session_id}/token-usage")
    async def _api_get_session_token_usage(session_id: str) -> JSONResponse:
        usage = await store.get_session_token_usage(session_id)
        return JSONResponse(usage)

    # ── Skills REST ────────────────────────────────────────

    _skill_manager: Any = None

    def _get_skill_manager() -> Any:
        nonlocal _skill_manager
        if _skill_manager is None:
            from whaleclaw.skills.manager import SkillManager

            _skill_manager = SkillManager()
        return _skill_manager

    @app.get("/api/skills")
    async def _api_list_skills() -> list[dict[str, Any]]:
        mgr = _get_skill_manager()
        bundled = mgr.discover()
        installed_ids = {s.id for s in mgr.list_installed()}
        result: list[dict[str, object]] = []
        for s in bundled:
            result.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "triggers": s.triggers,
                    "trigger_description": s.trigger_description,
                    "tools": s.tools,
                    "max_tokens": s.max_tokens,
                    "source": "user" if s.id in installed_ids else "bundled",
                }
            )
        return result

    @app.post("/api/skills/install")
    async def _api_install_skill(body: dict[str, str]) -> JSONResponse:
        source = body.get("source", "").strip()
        if not source:
            return JSONResponse({"error": "缺少 source 参数"}, status_code=400)
        mgr = _get_skill_manager()
        try:
            skill = mgr.install(source)
            return JSONResponse(
                {
                    "ok": True,
                    "skill": {"id": skill.id, "name": skill.name},
                }
            )
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/skills/{skill_id}", response_model=None)
    async def _api_get_skill_detail(skill_id: str):  # noqa: ANN201
        mgr = _get_skill_manager()
        all_skills = mgr.discover()
        skill = next((s for s in all_skills if s.id == skill_id), None)
        if not skill:
            return JSONResponse({"error": "技能不存在"}, status_code=404)
        installed_ids = {s.id for s in mgr.list_installed()}
        raw_content = skill.source_path.read_text(encoding="utf-8")
        return {
            "id": skill.id,
            "name": skill.name,
            "triggers": skill.triggers,
            "trigger_description": skill.trigger_description,
            "instructions": skill.instructions,
            "tools": skill.tools,
            "examples": skill.examples,
            "max_tokens": skill.max_tokens,
            "source": "user" if skill.id in installed_ids else "bundled",
            "raw_markdown": raw_content,
        }

    @app.get("/api/skills/{skill_id}/raw")
    async def _api_get_skill_detail_raw(skill_id: str) -> JSONResponse:
        mgr = _get_skill_manager()
        all_skills = mgr.discover()
        skill = next((s for s in all_skills if s.id == skill_id), None)
        if not skill:
            return JSONResponse({"error": "技能不存在"}, status_code=404)
        try:
            raw_content = skill.source_path.read_text(encoding="utf-8")
        except Exception as exc:
            return JSONResponse({"error": f"读取技能内容失败: {exc}"}, status_code=500)
        return JSONResponse({"id": skill.id, "name": skill.name, "raw_markdown": raw_content})

    @app.delete("/api/skills/{skill_id}")
    async def _api_uninstall_skill(skill_id: str) -> JSONResponse:
        mgr = _get_skill_manager()
        removed = mgr.uninstall(skill_id)
        if not removed:
            return JSONResponse({"error": "技能不存在或为内置技能"}, status_code=404)
        return JSONResponse({"ok": True})

    # ── ClawHub REST ───────────────────────────────────────

    def _read_clawhub_cfg() -> dict[str, Any]:
        clawhub_cfg = _as_str_object_dict(config.plugins.get("clawhub", {}))
        enabled = bool(clawhub_cfg.get("enabled", False))
        registry_url = str(clawhub_cfg.get("registry_url", "https://clawhub.ai")).strip()
        if not registry_url:
            registry_url = "https://clawhub.ai"
        api_token = str(clawhub_cfg.get("api_token", "")).strip()
        return {
            "enabled": enabled,
            "registry_url": registry_url,
            "api_token": api_token,
        }

    @app.get("/api/plugins/clawhub")
    async def _api_get_clawhub_config() -> JSONResponse:
        cfg = _read_clawhub_cfg()
        cli = get_clawhub_cli_status()
        return JSONResponse(
            {
                "enabled": cfg["enabled"],
                "registry_url": cfg["registry_url"],
                "has_token": bool(cfg["api_token"]),
                "cli_available": bool(cli["available"]),
                "cli_path": str(cli["path"]),
                "cli_version": str(cli["version"]),
            }
        )

    @app.post("/api/plugins/clawhub")
    async def _api_set_clawhub_config(body: dict[str, Any]) -> JSONResponse:
        enabled = bool(body.get("enabled", False))
        registry_url = str(body.get("registry_url", "https://clawhub.ai")).strip()
        if not registry_url:
            return JSONResponse({"error": "registry_url 不能为空"}, status_code=400)
        api_token = str(body.get("api_token", "")).strip()

        current_cfg = _as_str_object_dict(config.plugins.get("clawhub", {}))
        current_cfg["enabled"] = enabled
        current_cfg["registry_url"] = registry_url
        if api_token:
            current_cfg["api_token"] = api_token
        elif bool(body.get("clear_token", False)):
            current_cfg.pop("api_token", None)
        config.plugins["clawhub"] = current_cfg

        user_cfg = _read_json_config(CONFIG_FILE)
        plugins_cfg = _as_str_object_dict(user_cfg.get("plugins", {}))
        user_cfg["plugins"] = plugins_cfg
        user_clawhub = _as_str_object_dict(plugins_cfg.get("clawhub", {}))
        plugins_cfg["clawhub"] = user_clawhub
        user_clawhub["enabled"] = enabled
        user_clawhub["registry_url"] = registry_url
        if api_token:
            user_clawhub["api_token"] = api_token
        elif bool(body.get("clear_token", False)):
            user_clawhub.pop("api_token", None)

        try:
            _write_json_config(CONFIG_FILE, user_cfg)
        except Exception as exc:
            return JSONResponse(
                {"error": f"保存配置失败: {exc}"},
                status_code=500,
            )

        return JSONResponse(
            {
                "ok": True,
                "enabled": enabled,
                "registry_url": registry_url,
                "has_token": bool(api_token),
                "cli_available": is_clawhub_cli_available(),
                "persisted_to": str(CONFIG_FILE),
            }
        )

    @app.post("/api/clawhub/install-cli")
    async def _api_clawhub_install_cli() -> JSONResponse:
        try:
            info = install_clawhub_cli()
        except ClawHubCliError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"安装失败: {exc}"}, status_code=500)
        return JSONResponse(
            {
                "ok": True,
                "cli_available": True,
                "cli_path": info["path"],
                "cli_version": info["version"],
                "output": info["output"],
            }
        )

    @app.get("/api/clawhub/auth-status")
    async def _api_clawhub_auth_status() -> JSONResponse:
        cfg = _read_clawhub_cfg()
        status = get_clawhub_auth_status(
            registry_url=str(cfg["registry_url"]),
            workspace_dir=WORKSPACE_DIR,
            api_token=str(cfg["api_token"]) or None,
        )
        return JSONResponse(
            {
                "logged_in": bool(status["logged_in"]),
                "message": str(status["message"]),
            }
        )

    @app.post("/api/clawhub/login")
    async def _api_clawhub_login() -> JSONResponse:
        cfg = _read_clawhub_cfg()
        try:
            result = login_clawhub_cli(
                registry_url=str(cfg["registry_url"]),
                workspace_dir=WORKSPACE_DIR,
                api_token=str(cfg["api_token"]) or None,
            )
        except ClawHubCliError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"登录失败: {exc}"}, status_code=500)
        return JSONResponse(
            {
                "ok": bool(result["ok"]),
                "message": str(result["message"]),
                "output": str(result["output"]),
            }
        )

    @app.post("/api/clawhub/logout")
    async def _api_clawhub_logout() -> JSONResponse:
        cfg = _read_clawhub_cfg()
        try:
            result = logout_clawhub_cli(
                registry_url=str(cfg["registry_url"]),
                workspace_dir=WORKSPACE_DIR,
                api_token=str(cfg["api_token"]) or None,
            )
        except ClawHubCliError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"退出登录失败: {exc}"}, status_code=500)
        return JSONResponse(
            {
                "ok": bool(result["ok"]),
                "message": str(result["message"]),
                "output": str(result["output"]),
            }
        )

    @app.get("/api/clawhub/search")
    async def _api_clawhub_search(q: str = "", limit: int = 24) -> JSONResponse:
        query = q.strip()
        if not query:
            return JSONResponse({"error": "缺少查询参数 q"}, status_code=400)
        limit = max(1, min(limit, 200))
        cfg = _read_clawhub_cfg()
        if not bool(cfg["enabled"]):
            return JSONResponse({"error": "ClawHub 未启用，请先在技能页激活"}, status_code=400)
        try:
            items = clawhub_search_skills(
                query=query,
                registry_url=str(cfg["registry_url"]),
                workspace_dir=WORKSPACE_DIR,
                api_token=str(cfg["api_token"]) or None,
                limit=limit,
            )
        except ClawHubCliError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"ClawHub 搜索失败: {exc}"}, status_code=500)
        return JSONResponse({"items": items[:limit]})

    @app.post("/api/clawhub/install")
    async def _api_clawhub_install(body: dict[str, Any]) -> JSONResponse:
        slug = str(body.get("slug", "")).strip()
        if not slug:
            return JSONResponse({"error": "缺少 slug 参数"}, status_code=400)
        version_raw = str(body.get("version", "")).strip()
        version = version_raw or None
        repo_url = str(body.get("repo_url", "")).strip()

        cfg = _read_clawhub_cfg()
        if not bool(cfg["enabled"]):
            return JSONResponse({"error": "ClawHub 未启用，请先在技能页激活"}, status_code=400)

        mgr = _get_skill_manager()
        before = {s.id for s in mgr.list_installed()}

        try:
            output = clawhub_install_skill(
                slug=slug,
                version=version,
                registry_url=str(cfg["registry_url"]),
                workspace_dir=WORKSPACE_DIR,
                install_dir=WORKSPACE_DIR / "skills",
                api_token=str(cfg["api_token"]) or None,
            )
        except ClawHubCliError as exc:
            msg = str(exc)
            # Fallback path for registry throttling: install from upstream repo when available.
            if "HTTP 429" in msg and repo_url:
                try:
                    skill = mgr.install(repo_url)
                    output = f"ClawHub 限流，已回退到仓库安装: {repo_url}"
                    return JSONResponse(
                        {
                            "ok": True,
                            "slug": slug,
                            "version": version or "",
                            "installed_ids": [skill.id],
                            "output": output,
                        }
                    )
                except Exception:
                    pass
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"ClawHub 安装失败: {exc}"}, status_code=500)

        after = {s.id for s in mgr.list_installed()}
        added = sorted(after - before)
        return JSONResponse(
            {
                "ok": True,
                "slug": slug,
                "version": version or "",
                "installed_ids": added,
                "output": output,
            }
        )

    @app.post("/api/clawhub/publish-installed")
    async def _api_clawhub_publish_installed(body: dict[str, Any]) -> JSONResponse:
        skill_id = str(body.get("skill_id", "")).strip()
        if not skill_id:
            return JSONResponse({"error": "缺少 skill_id 参数"}, status_code=400)
        publish_slug_raw = body.get("publish_slug")
        publish_slug = str(publish_slug_raw).strip() if publish_slug_raw is not None else ""
        publish_version_raw = body.get("publish_version")
        publish_version = (
            str(publish_version_raw).strip() if publish_version_raw is not None else ""
        )

        cfg = _read_clawhub_cfg()
        if not bool(cfg["enabled"]):
            return JSONResponse({"error": "ClawHub 未启用，请先在技能页激活"}, status_code=400)
        if not is_clawhub_cli_available():
            return JSONResponse({"error": "CLI 未安装，请先安装 CLI"}, status_code=400)

        auth = get_clawhub_auth_status(
            registry_url=str(cfg["registry_url"]),
            workspace_dir=WORKSPACE_DIR,
            api_token=str(cfg["api_token"]) or None,
        )
        if not bool(auth["logged_in"]):
            return JSONResponse({"error": "未登录 ClawHub，请先登录"}, status_code=400)

        mgr = _get_skill_manager()
        installed = mgr.list_installed()
        target = next((s for s in installed if s.id == skill_id), None)
        if target is None:
            return JSONResponse({"error": "仅支持发布已安装(非内置)技能"}, status_code=404)

        try:
            output = clawhub_publish_installed_skill(
                skill_dir=target.source_path.parent,
                skill_slug=publish_slug or target.id,
                skill_version=publish_version or None,
                registry_url=str(cfg["registry_url"]),
                workspace_dir=WORKSPACE_DIR,
                api_token=str(cfg["api_token"]) or None,
            )
        except ClawHubCliError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        except Exception as exc:
            return JSONResponse({"error": f"发布失败: {exc}"}, status_code=500)

        return JSONResponse(
            {
                "ok": True,
                "skill_id": skill_id,
                "output": output,
            }
        )

    # ── Cron Jobs REST ─────────────────────────────────────

    def _job_to_dict(job: "CronJob") -> dict[str, Any]:
        from whaleclaw.cron.scheduler import CronJob as _CJ  # noqa: F811

        sched = job.schedule_obj
        result: dict[str, Any] = {
            "id": job.id,
            "name": job.name,
            "schedule_kind": sched.kind if sched else "cron",
            "enabled": job.enabled,
            "one_shot": job.one_shot,
            "created_at": job.created_at.isoformat(),
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "message": job.action.payload.get("text", ""),
            "action_type": job.action.type,
        }
        if sched:
            if sched.kind == "cron":
                result["cron_expr"] = sched.expr or job.schedule
            elif sched.kind == "every":
                result["every_minutes"] = sched.every_seconds // 60
            elif sched.kind == "at":
                result["at"] = sched.at.isoformat() if sched.at else None
        elif job.schedule:
            result["cron_expr"] = job.schedule
        return result

    @app.get("/api/cron/jobs")
    async def _api_list_cron_jobs() -> list[dict[str, Any]]:
        jobs = await cron_scheduler.list_jobs()
        return [_job_to_dict(j) for j in jobs]

    @app.post("/api/cron/jobs")
    async def _api_create_cron_job(body: dict[str, Any]) -> JSONResponse:
        from datetime import datetime as _dt
        from uuid import uuid4

        from whaleclaw.cron.scheduler import CronAction as _CA
        from whaleclaw.cron.scheduler import CronJob as _CJ
        from whaleclaw.cron.scheduler import Schedule as _Sched

        name = str(body.get("name", "")).strip()
        message = str(body.get("message", "")).strip()
        kind = str(body.get("schedule_kind", "cron")).strip()
        enabled = bool(body.get("enabled", True))
        one_shot = bool(body.get("one_shot", False))

        if not message:
            return JSONResponse({"error": "缺少 message"}, status_code=400)

        now = _dt.now()

        if kind == "cron":
            cron_expr = str(body.get("cron_expr", "")).strip()
            if not cron_expr or len(cron_expr.split()) != 5:
                return JSONResponse({"error": "cron 表达式需要 5 个字段"}, status_code=400)
            sched = _Sched(kind="cron", expr=cron_expr)
        elif kind == "every":
            try:
                minutes = int(body.get("minutes", 0))
            except (TypeError, ValueError):
                return JSONResponse({"error": "minutes 必须为整数"}, status_code=400)
            if minutes < 1:
                return JSONResponse({"error": "间隔必须大于 0"}, status_code=400)
            sched = _Sched(kind="every", every_seconds=minutes * 60)
        elif kind == "at":
            from datetime import datetime as _dt2

            at_str = str(body.get("at", "")).strip()
            if not at_str:
                return JSONResponse({"error": "at 类型需要 at 参数"}, status_code=400)
            try:
                at_time = _dt2.fromisoformat(at_str)
            except ValueError:
                return JSONResponse({"error": f"无法解析时间: {at_str}"}, status_code=400)
            sched = _Sched(kind="at", at=at_time)
            one_shot = True
        else:
            return JSONResponse({"error": f"未知调度类型: {kind}"}, status_code=400)

        action_type = str(body.get("action_type", "message")).strip()
        if action_type not in ("message", "agent"):
            action_type = "message"

        target_session = str(body.get("session_id", "")).strip() or "user"

        job = _CJ(
            id=f"cron-{uuid4().hex[:12]}",
            name=name or f"任务: {message[:20]}",
            schedule_obj=sched,
            action=_CA(
                type=action_type,  # pyright: ignore[reportArgumentType]
                target=target_session,
                payload={"text": message},
            ),
            enabled=enabled,
            created_at=now,
            one_shot=one_shot,
        )
        await cron_scheduler.add_job(job)
        return JSONResponse(_job_to_dict(job), status_code=201)

    @app.put("/api/cron/jobs/{job_id}")
    async def _api_update_cron_job(job_id: str, body: dict[str, Any]) -> JSONResponse:
        from datetime import datetime as _dt

        from whaleclaw.cron.scheduler import CronAction as _CA
        from whaleclaw.cron.scheduler import Schedule as _Sched

        existing = await cron_scheduler.get_job(job_id)
        if not existing:
            return JSONResponse({"error": "任务不存在"}, status_code=404)

        name = str(body.get("name", existing.name)).strip()
        message = str(body.get("message", existing.action.payload.get("text", ""))).strip()
        kind = str(body.get("schedule_kind", "")).strip()
        enabled = body.get("enabled", existing.enabled)
        if isinstance(enabled, bool):
            pass
        else:
            enabled = bool(enabled)
        one_shot = body.get("one_shot", existing.one_shot)

        sched = existing.schedule_obj
        if kind:
            if kind == "cron":
                cron_expr = str(body.get("cron_expr", "")).strip()
                if not cron_expr or len(cron_expr.split()) != 5:
                    return JSONResponse({"error": "cron 表达式需要 5 个字段"}, status_code=400)
                sched = _Sched(kind="cron", expr=cron_expr)
                one_shot = False
            elif kind == "every":
                try:
                    minutes = int(body.get("minutes", 0))
                except (TypeError, ValueError):
                    return JSONResponse({"error": "minutes 必须为整数"}, status_code=400)
                if minutes < 1:
                    return JSONResponse({"error": "间隔必须大于 0"}, status_code=400)
                sched = _Sched(kind="every", every_seconds=minutes * 60)
                one_shot = False
            elif kind == "at":
                at_str = str(body.get("at", "")).strip()
                if not at_str:
                    return JSONResponse({"error": "at 类型需要 at 参数"}, status_code=400)
                try:
                    at_time = _dt.fromisoformat(at_str)
                except ValueError:
                    return JSONResponse({"error": f"无法解析时间: {at_str}"}, status_code=400)
                sched = _Sched(kind="at", at=at_time)
                one_shot = True
            else:
                return JSONResponse({"error": f"未知调度类型: {kind}"}, status_code=400)

        action_type = str(body.get("action_type", existing.action.type)).strip()
        if action_type not in ("message", "agent"):
            action_type = existing.action.type

        target_session = str(body.get("session_id", "")).strip() or existing.action.target

        updated = existing.model_copy(update={
            "name": name,
            "schedule_obj": sched,
            "action": _CA(
                type=action_type,  # pyright: ignore[reportArgumentType]
                target=target_session,
                payload={"text": message},
            ),
            "enabled": enabled,
            "one_shot": one_shot,
        })
        await cron_scheduler.update_job(updated)
        return JSONResponse(_job_to_dict(updated))

    @app.delete("/api/cron/jobs/{job_id}")
    async def _api_delete_cron_job(job_id: str) -> JSONResponse:
        existing = await cron_scheduler.get_job(job_id)
        if not existing:
            return JSONResponse({"error": "任务不存在"}, status_code=404)
        await cron_scheduler.remove_job(job_id)
        return JSONResponse({"ok": True})

    # ── Tools REST ────────────────────────────────────────

    @app.get("/api/tools")
    async def _api_list_tools() -> list[dict[str, Any]]:
        reg = _tool_registry()
        tools = reg.list_tools()
        evomap_raw = config.plugins.get("evomap", {})
        evomap_enabled = bool(evomap_raw.get("enabled", False))
        if not evomap_enabled:
            tools = [t for t in tools if not t.name.startswith("evomap_")]
        result: list[dict[str, object]] = []
        for t in tools:
            result.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "category": _categorize_tool_name(t.name),
                    "parameters": [
                        {
                            "name": p.name,
                            "type": p.type,
                            "description": p.description,
                            "required": p.required,
                            **({"enum": p.enum} if p.enum else {}),
                        }
                        for p in t.parameters
                    ],
                }
            )
        return result

    # ── MCP REST ───────────────────────────────────────────

    @app.get("/api/mcp/servers")
    async def _api_list_mcp_servers() -> JSONResponse:
        """聚合内置 MCP + mcporter CLI 两个来源的服务列表。"""
        servers = aggregate_mcp_servers()
        return JSONResponse({
            "servers": servers,
            "mcporter_available": is_mcporter_available(),
        })

    @app.delete("/api/mcp/servers/{server_id}")
    async def _api_delete_mcp_server(
        server_id: str,
        source: str = "mcporter",
        config_path: str = "",
    ) -> JSONResponse:
        if source == "mcporter":
            ok = remove_mcporter_server(server_id, config_path=config_path or None)
            if ok:
                return JSONResponse({"ok": True})
            return JSONResponse({"error": "删除失败（mcporter 未安装或服务不存在）"}, status_code=400)
        return JSONResponse({"error": f"不支持删除来源为 {source} 的服务"}, status_code=400)

    # ── Plugins REST ──────────────────────────────────────

    @app.get("/api/plugins/multi-agent")
    async def _api_get_multi_agent_config() -> JSONResponse:
        raw = config.plugins.get("multi_agent", {})
        current = _normalize_multi_agent_config(raw)
        return JSONResponse(current)

    @app.post("/api/plugins/multi-agent")
    async def _api_set_multi_agent_config(body: dict[str, Any]) -> JSONResponse:
        current = _normalize_multi_agent_config(body)

        config.plugins["multi_agent"] = current

        user_cfg = _read_json_config(CONFIG_FILE)
        uc_plugins = _as_str_object_dict(user_cfg.get("plugins", {}))
        user_cfg["plugins"] = uc_plugins
        uc_plugins["multi_agent"] = current

        try:
            _write_json_config(CONFIG_FILE, user_cfg)
        except Exception as exc:
            return JSONResponse(
                {"error": f"保存配置失败: {exc}"},
                status_code=500,
            )

        return JSONResponse(
            {
                "ok": True,
                **current,
                "persisted_to": str(CONFIG_FILE),
            }
        )

    @app.post("/api/plugins/multi-agent/toggle")
    async def _api_toggle_multi_agent(body: dict[str, Any]) -> JSONResponse:
        """Toggle multi-agent enabled state independently of other config."""
        if "enabled" not in body:
            return JSONResponse({"error": "缺少 enabled 参数"}, status_code=400)
        enabled = bool(body["enabled"])

        ma_cfg = _as_str_object_dict(config.plugins.get("multi_agent", {}))
        if not ma_cfg:
            ma_cfg = _default_multi_agent_config()
        ma_cfg["enabled"] = enabled
        config.plugins["multi_agent"] = ma_cfg

        user_cfg = _read_json_config(CONFIG_FILE)
        uc_plugins = _as_str_object_dict(user_cfg.get("plugins", {}))
        user_cfg["plugins"] = uc_plugins
        uc_ma = _as_str_object_dict(uc_plugins.get("multi_agent", {}))
        uc_ma["enabled"] = enabled
        uc_plugins["multi_agent"] = uc_ma

        try:
            _write_json_config(CONFIG_FILE, user_cfg)
        except Exception as exc:
            return JSONResponse(
                {"error": f"切换多Agent模式失败: {exc}"},
                status_code=500,
            )

        return JSONResponse({"ok": True, "enabled": enabled})

    @app.get("/api/plugins/evomap")
    async def _api_get_evomap_config() -> JSONResponse:
        evomap_raw = config.plugins.get("evomap", {})
        enabled = bool(evomap_raw.get("enabled", False))
        return JSONResponse({"enabled": enabled})

    @app.post("/api/plugins/evomap")
    async def _api_set_evomap_config(body: dict[str, Any]) -> JSONResponse:
        if "enabled" not in body:
            return JSONResponse({"error": "缺少 enabled 参数"}, status_code=400)
        enabled = bool(body.get("enabled", False))

        current_cfg = _as_str_object_dict(config.plugins.get("evomap", {}))
        current_cfg["enabled"] = enabled
        config.plugins["evomap"] = current_cfg

        user_cfg = _read_json_config(CONFIG_FILE)
        uc_plugins_ev = _as_str_object_dict(user_cfg.get("plugins", {}))
        user_cfg["plugins"] = uc_plugins_ev
        uc_evomap = _as_str_object_dict(uc_plugins_ev.get("evomap", {}))
        uc_plugins_ev["evomap"] = uc_evomap
        uc_evomap["enabled"] = enabled
        try:
            _write_json_config(CONFIG_FILE, user_cfg)
        except Exception as exc:
            return JSONResponse(
                {"error": f"保存配置失败: {exc}"},
                status_code=500,
            )

        reg = _tool_registry()
        has_evomap_tools = any(t.name.startswith("evomap_") for t in reg.list_tools())

        if enabled and not has_evomap_tools:
            try:
                await _register_evomap_plugin_dynamically(
                    config, reg, plugin_registry, hook_manager,
                )
            except Exception as exc:
                log.warning("evomap.dynamic_register_failed", error=str(exc))

        if not enabled and has_evomap_tools:
            _unregister_evomap_tools(reg)
            existing = plugin_registry.get("evomap")
            if existing:
                await plugin_registry.unregister("evomap")

        return JSONResponse({"ok": True, "enabled": enabled, "persisted_to": str(CONFIG_FILE)})

    @app.get("/api/plugins/browser")
    async def _api_get_browser_config() -> JSONResponse:
        browser_raw = _as_str_object_dict(config.plugins.get("browser", {}))
        visible = True
        if "visible" in browser_raw:
            visible = bool(browser_raw.get("visible"))
        return JSONResponse({"visible": visible})

    @app.post("/api/plugins/browser")
    async def _api_set_browser_config(body: dict[str, Any]) -> JSONResponse:
        if "visible" not in body:
            return JSONResponse({"error": "缺少 visible 参数"}, status_code=400)
        visible = bool(body.get("visible", True))

        current_cfg = _as_str_object_dict(config.plugins.get("browser", {}))
        current_cfg["visible"] = visible
        config.plugins["browser"] = current_cfg

        user_cfg = _read_json_config(CONFIG_FILE)
        uc_plugins_browser = _as_str_object_dict(user_cfg.get("plugins", {}))
        user_cfg["plugins"] = uc_plugins_browser
        uc_browser = _as_str_object_dict(uc_plugins_browser.get("browser", {}))
        uc_plugins_browser["browser"] = uc_browser
        uc_browser["visible"] = visible

        try:
            _write_json_config(CONFIG_FILE, user_cfg)
        except Exception as exc:
            return JSONResponse(
                {"error": f"保存配置失败: {exc}"},
                status_code=500,
            )

        return JSONResponse({"ok": True, "visible": visible, "persisted_to": str(CONFIG_FILE)})

    # ── Memory Style REST ─────────────────────────────────

    @app.get("/api/memory/style")
    async def _api_get_memory_style() -> JSONResponse:
        mgr = _memory_manager()
        style = await mgr.get_global_style_directive()
        source = await mgr.get_global_style_source()
        return JSONResponse(
            {
                "enabled": config.agent.memory.global_style_enabled,
                "style_directive": style,
                "has_style": bool(style.strip()),
                "source": source,
            }
        )

    @app.post("/api/memory/style")
    async def _api_set_memory_style(body: dict[str, str]) -> JSONResponse:
        directive = str(body.get("style_directive", "")).strip()
        if not directive:
            return JSONResponse({"error": "style_directive 不能为空"}, status_code=400)
        if len(directive) > 300:
            return JSONResponse({"error": "style_directive 过长（最多 300 字）"}, status_code=400)
        mgr = _memory_manager()
        changed = await mgr.set_global_style_directive(
            directive,
            source="api:webchat",
        )
        return JSONResponse({"ok": True, "changed": changed})

    @app.delete("/api/memory/style")
    async def _api_clear_memory_style() -> JSONResponse:
        mgr = _memory_manager()
        removed = await mgr.clear_global_style_directive()
        return JSONResponse({"ok": True, "removed": removed})

    # ── File upload ─────────────────────────────────────────

    @app.post("/api/upload")
    async def _api_upload_file(file: UploadFile) -> JSONResponse:
        if not file.filename:
            return JSONResponse({"error": "文件名为空"}, status_code=400)
        dest = _UPLOAD_DIR / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return JSONResponse(
            {
                "url": f"/api/files/{file.filename}",
                "filename": file.filename,
                "size": dest.stat().st_size,
            }
        )

    @app.get("/api/files/{filename}", response_model=None)
    async def _api_get_file(filename: str) -> FileResponse | JSONResponse:
        path = _UPLOAD_DIR / filename
        if not path.is_file():
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        return FileResponse(path)

    def _resolve_local_path(path: str) -> Path | None:
        """Decode and resolve a local file path, with fuzzy fallback."""
        import re as _re

        decoded = unquote(unquote(path))
        fp = Path(decoded).resolve()
        if fp.is_file():
            return fp
        stem = fp.stem
        hash_match = _re.search(r"_([0-9a-f]{6,8})$", stem)
        if hash_match and fp.parent.is_dir():
            suffix_pattern = hash_match.group(0) + fp.suffix
            for candidate in fp.parent.iterdir():
                if candidate.name.endswith(suffix_pattern) and candidate.is_file():
                    return candidate
        return None

    @app.get("/api/local-file", response_model=None)
    async def _api_get_local_file(
        path: str,
        download: bool = False,
    ) -> FileResponse | JSONResponse:
        """Serve a local file generated by the Agent."""
        fp = _resolve_local_path(path)
        if not fp:
            log.warning("local-file.not_found", path=path)
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        if download:
            return FileResponse(
                fp,
                filename=fp.name,
                media_type="application/octet-stream",
            )
        return FileResponse(fp, filename=fp.name)

    @app.get("/api/file-info")
    async def _api_file_info(path: str) -> JSONResponse:
        """Return metadata about a local file (name, size)."""
        fp = _resolve_local_path(path)
        if not fp:
            return JSONResponse({"error": "文件不存在"}, status_code=404)
        size = fp.stat().st_size
        return JSONResponse(
            {
                "name": fp.name,
                "size": size,
                "size_human": (
                    f"{size / 1048576:.1f}MB"
                    if size >= 1048576
                    else f"{size / 1024:.0f}KB"
                    if size >= 1024
                    else f"{size}B"
                ),
                "ext": fp.suffix.lstrip(".").lower(),
            }
        )

    # ── WebSocket ───────────────────────────────────────────

    @app.websocket("/ws")
    async def _api_ws_endpoint(websocket: WebSocket) -> None:
        gc = state["group_compressor"]
        await websocket_handler(
            websocket,
            config,
            session_manager=_mgr(),
            registry=_tool_registry(),
            memory_manager=_memory_manager(),
            hook_manager=_hook_manager(),
            group_compressor=gc if isinstance(gc, SessionGroupCompressor) else None,
            compression_ready_fn=lambda: bool(state["compression_ready"]),
        )

    # ── Static files & SPA fallback ────────────────────────

    if _STATIC_DIR.is_dir():
        if (_STATIC_DIR / "assets").is_dir():
            app.mount(
                "/assets",
                StaticFiles(directory=str(_STATIC_DIR / "assets")),
                name="assets",
            )

        _boot_ts = str(int(time.time()))

        @app.get("/")
        async def _index() -> HTMLResponse:
            html = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")
            v = f"{__version__}.{_boot_ts}"
            html = html.replace("/assets/app.css", f"/assets/app.css?v={v}")
            html = html.replace("/assets/app.js", f"/assets/app.js?v={v}")
            return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    else:

        @app.get("/")
        async def _health() -> dict[str, str]:
            return {"status": "ok", "version": __version__}

    return app


async def _load_plugins(
    config: WhaleclawConfig,
    tool_registry: ToolRegistry,
    plugin_registry: PluginRegistry,
    hook_manager: HookManager,
) -> None:
    """Discover, load, and register all plugins."""
    from whaleclaw.plugins.sdk import WhaleclawPluginApi

    loader = PluginLoader()
    metas = loader.discover()
    if not metas:
        return

    def _plugin_cfg(pid: str) -> dict[str, Any]:
        val = config.plugins.get(pid, {})
        return _as_str_object_dict(val)

    for meta in metas:
        try:
            plugin = loader.load(meta.id)

            api = WhaleclawPluginApi(
                plugin_id=meta.id,
                get_config_fn=lambda pid, key, default: _plugin_cfg(pid).get(key, default),
                get_secret_fn=lambda pid, key: None,
                channel_register_fn=lambda ch: None,
                tool_register_fn=lambda t: tool_registry.register(t),
                hook_register_fn=lambda h, cb, p: hook_manager.register(h, cb, p),
                command_register_fn=lambda cmd, handler: None,
            )
            plugin.register(api)
            await plugin_registry.register(plugin, meta)

            log.info("plugin.loaded", plugin_id=meta.id, name=meta.name)
        except Exception as exc:
            log.warning(
                "plugin.load_failed",
                plugin_id=meta.id,
                error=str(exc),
            )


async def _register_evomap_plugin_dynamically(
    config: WhaleclawConfig,
    tool_registry: ToolRegistry,
    plugin_registry: PluginRegistry,
    hook_manager: HookManager,
) -> None:
    """Register EvoMap plugin tools at runtime when the toggle is switched on."""
    from whaleclaw.plugins.evomap.plugin import EvoMapPlugin
    from whaleclaw.plugins.loader import PluginMeta
    from whaleclaw.plugins.sdk import WhaleclawPluginApi

    existing = plugin_registry.get("evomap")
    if existing:
        await plugin_registry.unregister("evomap")

    plugin = EvoMapPlugin()

    def _evo_cfg(pid: str) -> dict[str, Any]:
        val = config.plugins.get(pid, {})
        return _as_str_object_dict(val)

    api = WhaleclawPluginApi(
        plugin_id="evomap",
        get_config_fn=lambda pid, key, default: _evo_cfg(pid).get(key, default),
        get_secret_fn=lambda pid, key: None,
        channel_register_fn=lambda ch: None,
        tool_register_fn=lambda t: tool_registry.register(t),
        hook_register_fn=lambda h, cb, p: hook_manager.register(h, cb, p),
        command_register_fn=lambda cmd, handler: None,
    )

    plugin.register(api)
    meta = PluginMeta(
        id="evomap",
        name="EvoMap",
        description="EvoMap collaborative evolution marketplace",
        version="1.0.0",
        author="WhaleClaw",
        entry="whaleclaw.plugins.evomap.plugin",
        path="",
    )
    await plugin_registry.register(plugin, meta)
    await plugin.on_start()
    log.info("evomap.dynamic_register_ok")


def _unregister_evomap_tools(tool_registry: ToolRegistry) -> None:
    """Remove all evomap_* tools from the registry."""
    evomap_names = [t.name for t in tool_registry.list_tools() if t.name.startswith("evomap_")]
    for name in evomap_names:
        tool_registry.unregister(name)
    if evomap_names:
        log.info("evomap.tools_unregistered", tools=evomap_names)
