"""Chat commands — slash-commands handled before reaching the LLM."""

from __future__ import annotations

from typing import TYPE_CHECKING

from whaleclaw.sessions.manager import Session, SessionManager
from whaleclaw.utils.log import get_logger

if TYPE_CHECKING:
    from whaleclaw.config.schema import WhaleclawConfig
    from whaleclaw.providers.router import ModelRouter
    from whaleclaw.sessions.compressor import ContextCompressor
    from whaleclaw.sessions.store import SessionStore

log = get_logger(__name__)

_HELP_TEXT = """\
可用命令:
  /new, /reset   — 重置当前会话
  /status        — 显示会话状态
  /model <id>    — 切换模型 (如 /model openai/gpt-5.2)
  /think <level> — 设置思考深度 (off/low/medium/high)
  /compact       — 压缩会话上下文 (生成 L0/L1 摘要)
  /help          — 显示此帮助"""

_VALID_THINKING = {"off", "low", "medium", "high", "xhigh"}


class ChatCommand:
    """Parse and execute slash-commands."""

    def __init__(
        self,
        session_manager: SessionManager,
        *,
        config: WhaleclawConfig | None = None,
        router: ModelRouter | None = None,
        compressor: ContextCompressor | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._sm = session_manager
        self._config = config
        self._router = router
        self._compressor = compressor
        self._store = session_store

    async def handle(self, text: str, session: Session) -> str | None:
        """If *text* is a command, execute it and return a response string.

        Returns ``None`` if *text* is not a command.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None

        parts = stripped.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/new", "/reset"):
            await self._sm.reset(session.id)
            return "会话已重置。"

        if cmd == "/status":
            return (
                f"会话 ID: {session.id}\n"
                f"模型: {session.model}\n"
                f"思考深度: {session.thinking_level}\n"
                f"消息数: {len(session.messages)}\n"
                f"渠道: {session.channel}"
            )

        if cmd == "/model":
            if not arg:
                return f"当前模型: {session.model}\n用法: /model <provider/model>"
            await self._sm.update_model(session, arg)
            return f"已切换到 {arg}"

        if cmd == "/think":
            if not arg or arg not in _VALID_THINKING:
                opts = "|".join(sorted(_VALID_THINKING))
                return f"当前: {session.thinking_level}\n用法: /think <{opts}>"
            await self._sm.update_thinking(session, arg)
            return f"思考深度已设置为 {arg}"

        if cmd == "/compact":
            return await self._handle_compact(session)

        if cmd == "/help":
            return _HELP_TEXT

        return f"未知命令: {cmd}\n输入 /help 查看可用命令。"

    async def _handle_compact(self, session: Session) -> str:
        """Generate L0/L1 summaries and persist to DB."""
        if not session.messages or len(session.messages) < 8:
            return "当前会话消息太少，无需压缩。"

        if not self._config or not self._router or not self._compressor or not self._store:
            return "未配置压缩模型。请在配置中设置 agent.summarizer.model 和对应的 API Key。"

        summarizer_cfg = self._config.agent.summarizer
        if not summarizer_cfg.enabled:
            return "上下文压缩功能已禁用。可在配置中设置 agent.summarizer.enabled = true 启用。"

        from whaleclaw.providers.base import Message
        from whaleclaw.sessions.context_window import estimate_tokens

        non_system = [m for m in session.messages if m.role != "system"]
        protected_count = min(6, len(non_system))
        to_compress = non_system[:-protected_count] if protected_count < len(non_system) else []

        if len(to_compress) < 8:
            return "保护区外的消息太少，无需压缩。"

        msgs = [Message(role=m.role, content=m.content) for m in to_compress]

        msg_rows = await self._store.get_messages(session.id)
        if not msg_rows:
            return "无法读取消息记录。"

        start_id = msg_rows[0].id
        end_idx = min(len(to_compress), len(msg_rows))
        end_id = msg_rows[end_idx - 1].id

        ok = await self._compressor.compress_segment(
            session_id=session.id,
            messages=msgs,
            msg_id_start=start_id,
            msg_id_end=end_id,
            store=self._store,
            router=self._router,
            model=summarizer_cfg.model,
        )

        if not ok:
            return "压缩失败，请检查压缩模型的 API Key 是否已配置。"

        summaries = await self._store.get_summaries(session.id)
        l0 = [s for s in summaries if s.level == "L0"]
        l1 = [s for s in summaries if s.level == "L1"]
        l0_tokens = sum(s.token_count for s in l0)
        l1_tokens = sum(s.token_count for s in l1)

        original_tokens = sum(estimate_tokens(m.content) for m in msgs)

        return (
            f"已为 {len(to_compress)} 条历史消息生成分层摘要。\n"
            f"  L0 (一句话): ~{l0_tokens} tokens\n"
            f"  L1 (概要):   ~{l1_tokens} tokens\n"
            f"  原始:        ~{original_tokens} tokens\n"
            f"保留最近 {protected_count} 条消息不变。\n"
            f"摘要已存入数据库，后续对话将自动加载。"
        )
