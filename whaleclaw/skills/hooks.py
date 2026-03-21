"""SkillHooks 协议 — 技能即插即用的回调接口。

每个技能可以在自己的目录下放一个 hooks.py，实现 SkillHooks 协议中
自己关心的方法。未覆盖的方法使用 DefaultSkillHooks 的空实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from whaleclaw.agent.helpers.tool_guards import GuardDecision
    from whaleclaw.providers.base import ImageContent, Message, ToolCall
    from whaleclaw.sessions.manager import Session
    from whaleclaw.tools.base import ToolResult


@dataclass(slots=True)
class StageRule:
    """条件性 system message 注入规则。"""

    name: str
    condition: Any  # Callable[[dict[str, object], str], bool]
    system_hint: Any  # str | Callable[[dict[str, object], str], str]

    def get_hint(
        self,
        state: dict[str, object],
        user_message: str,
        session: Any = None,
    ) -> str:
        """返回 hint 文本，支持静态字符串和动态 callable。

        callable 签名可以是 (state, user_message) 或 (state, user_message, session)。
        """
        if callable(self.system_hint):
            import inspect
            sig = inspect.signature(self.system_hint)
            if len(sig.parameters) >= 3:
                return self.system_hint(state, user_message, session)
            return self.system_hint(state, user_message)
        return self.system_hint


@runtime_checkable
class SkillHooks(Protocol):
    """技能回调协议。所有方法都有默认空实现，技能只需覆盖自己关心的。"""

    # ── 参数守卫 ──────────────────────────────────────────────

    def build_param_guard_reply(self, state: dict[str, object]) -> str | None:
        """返回自定义参数守卫展示文案，None 表示走通用逻辑。"""
        ...

    def missing_required(
        self, state: dict[str, object], *, control_message_only: bool
    ) -> bool | None:
        """判断是否仍缺少必填参数，None 表示走通用逻辑。"""
        ...

    # ── 阶段规则 ──────────────────────────────────────────────

    @property
    def stage_rules(self) -> list[StageRule]:
        """返回阶段性 system message 注入规则列表。"""
        ...

    # ── 意图检测 ──────────────────────────────────────────────

    def is_execution_request(
        self, message: str, *, has_new_input_images: bool, session: Session | None
    ) -> bool | None:
        """判断用户消息是否为执行请求，None 表示走通用逻辑。"""
        ...

    def is_control_message(self, message: str) -> bool:
        """判断是否为纯控制消息（如切换模型/参数）。"""
        ...

    def is_activation_message(self, message: str) -> bool:
        """判断是否为技能激活消息（如"用xxx"）。"""
        ...

    # ── Guard 状态更新 ────────────────────────────────────────

    def update_guard_state(
        self,
        state: dict[str, object],
        message: str,
        images: list[ImageContent] | None,
        *,
        session: Session | None,
        has_new_input_images: bool,
    ) -> dict[str, object] | None:
        """自定义 guard 状态更新逻辑，返回更新后的 state 或 None 走通用。"""
        ...

    # ── 命令构建 ──────────────────────────────────────────────

    def build_command(self, state: dict[str, object], session: Session | None) -> str | None:
        """构建执行命令字符串，None 表示不构建。"""
        ...

    # ── 结果解析 ──────────────────────────────────────────────

    def parse_result(self, output: str) -> dict[str, str]:
        """从工具输出中解析结构化结果。"""
        ...

    # ── 执行约束 system message ───────────────────────────────

    def build_execution_system_message(
        self, session: Session | None
    ) -> Message | None:
        """构建执行阶段的约束 system message。"""
        ...

    def build_command_template_system_message(
        self, recommended_command: str
    ) -> Message | None:
        """构建命令模板 system message。"""
        ...

    # ── 回复后处理 ────────────────────────────────────────────

    def postprocess_reply(self, text: str, session: Session | None) -> str:
        """对最终回复文本做后处理。"""
        ...

    # ── 锁定状态展示 ──────────────────────────────────────────

    def build_lock_status_extra(self, session: Session | None) -> str:
        """锁定状态查询时附加的额外信息（如当前模型）。"""
        ...

    def build_already_locked_reply(self, session: Session | None) -> str | None:
        """激活消息命中但已锁定时的回复，None 走通用。"""
        ...

    # ── 渠道层 ────────────────────────────────────────────────

    @property
    def image_buffer_enabled(self) -> bool:
        """是否启用图片缓冲（飞书渠道）。"""
        ...

    def image_buffer_hint(self, image_labels: str) -> str | None:
        """图片缓冲时的提示文案，None 走通用。"""
        ...

    # ── 工具守卫 ──────────────────────────────────────────────

    def on_tool_failure(
        self, tc: ToolCall, result: ToolResult
    ) -> GuardDecision | None:
        """工具执行失败时的自定义守卫决策，None 走通用。"""
        ...

    # ── 工具修复 ──────────────────────────────────────────────

    def repair_tool_call(self, command: str) -> tuple[str, bool] | None:
        """修复工具调用参数，返回 (修复后命令, 是否修改) 或 None 走通用。"""
        ...

    # ── Bash 执行 ─────────────────────────────────────────────

    @property
    def long_running_script_pattern(self) -> re.Pattern[str] | None:
        """长时运行脚本的正则匹配模式。"""
        ...

    @property
    def long_running_timeout_seconds(self) -> int:
        """长时运行脚本的超时秒数。"""
        ...

    @property
    def parallel_limit(self) -> int:
        """批量并行执行的上限。"""
        ...

    @property
    def batch_delay_seconds(self) -> float:
        """批次间延迟秒数。"""
        ...

    def is_parallelizable_bash_call(self, tc: ToolCall) -> bool:
        """判断 bash 调用是否可并行。"""
        ...

    # ── 控制消息直接返回 ──────────────────────────────────────

    def handle_control_message(
        self, message: str, state: dict[str, object], session: Session | None
    ) -> str | None:
        """处理纯控制消息，返回直接回复文案或 None 继续常规流程。"""
        ...

    # ── 工具选择增强 ──────────────────────────────────────────

    def extra_tool_names(self) -> set[str]:
        """当技能有推荐命令时，需要额外加入的工具名。"""
        ...

    # ── Bash 执行后状态更新 ───────────────────────────────────

    def on_bash_success(
        self, tc: ToolCall, result: ToolResult, session: Session | None
    ) -> dict[str, Any] | None:
        """bash 执行成功后更新 session metadata，返回需要更新的字段或 None。"""
        ...


class DefaultSkillHooks:
    """SkillHooks 的默认空实现。技能 hooks 可以继承此类只覆盖需要的方法。"""

    def build_param_guard_reply(self, state: dict[str, object]) -> str | None:
        return None

    def missing_required(
        self, state: dict[str, object], *, control_message_only: bool
    ) -> bool | None:
        return None

    @property
    def stage_rules(self) -> list[StageRule]:
        return []

    def is_execution_request(
        self, message: str, *, has_new_input_images: bool, session: Session | None
    ) -> bool | None:
        return None

    def is_control_message(self, message: str) -> bool:
        return False

    def is_activation_message(self, message: str) -> bool:
        return False

    def update_guard_state(
        self,
        state: dict[str, object],
        message: str,
        images: list[ImageContent] | None,
        *,
        session: Session | None,
        has_new_input_images: bool,
    ) -> dict[str, object] | None:
        return None

    def build_command(self, state: dict[str, object], session: Session | None) -> str | None:
        return None

    def parse_result(self, output: str) -> dict[str, str]:
        return {}

    def build_execution_system_message(
        self, session: Session | None
    ) -> Message | None:
        return None

    def build_command_template_system_message(
        self, recommended_command: str
    ) -> Message | None:
        return None

    def postprocess_reply(self, text: str, session: Session | None) -> str:
        return text

    def build_lock_status_extra(self, session: Session | None) -> str:
        return ""

    def build_already_locked_reply(self, session: Session | None) -> str | None:
        return None

    @property
    def image_buffer_enabled(self) -> bool:
        return False

    def image_buffer_hint(self, image_labels: str) -> str | None:
        return None

    def on_tool_failure(
        self, tc: ToolCall, result: ToolResult
    ) -> GuardDecision | None:
        return None

    def repair_tool_call(self, command: str) -> tuple[str, bool] | None:
        return None

    @property
    def long_running_script_pattern(self) -> re.Pattern[str] | None:
        return None

    @property
    def long_running_timeout_seconds(self) -> int:
        return 300

    @property
    def parallel_limit(self) -> int:
        return 5

    @property
    def batch_delay_seconds(self) -> float:
        return 1.5

    def is_parallelizable_bash_call(self, tc: ToolCall) -> bool:
        return False

    def handle_control_message(
        self, message: str, state: dict[str, object], session: Session | None
    ) -> str | None:
        return None

    def extra_tool_names(self) -> set[str]:
        return set()

    def on_bash_success(
        self, tc: ToolCall, result: ToolResult, session: Session | None
    ) -> dict[str, Any] | None:
        return None


def get_skill_hooks(skill: Any) -> SkillHooks | None:
    """从 Skill 对象获取 hooks 实例，如果没有则返回 None。"""
    return getattr(skill, "hooks", None)


_all_hooks: list[SkillHooks] = []


def register_hooks(hooks: SkillHooks) -> None:
    """将 hooks 实例注册到全局列表（由 SkillManager._load_hooks 调用）。"""
    cls = type(hooks)
    if not any(type(h) is cls for h in _all_hooks):
        _all_hooks.append(hooks)


def get_all_hooks() -> list[SkillHooks]:
    """返回所有已注册的 hooks 实例。"""
    return list(_all_hooks)


__all__ = [
    "DefaultSkillHooks",
    "SkillHooks",
    "StageRule",
    "get_all_hooks",
    "get_skill_hooks",
    "register_hooks",
]
