"""Reminder tool — shortcut for one-shot cron jobs.

Kept as a separate tool so LLMs can use the simpler
``reminder(message, minutes)`` interface instead of the full cron tool.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from whaleclaw.cron.scheduler import CronAction, CronJob, CronScheduler, Schedule
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult


class ReminderTool(Tool):
    """Set a one-shot reminder or scheduled task N minutes from now."""

    def __init__(self, scheduler: CronScheduler) -> None:
        self._scheduler = scheduler
        self.current_session_id: str = ""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="reminder",
            description=(
                "设置定时提醒或定时任务。用户说'N分钟后做某事'时必须调用此工具。"
                "action='agent' 表示到时间后自动执行（message 作为任务指令发给 agent）；"
                "action='message'（默认）表示到时间后只通知用户。"
            ),
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="提醒内容，或 action='agent' 时的任务指令。",
                ),
                ToolParameter(
                    name="minutes",
                    type="integer",
                    description="从现在起多少分钟后触发。与 at_time 二选一。",
                    required=False,
                ),
                ToolParameter(
                    name="at_time",
                    type="string",
                    description=(
                        "指定触发的绝对时间。支持格式：'HH:MM'（今天或明天）、"
                        "'YYYY-MM-DD HH:MM'、'MM-DD HH:MM'。"
                        "用户说'明天上午8点'、'晚上10点半'等时使用此参数。"
                        "与 minutes 二选一，优先使用 at_time。"
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="action",
                    type="string",
                    description=(
                        "'agent' = 到时间后自动执行 message 中的任务；"
                        "'message' = 到时间后只发通知提醒用户（默认）。"
                    ),
                    required=False,
                ),
            ],
        )

    @staticmethod
    def _parse_at_time(raw: str) -> datetime | None:
        """解析绝对时间字符串，返回 datetime 或 None。"""
        raw = raw.strip()
        now = datetime.now()
        for fmt in ("%Y-%m-%d %H:%M", "%m-%d %H:%M"):
            try:
                return datetime.strptime(raw, fmt).replace(year=now.year)
            except ValueError:
                continue
        # HH:MM — 如果已过则自动推到明天
        try:
            t = datetime.strptime(raw, "%H:%M").time()
            target = datetime.combine(now.date(), t)
            if target <= now:
                target += timedelta(days=1)
            return target
        except ValueError:
            pass
        return None

    async def execute(self, **kwargs: object) -> ToolResult:
        message = str(kwargs.get("message", ""))
        raw_at = kwargs.get("at_time")
        raw_min = kwargs.get("minutes")

        now = datetime.now()
        target: datetime | None = None
        time_desc = ""

        if raw_at is not None and str(raw_at).strip():
            target = self._parse_at_time(str(raw_at))
            if target is None:
                return ToolResult(
                    success=False, output="",
                    error=f"无法解析时间 '{raw_at}'，支持格式：HH:MM / MM-DD HH:MM / YYYY-MM-DD HH:MM",
                )
            if target <= now:
                return ToolResult(
                    success=False, output="",
                    error=f"指定时间 {target.strftime('%m-%d %H:%M')} 已过，请设置未来的时间",
                )
            time_desc = target.strftime("%m-%d %H:%M")
        elif raw_min is not None:
            if isinstance(raw_min, bool):
                return ToolResult(success=False, output="", error="minutes 必须为整数")
            if isinstance(raw_min, int):
                minutes = raw_min
            elif isinstance(raw_min, float):
                minutes = int(raw_min)
            elif isinstance(raw_min, str):
                try:
                    minutes = int(raw_min.strip())
                except ValueError:
                    return ToolResult(success=False, output="", error="minutes 必须为整数")
            else:
                return ToolResult(success=False, output="", error="minutes 必须为整数")
            if minutes < 1:
                return ToolResult(success=False, output="", error="minutes 必须大于 0")
            target = now + timedelta(minutes=minutes)
            time_desc = f"{minutes} 分钟后"
        else:
            return ToolResult(success=False, output="", error="需要 minutes 或 at_time 参数")

        action_type = str(kwargs.get("action", "message")).strip().lower()
        if action_type not in ("message", "agent"):
            action_type = "message"

        job = CronJob(
            id=f"reminder-{uuid4().hex[:12]}",
            name=f"提醒: {message[:20]}",
            schedule_obj=Schedule(kind="at", at=target),
            action=CronAction(
                type=action_type,  # pyright: ignore[reportArgumentType]
                target=self.current_session_id or "user",
                payload={"text": message},
            ),
            enabled=True,
            created_at=now,
            one_shot=True,
        )
        await self._scheduler.add_job(job)
        total = len(await self._scheduler.list_jobs())
        trigger_str = target.strftime("%m-%d %H:%M")
        if action_type == "agent":
            return ToolResult(
                success=True,
                output=f"定时任务已设置，将在 {trigger_str} 自动执行: {message}（定时任务 +1，合计 {total}）",
            )
        return ToolResult(
            success=True,
            output=f"提醒已设置，将在 {trigger_str} 通知（定时任务 +1，合计 {total}）",
        )
