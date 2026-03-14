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
                "Set a timer for N minutes from now. "
                "Use action='agent' to auto-execute (the message will be sent "
                "to the agent as a new task). "
                "Use action='message' (default) to just notify the user."
            ),
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Reminder text, or task instruction when action='agent'.",
                ),
                ToolParameter(
                    name="minutes",
                    type="integer",
                    description="Minutes from now to trigger.",
                ),
                ToolParameter(
                    name="action",
                    type="string",
                    description=(
                        "'agent' = auto-execute message as a task when time is up; "
                        "'message' = just send a notification (default)."
                    ),
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        message = str(kwargs.get("message", ""))
        raw_min = kwargs.get("minutes")
        if raw_min is None:
            return ToolResult(success=False, output="", error="缺少 minutes 参数")
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

        action_type = str(kwargs.get("action", "message")).strip().lower()
        if action_type not in ("message", "agent"):
            action_type = "message"

        now = datetime.now()
        target = now + timedelta(minutes=minutes)
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
        if action_type == "agent":
            return ToolResult(
                success=True,
                output=f"定时任务已设置，{minutes} 分钟后自动执行: {message}",
            )
        return ToolResult(
            success=True,
            output=f"提醒已设置，{minutes} 分钟后通知",
        )
