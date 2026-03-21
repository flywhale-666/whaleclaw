"""Cron management tool — list/add/remove/trigger jobs.

Supports three schedule kinds:
- at: one-shot at N minutes from now (or ISO timestamp)
- cron: recurring via 5-field cron expression
- every: recurring at fixed interval (minutes)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import uuid4

from whaleclaw.cron.scheduler import CronAction, CronJob, CronScheduler, Schedule
from whaleclaw.tools.base import Tool, ToolDefinition, ToolParameter, ToolResult


class CronManageTool(Tool):
    """Full cron tool: list, add, remove, trigger with at/cron/every support."""

    def __init__(self, scheduler: CronScheduler) -> None:
        self._scheduler = scheduler
        self.current_session_id: str = ""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="cron",
            description=(
                "管理定时任务。"
                "list: 查看所有定时任务；add: 创建定时任务；"
                "remove: 删除定时任务；trigger: 立即触发某个任务。"
            ),
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="要执行的操作。",
                    enum=["list", "add", "remove", "trigger"],
                ),
                ToolParameter(
                    name="name",
                    type="string",
                    description="任务名称（add 时使用）。",
                    required=False,
                ),
                ToolParameter(
                    name="message",
                    type="string",
                    description="任务触发时要发送的消息内容（add 时必填）。",
                    required=False,
                ),
                ToolParameter(
                    name="schedule_kind",
                    type="string",
                    description="调度类型：at=一次性，cron=cron表达式，every=固定间隔。",
                    required=False,
                    enum=["at", "cron", "every"],
                ),
                ToolParameter(
                    name="minutes",
                    type="integer",
                    description="at: 几分钟后触发；every: 间隔几分钟。与 at_time 二选一。",
                    required=False,
                ),
                ToolParameter(
                    name="at_time",
                    type="string",
                    description=(
                        "指定触发的绝对时间（schedule_kind=at 时使用）。"
                        "支持格式：'HH:MM'、'YYYY-MM-DD HH:MM'、'MM-DD HH:MM'。"
                        "与 minutes 二选一，优先使用 at_time。"
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="cron_expr",
                    type="string",
                    description="5字段 cron 表达式，如 '0 7 * * *'（schedule_kind=cron 时使用）。",
                    required=False,
                ),
                ToolParameter(
                    name="job_id",
                    type="string",
                    description="任务 ID（remove/trigger 时使用）。",
                    required=False,
                ),
            ],
        )

    async def execute(self, **kwargs: object) -> ToolResult:
        action = str(kwargs.get("action", "")).lower()

        if action == "list":
            return await self._list()
        if action == "add":
            return await self._add(kwargs)
        if action == "remove":
            return await self._remove(kwargs)
        if action == "trigger":
            return await self._trigger(kwargs)
        return ToolResult(success=False, output="", error=f"未知操作: {action}")

    async def _list(self) -> ToolResult:
        jobs = await self._scheduler.list_jobs()
        if not jobs:
            return ToolResult(success=True, output="当前没有定时任务")
        lines: list[str] = []
        for j in jobs:
            sched_desc = self._describe_schedule(j)
            status = "✅" if j.enabled else "⏸️"
            lines.append(f"{status} {j.id}: {j.name} — {sched_desc}")
        return ToolResult(success=True, output="\n".join(lines))

    def _describe_schedule(self, job: CronJob) -> str:
        sched = job.schedule_obj
        if sched is None:
            return f"cron: {job.schedule}"
        if sched.kind == "at":
            return f"一次性: {sched.at.strftime('%m-%d %H:%M') if sched.at else '?'}"
        if sched.kind == "every":
            mins = sched.every_seconds // 60
            return f"每 {mins} 分钟" if mins > 0 else f"每 {sched.every_seconds} 秒"
        return f"cron: {sched.expr}"

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
        try:
            t = datetime.strptime(raw, "%H:%M").time()
            target = datetime.combine(now.date(), t)
            if target <= now:
                target += timedelta(days=1)
            return target
        except ValueError:
            pass
        return None

    async def _add(self, kwargs: dict[str, object]) -> ToolResult:
        name = str(kwargs.get("name", "")) or None
        message = str(kwargs.get("message", "")) or None
        kind = str(kwargs.get("schedule_kind", "at")).lower()
        raw_min = kwargs.get("minutes")
        raw_at = kwargs.get("at_time")
        cron_expr = str(kwargs.get("cron_expr", "")) or None

        if not message:
            return ToolResult(success=False, output="", error="缺少 message 参数")

        now = datetime.now()

        if kind == "at":
            target_time: datetime | None = None
            if raw_at is not None and str(raw_at).strip():
                target_time = self._parse_at_time(str(raw_at))
                if target_time is None:
                    return ToolResult(
                        success=False, output="",
                        error=f"无法解析时间 '{raw_at}'，支持格式：HH:MM / MM-DD HH:MM / YYYY-MM-DD HH:MM",
                    )
                if target_time <= now:
                    return ToolResult(
                        success=False, output="",
                        error=f"指定时间 {target_time.strftime('%m-%d %H:%M')} 已过，请设置未来的时间",
                    )
                confirm = f"将在 {target_time.strftime('%m-%d %H:%M')} 触发"
            elif raw_min is not None:
                try:
                    minutes = int(str(raw_min))
                except (TypeError, ValueError):
                    return ToolResult(success=False, output="", error="minutes 必须为整数")
                if minutes < 1:
                    return ToolResult(success=False, output="", error="minutes 必须大于 0")
                target_time = now + timedelta(minutes=minutes)
                confirm = f"{minutes} 分钟后提醒"
            else:
                return ToolResult(success=False, output="", error="at 类型需要 minutes 或 at_time 参数")
            sched = Schedule(kind="at", at=target_time)
            auto_name = name or f"提醒: {message[:20]}"
            one_shot = True

        elif kind == "cron":
            if not cron_expr:
                return ToolResult(success=False, output="", error="cron 类型需要 cron_expr 参数")
            parts = cron_expr.split()
            if len(parts) != 5:
                return ToolResult(
                    success=False, output="",
                    error="cron 表达式需要 5 个字段: 分 时 日 月 星期",
                )
            sched = Schedule(kind="cron", expr=cron_expr)
            auto_name = name or f"定时: {message[:20]}"
            one_shot = False
            confirm = f"按 cron 表达式 '{cron_expr}' 重复执行"

        elif kind == "every":
            if raw_min is None:
                return ToolResult(success=False, output="", error="every 类型需要 minutes 参数")
            try:
                minutes = int(str(raw_min))
            except (TypeError, ValueError):
                return ToolResult(success=False, output="", error="minutes 必须为整数")
            if minutes < 1:
                return ToolResult(success=False, output="", error="间隔必须大于 0 分钟")
            sched = Schedule(kind="every", every_seconds=minutes * 60)
            auto_name = name or f"循环: {message[:20]}"
            one_shot = False
            confirm = f"每 {minutes} 分钟执行一次"

        else:
            return ToolResult(success=False, output="", error=f"未知调度类型: {kind}")

        job = CronJob(
            id=f"cron-{uuid4().hex[:12]}",
            name=auto_name,
            schedule_obj=sched,
            action=CronAction(
                type="message",
                target=self.current_session_id or "user",
                payload={"text": message},
            ),
            enabled=True,
            created_at=now,
            one_shot=one_shot,
        )
        await self._scheduler.add_job(job)
        total = len(await self._scheduler.list_jobs())
        return ToolResult(success=True, output=f"已创建任务 {job.id}: {confirm}（定时任务 +1，合计 {total}）")

    async def _remove(self, kwargs: dict[str, object]) -> ToolResult:
        job_id = str(kwargs.get("job_id", "")) or None
        if not job_id:
            return ToolResult(success=False, output="", error="remove 需要 job_id")
        await self._scheduler.remove_job(job_id)
        total = len(await self._scheduler.list_jobs())
        return ToolResult(success=True, output=f"已删除任务: {job_id}（剩余 {total} 个定时任务）")

    async def _trigger(self, kwargs: dict[str, object]) -> ToolResult:
        job_id = str(kwargs.get("job_id", "")) or None
        if not job_id:
            return ToolResult(success=False, output="", error="trigger 需要 job_id")
        await self._scheduler.trigger_job(job_id)
        return ToolResult(success=True, output=f"已触发: {job_id}")
