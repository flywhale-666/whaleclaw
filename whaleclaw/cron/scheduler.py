"""Cron job scheduler with cron/every/at schedule support."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any, Literal

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()

OnFireCallback = Callable[[str, "CronAction"], Awaitable[None]]
PersistSaveCallback = Callable[["CronJob"], Awaitable[None]]
PersistDeleteCallback = Callable[[str], Awaitable[None]]


def _parse_cron_field(field: str, min_val: int, max_val: int, now_val: int) -> bool:
    """Parse a single cron field supporting *, N, */N, N-M, and comma lists."""
    for part in field.split(","):
        part = part.strip()
        if part == "*":
            return True
        if "/" in part:
            base, step_s = part.split("/", 1)
            try:
                step = int(step_s)
            except ValueError:
                continue
            if step <= 0:
                continue
            if base == "*":
                if (now_val - min_val) % step == 0:
                    return True
            else:
                try:
                    start = int(base)
                except ValueError:
                    continue
                if now_val >= start and (now_val - start) % step == 0:
                    return True
        elif "-" in part:
            try:
                lo, hi = part.split("-", 1)
                if int(lo) <= now_val <= int(hi):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == now_val:
                    return True
            except ValueError:
                continue
    return False


class CronAction(BaseModel):
    """Action to perform when a cron job fires."""

    type: Literal["message", "agent", "webhook"]
    target: str
    payload: dict[str, Any] = {}


class Schedule(BaseModel):
    """Flexible schedule: cron expression, fixed interval, or one-shot timestamp."""

    kind: Literal["cron", "every", "at"]
    expr: str = ""
    every_seconds: int = 0
    at: datetime | None = None


class CronJob(BaseModel):
    """Cron job definition."""

    id: str
    name: str
    schedule: str = ""
    schedule_obj: Schedule | None = None
    action: CronAction
    enabled: bool = True
    created_at: datetime
    last_run: datetime | None = None
    next_run: datetime | None = None
    one_shot: bool = False


def _matches_cron_expr(expr: str, now: datetime) -> bool:
    """Check if a 5-field cron expression matches the given time."""
    parts = expr.split()
    if len(parts) != 5:
        return False
    return (
        _parse_cron_field(parts[0], 0, 59, now.minute)
        and _parse_cron_field(parts[1], 0, 23, now.hour)
        and _parse_cron_field(parts[2], 1, 31, now.day)
        and _parse_cron_field(parts[3], 1, 12, now.month)
        and _parse_cron_field(parts[4], 0, 6, now.weekday())
    )


class CronScheduler:
    """In-memory cron scheduler with 60s tick.

    Supports three schedule kinds:
    - cron: standard 5-field cron expression
    - every: fixed interval in seconds
    - at: one-shot at a specific timestamp
    """

    def __init__(self, on_fire: OnFireCallback | None = None) -> None:
        self._jobs: dict[str, CronJob] = {}
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._on_fire = on_fire
        self._on_persist_save: PersistSaveCallback | None = None
        self._on_persist_delete: PersistDeleteCallback | None = None

    def set_on_fire(self, callback: OnFireCallback) -> None:
        """Register callback invoked when a job fires."""
        self._on_fire = callback

    def set_persist(
        self,
        on_save: PersistSaveCallback,
        on_delete: PersistDeleteCallback,
    ) -> None:
        """注册持久化回调，add_job/remove_job 时自动触发。"""
        self._on_persist_save = on_save
        self._on_persist_delete = on_delete

    async def add_job(self, job: CronJob, *, persist: bool = True) -> None:
        self._jobs[job.id] = job
        if persist and self._on_persist_save is not None:
            try:
                await self._on_persist_save(job)
            except Exception as exc:
                logger.error("cron_persist_save_failed", job_id=job.id, error=str(exc))

    async def remove_job(self, job_id: str, *, persist: bool = True) -> None:
        self._jobs.pop(job_id, None)
        if persist and self._on_persist_delete is not None:
            try:
                await self._on_persist_delete(job_id)
            except Exception as exc:
                logger.error("cron_persist_delete_failed", job_id=job_id, error=str(exc))

    async def get_job(self, job_id: str) -> CronJob | None:
        return self._jobs.get(job_id)

    async def list_jobs(self) -> list[CronJob]:
        return list(self._jobs.values())

    async def update_job(self, job: CronJob) -> bool:
        """更新已有任务，返回是否成功。"""
        if job.id not in self._jobs:
            return False
        self._jobs[job.id] = job
        if self._on_persist_save is not None:
            try:
                await self._on_persist_save(job)
            except Exception as exc:
                logger.error("cron_persist_save_failed", job_id=job.id, error=str(exc))
        return True

    async def trigger_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if not job:
            return
        logger.info("cron_job_executing", job_id=job_id, action=job.action.model_dump())
        job = job.model_copy(update={"last_run": datetime.now()})
        self._jobs[job_id] = job

        if self._on_fire:
            try:
                await self._on_fire(job_id, job.action)
            except Exception as exc:
                logger.error("cron_job_fire_failed", job_id=job_id, error=str(exc))

        if job.one_shot:
            await self.remove_job(job_id)
            logger.info("cron_job_removed_one_shot", job_id=job_id)

    def _should_run(self, job: CronJob, now: datetime) -> bool:
        if not job.enabled:
            return False

        sched = job.schedule_obj
        if sched is None:
            return _matches_cron_expr(job.schedule, now)

        if sched.kind == "cron":
            return _matches_cron_expr(sched.expr or job.schedule, now)

        if sched.kind == "at":
            if sched.at is None:
                return False
            return now >= sched.at and (job.last_run is None or job.last_run < sched.at)

        if sched.kind == "every":
            if sched.every_seconds <= 0:
                return False
            if job.last_run is None:
                return True
            elapsed = (now - job.last_run).total_seconds()
            return elapsed >= sched.every_seconds

        return False

    async def _run_loop(self) -> None:
        while self._running:
            await asyncio.sleep(10)
            if not self._running:
                break
            now = datetime.now()
            for job in list(self._jobs.values()):
                if self._should_run(job, now):
                    await self.trigger_job(job.id)

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
