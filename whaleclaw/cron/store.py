"""Persistent storage for cron jobs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import aiosqlite

from whaleclaw.cron.scheduler import CronAction, CronJob, Schedule


class CronStore:
    """aiosqlite-backed persistence for cron jobs."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._conn = await aiosqlite.connect(str(self._path))
        await self._ensure_table()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _ensure_table(self) -> None:
        if not self._conn:
            return
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cron_jobs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                schedule TEXT NOT NULL DEFAULT '',
                schedule_obj_json TEXT,
                action_json TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                one_shot INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                last_run TEXT,
                next_run TEXT
            )
            """
        )
        # 旧表可能缺少新增字段，逐个补上
        migrations: list[str] = [
            "ALTER TABLE cron_jobs ADD COLUMN schedule_obj_json TEXT",
            "ALTER TABLE cron_jobs ADD COLUMN one_shot INTEGER NOT NULL DEFAULT 0",
        ]
        for stmt in migrations:
            try:
                await self._conn.execute(stmt)
            except Exception:
                pass
        await self._conn.commit()

    def _job_to_row(self, job: CronJob) -> tuple[object, ...]:
        return (
            job.id,
            job.name,
            job.schedule,
            json.dumps(job.schedule_obj.model_dump(mode="json")) if job.schedule_obj else None,
            json.dumps(job.action.model_dump(mode="json")),
            1 if job.enabled else 0,
            1 if job.one_shot else 0,
            job.created_at.isoformat(),
            job.last_run.isoformat() if job.last_run else None,
            job.next_run.isoformat() if job.next_run else None,
        )

    def _row_to_job(self, row: tuple[object, ...]) -> CronJob:
        (
            id_,
            name,
            schedule,
            schedule_obj_json,
            action_json,
            enabled,
            one_shot,
            created_at,
            last_run,
            next_run,
        ) = row
        action_data = json.loads(str(action_json))
        sched_obj = None
        if schedule_obj_json:
            sched_obj = Schedule.model_validate(json.loads(str(schedule_obj_json)))
        return CronJob(
            id=str(id_),
            name=str(name),
            schedule=str(schedule) if schedule else "",
            schedule_obj=sched_obj,
            action=CronAction.model_validate(action_data),
            enabled=bool(enabled),
            one_shot=bool(one_shot),
            created_at=datetime.fromisoformat(str(created_at)),
            last_run=datetime.fromisoformat(str(last_run)) if last_run else None,
            next_run=datetime.fromisoformat(str(next_run)) if next_run else None,
        )

    async def save_job(self, job: CronJob) -> None:
        if not self._conn:
            return
        row = self._job_to_row(job)
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO cron_jobs
            (id, name, schedule, schedule_obj_json, action_json,
             enabled, one_shot, created_at, last_run, next_run)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        await self._conn.commit()

    async def load_jobs(self) -> list[CronJob]:
        if not self._conn:
            return []
        cursor = await self._conn.execute(
            "SELECT id, name, schedule, schedule_obj_json, action_json,"
            " enabled, one_shot, created_at, last_run, next_run FROM cron_jobs"
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(r) for r in rows]

    async def delete_job(self, job_id: str) -> None:
        if not self._conn:
            return
        await self._conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        await self._conn.commit()
