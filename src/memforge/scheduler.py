"""Scheduled sync integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from memforge.runtime import SyncService

if TYPE_CHECKING:
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

SYNC_JOB_ID = "memforge-sync-all"
EXPIRY_JOB_ID = "memforge-retire-expired"
INDEX_HEALTH_JOB_ID = "memforge-index-health"
SOURCE_SCHEDULE_SCAN_JOB_ID = "memforge-source-schedule-scan"


def build_schedule_trigger(schedule: dict) -> CronTrigger:
    raw_time = schedule.get("time") or "02:00"
    hour_text, minute_text = raw_time.split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    timezone = schedule.get("timezone") or "UTC"
    frequency = schedule.get("frequency", "daily")

    if frequency == "hourly":
        return CronTrigger(minute=minute, timezone=timezone)
    if frequency == "weekly":
        return CronTrigger(
            day_of_week=int(schedule.get("day_of_week", 0)),
            hour=hour,
            minute=minute,
            timezone=timezone,
        )
    return CronTrigger(hour=hour, minute=minute, timezone=timezone)


class SyncScheduler:
    """Owns the APScheduler job that periodically syncs all active sources."""

    def __init__(self, db: "Database", sync_service: SyncService) -> None:
        self.db = db
        self.sync_service = sync_service
        self.scheduler = AsyncIOScheduler()

    async def start(self) -> None:
        self.scheduler.start()
        self._ensure_expiry_job()
        self._ensure_index_health_job()
        self._ensure_source_schedule_scan_job()
        await self.reload()

    async def reload(self) -> None:
        if self.scheduler.get_job(SYNC_JOB_ID):
            self.scheduler.remove_job(SYNC_JOB_ID)

        schedule = await self.db.get_schedule_config()
        if not schedule.get("enabled"):
            return

        self.scheduler.add_job(
            self.sync_service.run_all_active_sources,
            trigger=build_schedule_trigger(schedule),
            id=SYNC_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        logger.info(
            "Scheduled sync enabled: %s at %s %s",
            schedule.get("frequency"),
            schedule.get("time"),
            schedule.get("timezone"),
        )

    def _ensure_expiry_job(self) -> None:
        if self.scheduler.get_job(EXPIRY_JOB_ID):
            return
        self.scheduler.add_job(
            self._retire_expired_memories,
            trigger=CronTrigger(hour=0, minute=0, timezone="UTC"),
            id=EXPIRY_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    def _ensure_index_health_job(self) -> None:
        if self.scheduler.get_job(INDEX_HEALTH_JOB_ID):
            return
        self.scheduler.add_job(
            self._check_index_health,
            trigger=CronTrigger(hour=0, minute=30, timezone="UTC"),
            id=INDEX_HEALTH_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    def _ensure_source_schedule_scan_job(self) -> None:
        if self.scheduler.get_job(SOURCE_SCHEDULE_SCAN_JOB_ID):
            return
        self.scheduler.add_job(
            self._sync_due_sources,
            trigger=CronTrigger(minute="*", timezone="UTC"),
            id=SOURCE_SCHEDULE_SCAN_JOB_ID,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    async def _retire_expired_memories(self) -> None:
        retired_count = await self.sync_service.retire_expired_memories()
        if retired_count:
            logger.info("Retired %d expired memories", retired_count)

    async def _check_index_health(self) -> None:
        report = await self.sync_service.check_memory_index_health()
        if report.ok:
            logger.info("Memory index health check passed")
            return
        issue_counts: dict[str, int] = {}
        for issue in report.issues:
            issue_counts[issue.kind] = issue_counts.get(issue.kind, 0) + 1
        logger.error("Memory index health check found issues: %s", issue_counts)

    async def _sync_due_sources(self) -> None:
        for source in await self.db.claim_due_scheduled_sources(
            limit=50,
        ):
            source_id = str(source["id"])
            try:
                await self.sync_service.enqueue_source(source_id, trigger="schedule")
            except Exception:
                logger.exception("Scheduled source sync failed to start for %s", source_id)

    async def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
