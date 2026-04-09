"""
Track synchronization progress and state.

Manages sync job lifecycle and progress reporting.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.domain.models import SyncJob

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track synchronization progress.

    Manages sync job records and progress updates.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize progress tracker.

        Args:
            session: Database session
        """
        self.session = session

    async def create_job(
        self,
        user_id: int,
        sync_type: str,
        total_items: int = 0,
        # metadata: Optional[Dict[str, Any]] = None,
    ) -> SyncJob:
        """
        Create new sync job.

        Args:
            user_id: User ID
            sync_type: Type of sync (initial, incremental, etc.)
            total_items: Expected total items to process
            metadata: Additional job metadata

        Returns:
            Created sync job
        """
        job = SyncJob(
            user_id=user_id,
            sync_type=sync_type,
            status="pending",
            total_items=total_items,
            processed_items=0,
            failed_items=0,
            # metadata=metadata or {},
        )

        self.session.add(job)
        await self.session.flush()
        await self.session.refresh(job)

        logger.info(f"Created sync job {job.id} for user {user_id}")
        return job

    async def start_job(self, job_id: int) -> None:
        """
        Mark job as started.

        Args:
            job_id: Sync job ID
        """
        result = await self.session.execute(select(SyncJob).where(SyncJob.id == job_id))
        job = result.scalar_one()

        job.status = "running"
        job.started_at = datetime.utcnow()

        await self.session.flush()
        logger.info(f"Started sync job {job_id}")

    async def update_progress(
        self,
        job_id: int,
        processed: int,
        failed: int = 0,
        cursor: Optional[str] = None,
    ) -> None:
        """
        Update job progress.

        Args:
            job_id: Sync job ID
            processed: Number of items processed
            failed: Number of items failed
            cursor: Current pagination cursor
        """
        result = await self.session.execute(select(SyncJob).where(SyncJob.id == job_id))
        job = result.scalar_one()

        job.processed_items = (job.processed_items or 0) + processed
        job.failed_items = (job.failed_items or 0) + failed

        # if cursor is not None:
        # Safely update metadata without touching class variables
        # if job.metadata is None or not isinstance(job.metadata, dict):
        #    job.metadata = {}
        # job.metadata["cursor"] = cursor

        await self.session.flush()

        # Log progress safely
        total_items = job.total_items or 0
        processed_items = job.processed_items or 0
        if total_items > 0:
            progress_pct = (processed_items / total_items) * 100
            logger.info(
                f"Job {job_id} progress: {processed_items}/{total_items} ({progress_pct:.1f}%)"
            )
        else:
            logger.info(f"Job {job_id} progress: {processed_items} items processed")

    async def complete_job(self, job_id: int) -> None:
        """
        Mark job as completed.

        Args:
            job_id: Sync job ID
        """
        result = await self.session.execute(select(SyncJob).where(SyncJob.id == job_id))
        job = result.scalar_one()

        job.status = "completed"
        job.completed_at = datetime.utcnow()

        await self.session.flush()
        logger.info(
            f"Completed sync job {job_id}: {job.processed_items or 0} processed, {job.failed_items or 0} failed"
        )

    async def fail_job(self, job_id: int, error_message: str) -> None:
        """
        Mark job as failed.

        Args:
            job_id: Sync job ID
            error_message: Error message
        """
        result = await self.session.execute(select(SyncJob).where(SyncJob.id == job_id))
        job = result.scalar_one()

        job.status = "failed"
        job.error_message = error_message
        job.completed_at = datetime.utcnow()

        await self.session.flush()
        logger.error(f"Failed sync job {job_id}: {error_message}")

    async def get_job(self, job_id: int) -> Optional[SyncJob]:
        """
        Get sync job by ID.

        Args:
            job_id: Sync job ID

        Returns:
            Sync job or None
        """
        result = await self.session.execute(select(SyncJob).where(SyncJob.id == job_id))
        return result.scalar_one_or_none()

    async def get_last_successful_sync(
        self, user_id: int, sync_type: str
    ) -> Optional[SyncJob]:
        """
        Get last successful sync job for user.

        Args:
            user_id: User ID
            sync_type: Sync type

        Returns:
            Last successful sync job or None
        """
        # Ensure SyncJob has proper type hints for sync_type
        result = await self.session.execute(
            select(SyncJob)
            .where(SyncJob.user_id == user_id)
            .where(SyncJob.sync_type == sync_type)
            .where(SyncJob.status == "completed")
            .order_by(SyncJob.completed_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_recent_jobs(self, user_id: int, limit: int = 10) -> list[SyncJob]:
        """
        Get recent sync jobs for user.

        Args:
            user_id: User ID
            limit: Maximum number of jobs to return

        Returns:
            List of recent sync jobs
        """
        result = await self.session.execute(
            select(SyncJob)
            .where(SyncJob.user_id == user_id)
            .order_by(SyncJob.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    def calculate_eta(self, job: SyncJob) -> Optional[int]:
        """
        Calculate estimated time remaining (seconds).

        Args:
            job: Sync job

        Returns:
            Estimated seconds remaining or None
        """
        if not job.started_at or not job.total_items or not job.processed_items:
            return None

        elapsed = (datetime.utcnow() - job.started_at).total_seconds()
        if elapsed <= 0:
            return None

        rate = job.processed_items / elapsed  # items per second
        remaining_items = job.total_items - job.processed_items

        if rate > 0:
            return int(remaining_items / rate)

        return None
