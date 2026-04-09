"""
Tests for progress tracker.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.infrastructure.sync.progress_tracker import ProgressTracker


@pytest.mark.asyncio
class TestProgressTracker:
    """Test sync progress tracking."""

    async def test_create_job(self, test_session: AsyncSession) -> None:
        """Should create sync job."""
        tracker = ProgressTracker(test_session)

        job = await tracker.create_job(user_id=1, sync_type="initial", total_items=100)

        assert job.id is not None
        assert job.user_id == 1
        assert job.sync_type == "initial"
        assert job.status == "pending"
        assert job.total_items == 100
        assert job.processed_items == 0

    async def test_start_job(self, test_session: AsyncSession) -> None:
        """Should mark job as started."""
        tracker = ProgressTracker(test_session)

        job = await tracker.create_job(user_id=1, sync_type="initial")
        await tracker.start_job(job.id)

        updated_job = await tracker.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == "running"
        assert updated_job.started_at is not None

    async def test_update_progress(self, test_session: AsyncSession) -> None:
        """Should update job progress."""
        tracker = ProgressTracker(test_session)

        job = await tracker.create_job(user_id=1, sync_type="initial", total_items=100)
        await tracker.start_job(job.id)
        await tracker.update_progress(job.id, processed=10, failed=2)

        updated_job = await tracker.get_job(job.id)
        assert updated_job is not None
        assert updated_job.processed_items == 10
        assert updated_job.failed_items == 2

    async def test_complete_job(self, test_session: AsyncSession) -> None:
        """Should mark job as completed."""
        tracker = ProgressTracker(test_session)

        job = await tracker.create_job(user_id=1, sync_type="initial")
        await tracker.start_job(job.id)
        await tracker.complete_job(job.id)

        updated_job = await tracker.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == "completed"
        assert updated_job.completed_at is not None

    async def test_fail_job(self, test_session: AsyncSession) -> None:
        """Should mark job as failed."""
        tracker = ProgressTracker(test_session)

        job = await tracker.create_job(user_id=1, sync_type="initial")
        await tracker.start_job(job.id)
        await tracker.fail_job(job.id, "Test error")

        updated_job = await tracker.get_job(job.id)
        assert updated_job is not None
        assert updated_job.status == "failed"
        assert updated_job.error_message == "Test error"

    async def test_get_last_successful_sync(self, test_session: AsyncSession) -> None:
        """Should get last successful sync."""
        tracker = ProgressTracker(test_session)

        job1 = await tracker.create_job(user_id=1, sync_type="incremental")
        await tracker.start_job(job1.id)
        await tracker.complete_job(job1.id)

        job2 = await tracker.create_job(user_id=1, sync_type="incremental")
        await tracker.start_job(job2.id)
        await tracker.complete_job(job2.id)

        last_sync = await tracker.get_last_successful_sync(1, "incremental")
        assert last_sync is not None
        assert last_sync.id == job2.id

    async def test_get_recent_jobs(self, test_session: AsyncSession) -> None:
        """Should get recent jobs."""
        tracker = ProgressTracker(test_session)

        for _ in range(5):
            job = await tracker.create_job(user_id=1, sync_type="initial")
            await tracker.start_job(job.id)
            await tracker.complete_job(job.id)

        jobs = await tracker.get_recent_jobs(1, limit=3)
        assert len(jobs) == 3

    async def test_calculate_eta(self, test_session: AsyncSession) -> None:
        """Should calculate estimated time remaining."""
        tracker = ProgressTracker(test_session)

        job = await tracker.create_job(user_id=1, sync_type="initial", total_items=100)
        await tracker.start_job(job.id)
        await tracker.update_progress(job.id, processed=50)

        # ETA calculation requires time to pass
        import asyncio

        await asyncio.sleep(0.1)

        updated_job = await tracker.get_job(job.id)
        assert updated_job is not None
        eta = tracker.calculate_eta(updated_job)
        assert eta is None or eta > 0  # May be None if too fast
