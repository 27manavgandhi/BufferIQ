"""
Tests for sync service.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.domain.models import User
from bufferiq.infrastructure.buffer.buffer_client import BufferClient
from bufferiq.infrastructure.sync.progress_tracker import ProgressTracker
from bufferiq.infrastructure.sync.sync_service import SyncService
from bufferiq.infrastructure.sync.transformers import BufferTransformer


@pytest.fixture
async def test_user(test_session: AsyncSession) -> User:
    """Create test user."""
    user = User(
        buffer_org_id="test_org",
        buffer_access_token="test_token",
        email="test@example.com",
        is_active=True,
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    return user


@pytest.mark.asyncio
class TestSyncService:
    """Test data synchronization service."""

    async def test_initial_sync_creates_job(
        self,
        test_session: AsyncSession,
        test_buffer_client: BufferClient,
        test_user: User,
        mocker,
    ) -> None:
        """Initial sync should create and complete job."""
        # Mock API responses
        mocker.patch.object(
            test_buffer_client,
            "query",
            return_value={
                "organizations": [],
                "channels": [],
                "posts": [],
            },
        )

        transformer = BufferTransformer()
        tracker = ProgressTracker(test_session)
        service = SyncService(test_session, test_buffer_client, transformer, tracker)

        # Run sync
        job_id = await service.initial_sync(test_user.id)

        # Verify job created
        job = await tracker.get_job(job_id)
        assert job is not None
        assert job.sync_type == "initial"
        assert job.status == "completed"

    async def test_initial_sync_user_not_found(
        self,
        test_session: AsyncSession,
        test_buffer_client: BufferClient,
    ) -> None:
        """Should raise error if user not found."""
        transformer = BufferTransformer()
        tracker = ProgressTracker(test_session)
        service = SyncService(test_session, test_buffer_client, transformer, tracker)

        with pytest.raises(ValueError, match="User .* not found"):
            await service.initial_sync(999)

    async def test_incremental_sync(
        self,
        test_session: AsyncSession,
        test_buffer_client: BufferClient,
        test_user: User,
        mocker,
    ) -> None:
        """Incremental sync should create job."""
        # Mock API responses
        mocker.patch.object(
            test_buffer_client,
            "query",
            return_value={"posts": []},
        )

        transformer = BufferTransformer()
        tracker = ProgressTracker(test_session)
        service = SyncService(test_session, test_buffer_client, transformer, tracker)

        # Run sync
        job_id = await service.incremental_sync(test_user.id)

        # Verify job created
        job = await tracker.get_job(job_id)
        assert job is not None
        assert job.sync_type == "incremental"
        assert job.status == "completed"
