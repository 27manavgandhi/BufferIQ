"""
Data synchronization service.

Orchestrates syncing data from Buffer API to local database.
"""

import logging
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.domain.models import Channel, Organization, Post, User
from bufferiq.infrastructure.buffer.buffer_client import BufferClient
from bufferiq.infrastructure.buffer.queries import (
    GET_CHANNELS,
    GET_ORGANIZATIONS,
    GET_POSTS,
)
from bufferiq.infrastructure.sync.progress_tracker import ProgressTracker
from bufferiq.infrastructure.sync.transformers import BufferTransformer

logger = logging.getLogger(__name__)


class SyncService:
    """
    Synchronize data from Buffer API to database.

    Handles initial sync, incremental sync, and pagination.
    """

    def __init__(
        self,
        session: AsyncSession,
        client: BufferClient,
        transformer: BufferTransformer,
        progress_tracker: ProgressTracker,
    ) -> None:
        """
        Initialize sync service.

        Args:
            session: Database session
            client: Buffer API client
            transformer: Data transformer
            progress_tracker: Progress tracker
        """
        self.session = session
        self.client = client
        self.transformer = transformer
        self.tracker = progress_tracker

    async def initial_sync(self, user_id: int) -> int:
        """
        Perform initial sync for user.

        Fetches all organizations, channels, and posts.

        Args:
            user_id: User ID to sync

        Returns:
            Sync job ID

        Raises:
            ValueError: If user not found
        """
        logger.info(f"Starting initial sync for user {user_id}")

        # Get user
        user = await self._get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Set client user ID for rate limiting
        self.client.set_user_id(str(user_id))

        # Create sync job
        job = await self.tracker.create_job(
            user_id=user_id,
            sync_type="initial",
        )

        try:
            await self.tracker.start_job(job.id)

            # Sync organizations
            org_count = await self._sync_organizations(user)
            logger.info(f"Synced {org_count} organizations")

            # Sync channels for each organization
            channel_count = await self._sync_channels(user)
            logger.info(f"Synced {channel_count} channels")

            # Sync posts for each channel
            post_count = await self._sync_posts(user, job.id)
            logger.info(f"Synced {post_count} posts")

            # Mark job complete
            await self.tracker.complete_job(job.id)
            await self.session.commit()

            logger.info(
                f"Initial sync completed for user {user_id}: "
                f"{org_count} orgs, {channel_count} channels, {post_count} posts"
            )

            return job.id

        except Exception as e:
            await self.tracker.fail_job(job.id, str(e))
            await self.session.rollback()
            logger.error(f"Initial sync failed for user {user_id}: {e}")
            raise

    async def incremental_sync(self, user_id: int) -> int:
        """
        Perform incremental sync for user.

        Fetches only new/updated posts since last sync.

        Args:
            user_id: User ID to sync

        Returns:
            Sync job ID

        Raises:
            ValueError: If user not found
        """
        logger.info(f"Starting incremental sync for user {user_id}")

        # Get user
        user = await self._get_user(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Set client user ID
        self.client.set_user_id(str(user_id))

        # Get last successful sync
        last_sync = await self.tracker.get_last_successful_sync(user_id, "incremental")
        if last_sync and last_sync.completed_at:
            since = last_sync.completed_at
        else:
            since = datetime.utcnow() - timedelta(days=7)

        # Create sync job
        job = await self.tracker.create_job(
            user_id=user_id,
            sync_type="incremental",
            # metadata={"since": since.isoformat(), "started_by": "manual"},
            # metadata={"since": since.isoformat(), "started_by": "manual"},
        )

        try:
            await self.tracker.start_job(job.id)

            # Sync updated posts
            post_count = await self._sync_posts_since(user, job.id, since)
            logger.info(f"Synced {post_count} updated posts")

            # Mark job complete
            await self.tracker.complete_job(job.id)
            await self.session.commit()

            logger.info(
                f"Incremental sync completed for user {user_id}: {post_count} posts"
            )

            return job.id

        except Exception as e:
            await self.tracker.fail_job(job.id, str(e))
            await self.session.rollback()
            logger.error(f"Incremental sync failed for user {user_id}: {e}")
            raise

    async def _get_user(self, user_id: int) -> User | None:
        """Get user by ID."""
        result = await self.session.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    async def _sync_organizations(self, user: User) -> int:
        """Sync organizations for user."""
        # Fetch from API
        response = await self.client.query(GET_ORGANIZATIONS)
        orgs_data = response.get("organizations", [])

        count = 0
        for org_data in orgs_data:
            try:
                org = self.transformer.transform_organization(org_data, user.id)
                await self._upsert_organization(org)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to sync organization {org_data.get('id')}: {e}")

        await self.session.flush()
        return count

    async def _sync_channels(self, user: User) -> int:
        """Sync channels for all user's organizations."""
        # Get user's organizations
        result = await self.session.execute(
            select(Organization).where(Organization.user_id == user.id)
        )
        organizations = result.scalars().all()

        count = 0
        for org in organizations:
            # Fetch channels for organization
            response = await self.client.query(
                GET_CHANNELS, {"organizationId": org.buffer_org_id}
            )
            channels_data = response.get("channels", [])

            for channel_data in channels_data:
                try:
                    channel = self.transformer.transform_channel(channel_data, org.id)
                    await self._upsert_channel(channel)
                    count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to sync channel {channel_data.get('id')}: {e}"
                    )

        await self.session.flush()
        return count

    async def _sync_posts(self, user: User, job_id: int) -> int:
        """Sync all posts for user's channels."""
        # Get user's channels
        result = await self.session.execute(
            select(Channel).join(Organization).where(Organization.user_id == user.id)
        )
        channels = result.scalars().all()

        total_count = 0
        for channel in channels:
            count = await self._sync_channel_posts(channel, job_id)
            total_count += count

        return total_count

    async def _sync_channel_posts(self, channel: Channel, job_id: int) -> int:
        """Sync posts for specific channel with pagination."""
        count = 0
        offset = 0
        limit = 100  # Posts per page

        while True:
            # Fetch page of posts
            response = await self.client.query(
                GET_POSTS,
                {
                    "channelId": channel.buffer_channel_id,
                    "limit": limit,
                    "offset": offset,
                },
            )
            posts_data = response.get("posts", [])

            if not posts_data:
                break  # No more posts

            # Process posts
            for post_data in posts_data:
                try:
                    post = self.transformer.transform_post(post_data, channel.id)
                    await self._upsert_post(post)
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to sync post {post_data.get('id')}: {e}")
                    await self.tracker.update_progress(job_id, 0, 1)

            # Update progress
            await self.tracker.update_progress(job_id, len(posts_data), 0)
            await self.session.flush()

            # Check if more pages
            if len(posts_data) < limit:
                break  # Last page

            offset += limit

        return count

    async def _sync_posts_since(self, user: User, job_id: int, since: datetime) -> int:
        """Sync posts updated since timestamp."""
        # Get user's channels
        result = await self.session.execute(
            select(Channel).join(Organization).where(Organization.user_id == user.id)
        )
        channels = result.scalars().all()

        total_count = 0
        for channel in channels:
            # Fetch updated posts
            response = await self.client.query(
                GET_POSTS,
                {
                    "channelId": channel.buffer_channel_id,
                    "since": since.isoformat(),
                    "limit": 100,
                },
            )
            posts_data = response.get("posts", [])

            for post_data in posts_data:
                try:
                    post = self.transformer.transform_post(post_data, channel.id)
                    await self._upsert_post(post)
                    total_count += 1
                except Exception as e:
                    logger.warning(f"Failed to sync post {post_data.get('id')}: {e}")

            await self.tracker.update_progress(job_id, len(posts_data), 0)
            await self.session.flush()

        return total_count

    async def _upsert_organization(self, org: Organization) -> None:
        """Upsert organization (insert or update if exists)."""
        stmt = insert(Organization).values(
            user_id=org.user_id,
            buffer_org_id=org.buffer_org_id,
            name=org.name,
        )

        # On conflict, update name
        stmt = stmt.on_conflict_do_update(
            index_elements=["buffer_org_id"],
            set_={"name": stmt.excluded.name, "updated_at": datetime.utcnow()},
        )

        await self.session.execute(stmt)

    async def _upsert_channel(self, channel: Channel) -> None:
        """Upsert channel (insert or update if exists)."""
        stmt = insert(Channel).values(
            organization_id=channel.organization_id,
            buffer_channel_id=channel.buffer_channel_id,
            platform=channel.platform,
            handle=channel.handle,
            is_active=channel.is_active,
        )

        # On conflict, update details
        stmt = stmt.on_conflict_do_update(
            index_elements=["buffer_channel_id"],
            set_={
                "handle": stmt.excluded.handle,
                "is_active": stmt.excluded.is_active,
                "updated_at": datetime.utcnow(),
            },
        )

        await self.session.execute(stmt)

    async def _upsert_post(self, post: Post) -> None:
        """Upsert post (insert or update if exists)."""
        stmt = insert(Post).values(
            channel_id=post.channel_id,
            buffer_post_id=post.buffer_post_id,
            content=post.content,
            content_hash=post.content_hash,
            status=post.status,
            scheduled_at=post.scheduled_at,
            sent_at=post.sent_at,
            likes=post.likes,
            comments=post.comments,
            shares=post.shares,
            impressions=post.impressions,
            engagement_rate=post.engagement_rate,
        )

        # On conflict, update engagement metrics
        stmt = stmt.on_conflict_do_update(
            index_elements=["buffer_post_id"],
            set_={
                "status": stmt.excluded.status,
                "likes": stmt.excluded.likes,
                "comments": stmt.excluded.comments,
                "shares": stmt.excluded.shares,
                "impressions": stmt.excluded.impressions,
                "engagement_rate": stmt.excluded.engagement_rate,
                "updated_at": datetime.utcnow(),
            },
        )

        await self.session.execute(stmt)
