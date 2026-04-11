"""Data loading utilities for analysis."""

from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.core.logging import get_logger
from bufferiq.domain.models import Channel, Post

logger = get_logger(__name__)


class DataLoader:
    """Load and prepare data from database for analysis."""

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize data loader.

        Args:
            session: Async database session
        """
        self.session = session

    async def load_posts(
        self,
        channel_id: Optional[int] = None,
        platform: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_engagement: Optional[int] = None,
        status: str = "sent",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load posts with flexible filtering.

        Args:
            channel_id: Filter by specific channel ID
            platform: Filter by platform (linkedin, twitter, facebook)
            start_date: Filter posts after this date
            end_date: Filter posts before this date
            min_engagement: Minimum total engagement required
            status: Post status (default: sent)
            limit: Maximum number of posts to return

        Returns:
            DataFrame with post data including calculated metrics

        Example:
            >>> loader = DataLoader(session)
            >>> df = await loader.load_posts(platform="linkedin", limit=100)
            >>> print(df.columns)
            Index(['id', 'content', 'platform', 'likes', ...])
        """
        # Build query
        query = (
            select(
                Post.id,
                Post.buffer_post_id,
                Post.content,
                Post.status,
                Post.scheduled_at,
                Post.published_at,
                Post.likes,
                Post.comments,
                Post.shares,
                Post.clicks,
                Post.impressions,
                Post.engagement_rate,
                Post.created_at,
                Post.updated_at,
                Channel.platform,
                Channel.handle,
            )
            .join(Channel, Post.channel_id == Channel.id)
            .where(Post.status == status)
        )

        # Apply filters
        conditions = []
        if channel_id is not None:
            conditions.append(Post.channel_id == channel_id)
        if platform is not None:
            conditions.append(Channel.platform == platform)
        if start_date is not None:
            conditions.append(Post.published_at >= start_date)
        if end_date is not None:
            conditions.append(Post.published_at <= end_date)

        if conditions:
            query = query.where(and_(*conditions))

        # Order by published date
        query = query.order_by(Post.published_at.desc())

        # Apply limit
        if limit is not None:
            query = query.limit(limit)

        # Execute query
        result = await self.session.execute(query)
        rows = result.fetchall()

        logger.info(
            f"Loaded {len(rows)} posts from database | channel_id={channel_id} | platform={platform}"
        )

        # Convert to DataFrame
        if not rows:
            logger.warning("No posts found matching criteria")
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "id": row.id,
                    "buffer_post_id": row.buffer_post_id,
                    "content": row.content,
                    "status": row.status,
                    "scheduled_at": row.scheduled_at,
                    "published_at": row.published_at,
                    "likes": row.likes or 0,
                    "comments": row.comments or 0,
                    "shares": row.shares or 0,
                    "clicks": row.clicks or 0,
                    "impressions": row.impressions or 0,
                    "engagement_rate": row.engagement_rate or 0.0,
                    "platform": row.platform,
                    "handle": row.handle,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]
        )

        # Calculate derived metrics
        df["total_engagement"] = df["likes"] + df["comments"] + df["shares"]
        df["content_length"] = df["content"].str.len()
        df["word_count"] = df["content"].str.split().str.len()

        # Extract temporal features
        if "published_at" in df.columns and not df.empty:
            df["hour"] = pd.to_datetime(df["published_at"]).dt.hour
            df["day_of_week"] = pd.to_datetime(df["published_at"]).dt.dayofweek
            df["day_name"] = pd.to_datetime(df["published_at"]).dt.day_name()
            df["date"] = pd.to_datetime(df["published_at"]).dt.date
            df["week"] = pd.to_datetime(df["published_at"]).dt.isocalendar().week
            df["month"] = pd.to_datetime(df["published_at"]).dt.month
            df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Apply minimum engagement filter if specified
        if min_engagement is not None:
            df = df[df["total_engagement"] >= min_engagement]
            logger.info(
                f"Applied minimum engagement filter | min_engagement={min_engagement} | remaining_posts={len(df)}"
            )

        return df

    async def load_engagement_metrics(
        self, channel_id: Optional[int] = None, group_by: str = "platform"
    ) -> pd.DataFrame:
        """
        Load pre-aggregated engagement metrics.

        Args:
            channel_id: Filter by specific channel ID
            group_by: Grouping column (platform, channel_id, date)

        Returns:
            DataFrame with aggregated statistics

        Example:
            >>> df = await loader.load_engagement_metrics(group_by="platform")
            >>> print(df[['platform', 'post_count', 'avg_engagement_rate']])
        """
        # Determine grouping column
        if group_by == "platform":
            group_col = Channel.platform
        elif group_by == "channel_id":
            group_col = Post.channel_id
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

        # Build aggregation query
        query = (
            select(
                group_col.label("group_key"),
                func.count(Post.id).label("post_count"),
                func.sum(Post.likes).label("total_likes"),
                func.sum(Post.comments).label("total_comments"),
                func.sum(Post.shares).label("total_shares"),
                func.sum(Post.impressions).label("total_impressions"),
                func.avg(Post.likes).label("avg_likes"),
                func.avg(Post.comments).label("avg_comments"),
                func.avg(Post.shares).label("avg_shares"),
                func.avg(Post.impressions).label("avg_impressions"),
                func.avg(Post.engagement_rate).label("avg_engagement_rate"),
            )
            .join(Channel, Post.channel_id == Channel.id)
            .where(Post.status == "sent")
            .group_by(group_col)
        )

        if channel_id is not None:
            query = query.where(Post.channel_id == channel_id)

        # Execute query
        result = await self.session.execute(query)
        rows = result.fetchall()

        logger.info(
            f"Loaded engagement metrics | group_by={group_by} | groups={len(rows)}"
        )

        # Convert to DataFrame
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    group_by: row.group_key,
                    "post_count": row.post_count,
                    "total_likes": row.total_likes or 0,
                    "total_comments": row.total_comments or 0,
                    "total_shares": row.total_shares or 0,
                    "total_impressions": row.total_impressions or 0,
                    "avg_likes": float(row.avg_likes or 0.0),
                    "avg_comments": float(row.avg_comments or 0.0),
                    "avg_shares": float(row.avg_shares or 0.0),
                    "avg_impressions": float(row.avg_impressions or 0.0),
                    "avg_engagement_rate": float(row.avg_engagement_rate or 0.0),
                }
                for row in rows
            ]
        )

        return df

    async def get_platform_stats(self) -> pd.DataFrame:
        """
        Get statistics by platform.

        Returns:
            DataFrame with per-platform statistics

        Example:
            >>> df = await loader.get_platform_stats()
            >>> print(df[['platform', 'post_count', 'avg_engagement_rate']])
        """
        return await self.load_engagement_metrics(group_by="platform")
