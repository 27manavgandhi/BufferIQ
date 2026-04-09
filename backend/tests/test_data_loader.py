"""Tests for data loader."""

import pytest
import pandas as pd
from datetime import datetime, timezone

from bufferiq.ml.analysis.data_loader import DataLoader


@pytest.mark.asyncio
async def test_load_posts_basic(test_session, sample_post):
    """Test basic post loading."""
    loader = DataLoader(test_session)
    df = await loader.load_posts(status="sent")

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "content" in df.columns
    assert "platform" in df.columns
    assert "engagement_rate" in df.columns


@pytest.mark.asyncio
async def test_load_posts_with_platform_filter(test_session, sample_post):
    """Test loading posts filtered by platform."""
    loader = DataLoader(test_session)
    df = await loader.load_posts(platform="linkedin", status="sent")

    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert (df["platform"] == "linkedin").all()


@pytest.mark.asyncio
async def test_load_posts_with_limit(test_session, sample_post):
    """Test loading posts with limit."""
    loader = DataLoader(test_session)
    df = await loader.load_posts(limit=5, status="sent")

    assert isinstance(df, pd.DataFrame)
    assert len(df) <= 5


@pytest.mark.asyncio
async def test_load_posts_calculates_derived_metrics(test_session, sample_post):
    """Test that derived metrics are calculated."""
    loader = DataLoader(test_session)
    df = await loader.load_posts(status="sent")

    if len(df) > 0:
        assert "total_engagement" in df.columns
        assert "content_length" in df.columns
        assert "word_count" in df.columns
        assert "hour" in df.columns
        assert "day_of_week" in df.columns


@pytest.mark.asyncio
async def test_load_posts_empty_result(test_session):
    """Test loading posts with no results."""
    loader = DataLoader(test_session)
    df = await loader.load_posts(platform="nonexistent", status="sent")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@pytest.mark.asyncio
async def test_load_engagement_metrics(test_session, sample_post):
    """Test loading aggregated engagement metrics."""
    loader = DataLoader(test_session)
    df = await loader.load_engagement_metrics(group_by="platform")

    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert "platform" in df.columns
        assert "post_count" in df.columns
        assert "avg_engagement_rate" in df.columns


@pytest.mark.asyncio
async def test_get_platform_stats(test_session, sample_post):
    """Test getting platform statistics."""
    loader = DataLoader(test_session)
    df = await loader.get_platform_stats()

    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert "platform" in df.columns
        assert "post_count" in df.columns


@pytest.mark.asyncio
async def test_load_posts_with_date_range(test_session, sample_post):
    """Test loading posts with date range filter."""
    loader = DataLoader(test_session)
    start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(2024, 12, 31, tzinfo=timezone.utc)

    df = await loader.load_posts(
        start_date=start_date, end_date=end_date, status="sent"
    )

    assert isinstance(df, pd.DataFrame)


@pytest.mark.asyncio
async def test_load_posts_with_min_engagement(test_session, sample_post):
    """Test loading posts with minimum engagement filter."""
    loader = DataLoader(test_session)
    df = await loader.load_posts(min_engagement=10, status="sent")

    assert isinstance(df, pd.DataFrame)
    if len(df) > 0:
        assert (df["total_engagement"] >= 10).all()