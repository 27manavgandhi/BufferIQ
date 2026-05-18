"""Tests for gap detector."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from bufferiq.ml.gaps.detection.detector import (
    GapDetector,
    ContentGap,
    GapAnalysis,
    GapSeverity,
)


class TestGapDetector:
    """Test GapDetector class."""

    @pytest.fixture
    def mock_db(self):
        """Mock database session."""
        return Mock()

    @pytest.fixture
    def detector(self, mock_db):
        """Create detector instance."""
        return GapDetector(mock_db)

    @pytest.mark.asyncio
    async def test_detect_gaps_success(self, detector):
        """Test successful gap detection."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            competitor_ids=["comp1", "comp2"],
        )

        assert isinstance(analysis, GapAnalysis)
        assert analysis.total_gaps > 0
        assert 0 <= analysis.coverage_score <= 100

    @pytest.mark.asyncio
    async def test_detect_invalid_platform(self, detector):
        """Test detection with invalid platform."""
        with pytest.raises(ValueError, match="not supported"):
            await detector.detect(
                user_id="user123",
                platform="facebook",
            )

    @pytest.mark.asyncio
    async def test_platform_validation_linkedin(self, detector):
        """Test LinkedIn platform validation."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_platform_validation_twitter(self, detector):
        """Test Twitter platform validation."""
        analysis = await detector.detect(
            user_id="user123",
            platform="twitter",
        )
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_platform_validation_bluesky(self, detector):
        """Test Bluesky platform validation."""
        analysis = await detector.detect(
            user_id="user123",
            platform="bluesky",
        )
        assert analysis is not None

    def test_gap_severity_classification(self, detector):
        """Test gap severity classification."""
        # Create test gap
        gap = ContentGap(
            gap_id="test123",
            topic="Test Topic",
            keywords=["test"],
            description="Test",
            severity=GapSeverity.CRITICAL,
            priority_score=95.0,
            opportunity_score=90.0,
            competitor_coverage=5,
        )

        assert gap.severity == GapSeverity.CRITICAL
        assert gap.priority_score >= 90

    def test_gap_to_dict(self, detector):
        """Test gap serialization."""
        gap = ContentGap(
            gap_id="test123",
            topic="AI & ML",
            keywords=["AI", "ML"],
            description="Test gap",
            severity=GapSeverity.IMPORTANT,
            priority_score=85.0,
            opportunity_score=80.0,
            competitor_coverage=3,
        )

        gap_dict = gap.to_dict()

        assert isinstance(gap_dict, dict)
        assert gap_dict["gap_id"] == "test123"
        assert gap_dict["topic"] == "AI & ML"
        assert gap_dict["severity"] == "important"

    @pytest.mark.asyncio
    async def test_gap_analysis_structure(self, detector):
        """Test gap analysis structure."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )

        # Check structure
        assert hasattr(analysis, "total_gaps")
        assert hasattr(analysis, "critical_gaps")
        assert hasattr(analysis, "important_gaps")
        assert hasattr(analysis, "moderate_gaps")
        assert hasattr(analysis, "coverage_score")
        assert hasattr(analysis, "competitive_position")

    @pytest.mark.asyncio
    async def test_gap_prioritization(self, detector):
        """Test gap prioritization."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            competitor_ids=["comp1", "comp2"],
        )

        # Critical gaps should have highest priority
        if analysis.critical_gaps and analysis.moderate_gaps:
            critical_priorities = [g.priority_score for g in analysis.critical_gaps]
            moderate_priorities = [g.priority_score for g in analysis.moderate_gaps]

            assert min(critical_priorities) > max(moderate_priorities)

    @pytest.mark.asyncio
    async def test_competitive_position(self, detector):
        """Test competitive position determination."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )

        valid_positions = ["leader", "average", "behind"]
        assert analysis.competitive_position in valid_positions

    @pytest.mark.asyncio
    async def test_quick_wins_identification(self, detector):
        """Test quick wins identification."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )

        # Quick wins should have high opportunity and priority
        for gap in analysis.quick_wins:
            assert gap.opportunity_score > 70
            assert gap.priority_score > 80

    @pytest.mark.asyncio
    async def test_immediate_actions_generation(self, detector):
        """Test immediate actions generation."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )

        assert isinstance(analysis.immediate_actions, list)
        assert all(isinstance(action, str) for action in analysis.immediate_actions)

    def test_gap_analysis_serialization(self, detector):
        """Test gap analysis to_dict."""
        analysis = GapAnalysis(
            total_gaps=10,
            critical_gaps=[],
            important_gaps=[],
            moderate_gaps=[],
            coverage_score=75.0,
            competitive_position="average",
        )

        analysis_dict = analysis.to_dict()

        assert isinstance(analysis_dict, dict)
        assert analysis_dict["total_gaps"] == 10
        assert analysis_dict["coverage_score"] == 75.0

    @pytest.mark.asyncio
    async def test_detect_with_industry(self, detector):
        """Test detection with industry parameter."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            industry="technology",
        )

        assert analysis.total_gaps > 0

    @pytest.mark.asyncio
    async def test_detect_with_custom_lookback(self, detector):
        """Test detection with custom lookback period."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            lookback_days=30,
        )

        assert analysis is not None

    def test_severity_enum_values(self):
        """Test severity enum values."""
        assert GapSeverity.CRITICAL.value == "critical"
        assert GapSeverity.IMPORTANT.value == "important"
        assert GapSeverity.MODERATE.value == "moderate"
        assert GapSeverity.MINOR.value == "minor"

    @pytest.mark.asyncio
    async def test_empty_competitors_list(self, detector):
        """Test with empty competitors list."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            competitor_ids=[],
        )

        assert analysis.total_gaps >= 0

    @pytest.mark.asyncio
    async def test_multiple_competitors(self, detector):
        """Test with multiple competitors."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
            competitor_ids=["c1", "c2", "c3", "c4", "c5"],
        )

        assert analysis.total_gaps > 0

    def test_opportunity_score_range(self, detector):
        """Test opportunity score is in valid range."""
        gap = ContentGap(
            gap_id="test",
            topic="Test",
            keywords=[],
            description="",
            severity=GapSeverity.MODERATE,
            priority_score=50.0,
            opportunity_score=75.0,
            competitor_coverage=2,
        )

        assert 0 <= gap.opportunity_score <= 100

    def test_priority_score_range(self, detector):
        """Test priority score is in valid range."""
        gap = ContentGap(
            gap_id="test",
            topic="Test",
            keywords=[],
            description="",
            severity=GapSeverity.MODERATE,
            priority_score=65.0,
            opportunity_score=70.0,
            competitor_coverage=2,
        )

        assert 0 <= gap.priority_score <= 100

    @pytest.mark.asyncio
    async def test_total_gaps_calculation(self, detector):
        """Test total gaps equals sum of severity levels."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )

        calculated_total = (
            len(analysis.critical_gaps) +
            len(analysis.important_gaps) +
            len(analysis.moderate_gaps)
        )

        assert analysis.total_gaps == calculated_total

    def test_content_gap_confidence_default(self):
        """Test default confidence value."""
        gap = ContentGap(
            gap_id="test",
            topic="Test",
            keywords=[],
            description="",
            severity=GapSeverity.MODERATE,
            priority_score=50.0,
            opportunity_score=60.0,
            competitor_coverage=1,
        )

        assert gap.confidence == 0.8

    def test_trend_direction_values(self):
        """Test valid trend direction values."""
        valid_trends = ["rising", "stable", "falling", "growing"]

        gap = ContentGap(
            gap_id="test",
            topic="Test",
            keywords=[],
            description="",
            severity=GapSeverity.MODERATE,
            priority_score=50.0,
            opportunity_score=60.0,
            competitor_coverage=1,
            trend_direction="rising",
        )

        assert gap.trend_direction in valid_trends

    @pytest.mark.asyncio
    async def test_strategic_opportunities_ordering(self, detector):
        """Test strategic opportunities are ordered by priority."""
        analysis = await detector.detect(
            user_id="user123",
            platform="linkedin",
        )

        if len(analysis.strategic_opportunities) > 1:
            priorities = [g.priority_score for g in analysis.strategic_opportunities]
            assert priorities == sorted(priorities, reverse=True)