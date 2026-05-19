"""Tests for hashtag pattern detector."""

import pytest
from datetime import datetime, timedelta

from bufferiq.ml.hashtags.extraction.pattern_detector import (
    HashtagPatternDetector,
    HashtagPattern,
)
from bufferiq.ml.hashtags.extraction.extractor import ExtractedHashtag


class TestHashtagPatternDetector:
    """Test HashtagPatternDetector class."""

    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return HashtagPatternDetector(min_frequency=2)

    @pytest.fixture
    def sample_hashtags(self):
        """Create sample hashtags."""
        base_time = datetime.now()
        hashtags = []

        # Create frequent pattern
        for i in range(5):
            hashtags.append(
                ExtractedHashtag(
                    hashtag="ai",
                    original="#AI",
                    position=10,
                    context="context",
                    post_id=f"post_{i}",
                    platform="linkedin",
                    created_at=base_time - timedelta(days=i),
                    engagement=100,
                )
            )

        # Another frequent tag
        for i in range(3):
            hashtags.append(
                ExtractedHashtag(
                    hashtag="tech",
                    original="#Tech",
                    position=20,
                    context="context",
                    post_id=f"post_{i}",
                    platform="linkedin",
                    created_at=base_time - timedelta(days=i),
                    engagement=80,
                )
            )

        return hashtags

    def test_detect_patterns(self, detector, sample_hashtags):
        """Test pattern detection."""
        patterns = detector.detect_patterns(sample_hashtags)

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(p, HashtagPattern) for p in patterns)

    def test_detect_frequency_pattern(self, detector, sample_hashtags):
        """Test frequency pattern detection."""
        patterns = detector.detect_patterns(sample_hashtags)

        # Should detect frequent pattern
        freq_patterns = [p for p in patterns if p.pattern_type == "frequent"]
        assert len(freq_patterns) > 0

        freq_pattern = freq_patterns[0]
        assert freq_pattern.confidence > 0
        assert len(freq_pattern.examples) > 0

    def test_no_patterns_insufficient_data(self, detector):
        """Test no patterns with insufficient data."""
        # Only 1 hashtag (below min_frequency)
        hashtags = [
            ExtractedHashtag(
                hashtag="ai",
                original="#AI",
                position=10,
                context="context",
                post_id="post1",
                platform="linkedin",
                created_at=datetime.now(),
                engagement=100,
            )
        ]

        patterns = detector.detect_patterns(hashtags)

        # Might have timing pattern but no frequency pattern
        freq_patterns = [p for p in patterns if p.pattern_type == "frequent"]
        assert len(freq_patterns) == 0

    def test_detect_cooccurrence_pattern(self, detector):
        """Test co-occurrence pattern detection."""
        base_time = datetime.now()

        # Create hashtags that co-occur
        hashtags = []
        for i in range(3):
            # AI and Tech together
            hashtags.extend([
                ExtractedHashtag(
                    hashtag="ai",
                    original="#AI",
                    position=10,
                    context="context",
                    post_id=f"post_{i}",
                    platform="linkedin",
                    created_at=base_time,
                    engagement=100,
                ),
                ExtractedHashtag(
                    hashtag="tech",
                    original="#Tech",
                    position=20,
                    context="context",
                    post_id=f"post_{i}",
                    platform="linkedin",
                    created_at=base_time,
                    engagement=100,
                ),
            ])

        patterns = detector.detect_patterns(hashtags)

        cooccur_patterns = [p for p in patterns if p.pattern_type == "cooccurrence"]
        assert len(cooccur_patterns) > 0

    def test_detect_timing_pattern(self, detector, sample_hashtags):
        """Test timing pattern detection."""
        patterns = detector.detect_patterns(sample_hashtags)

        timing_patterns = [p for p in patterns if p.pattern_type == "timing"]
        assert len(timing_patterns) > 0

        timing_pattern = timing_patterns[0]
        assert "peak_hour" in timing_pattern.metadata

    def test_pattern_confidence(self, detector, sample_hashtags):
        """Test pattern confidence calculation."""
        patterns = detector.detect_patterns(sample_hashtags)

        for pattern in patterns:
            assert 0 <= pattern.confidence <= 1

    def test_pattern_examples(self, detector, sample_hashtags):
        """Test pattern examples."""
        patterns = detector.detect_patterns(sample_hashtags)

        freq_patterns = [p for p in patterns if p.pattern_type == "frequent"]
        if freq_patterns:
            pattern = freq_patterns[0]
            assert len(pattern.examples) > 0
            assert all(isinstance(ex, str) for ex in pattern.examples)

    def test_empty_hashtags(self, detector):
        """Test with empty hashtag list."""
        patterns = detector.detect_patterns([])

        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_pattern_metadata(self, detector, sample_hashtags):
        """Test pattern metadata."""
        patterns = detector.detect_patterns(sample_hashtags)

        for pattern in patterns:
            assert isinstance(pattern.metadata, dict)
            assert len(pattern.metadata) > 0