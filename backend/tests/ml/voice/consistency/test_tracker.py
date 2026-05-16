"""Tests for consistency tracker."""

import pytest
from datetime import datetime, timedelta
from bufferiq.ml.voice.consistency.tracker import ConsistencyTracker


class TestConsistencyTracker:
    """Test consistency tracker."""
    
    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return ConsistencyTracker()
    
    def test_add_score_basic(self, tracker):
        """Test adding basic score."""
        tracker.add_score(85.0, datetime.utcnow())
        
        assert len(tracker.history) == 1
    
    def test_add_multiple_scores(self, tracker):
        """Test adding multiple scores."""
        now = datetime.utcnow()
        
        tracker.add_score(80.0, now - timedelta(days=3))
        tracker.add_score(82.0, now - timedelta(days=2))
        tracker.add_score(85.0, now - timedelta(days=1))
        
        assert len(tracker.history) == 3
    
    def test_get_trend_insufficient_data(self, tracker):
        """Test trend with insufficient data."""
        trend = tracker.get_trend(days=30)
        
        assert trend['trend'] == 'insufficient_data'
        assert trend['average_score'] == 0.0
        assert trend['sample_size'] == 0
    
    def test_get_trend_basic(self, tracker):
        """Test basic trend calculation."""
        now = datetime.utcnow()
        
        for i in range(10):
            tracker.add_score(80.0 + i, now - timedelta(days=i))
        
        trend = tracker.get_trend(days=30)
        
        assert trend['trend'] in ['stable', 'improving', 'declining']
        assert trend['sample_size'] == 10
    
    def test_get_trend_improving(self, tracker):
        """Test improving trend detection."""
        now = datetime.utcnow()
        
        # Scores improving over time
        for i in range(10):
            tracker.add_score(70.0 + i * 2, now - timedelta(days=9 - i))
        
        trend = tracker.get_trend(days=30)
        
        assert trend['trend'] == 'improving'
    
    def test_get_trend_declining(self, tracker):
        """Test declining trend detection."""
        now = datetime.utcnow()
        
        # Scores declining over time
        for i in range(10):
            tracker.add_score(90.0 - i * 2, now - timedelta(days=9 - i))
        
        trend = tracker.get_trend(days=30)
        
        assert trend['trend'] == 'declining'
    
    def test_get_trend_stable(self, tracker):
        """Test stable trend detection."""
        now = datetime.utcnow()
        
        # Stable scores
        for i in range(10):
            tracker.add_score(80.0, now - timedelta(days=i))
        
        trend = tracker.get_trend(days=30)
        
        assert trend['trend'] == 'stable'
    
    def test_get_trend_filters_by_days(self, tracker):
        """Test trend filters by days parameter."""
        now = datetime.utcnow()
        
        # Old scores
        for i in range(5):
            tracker.add_score(60.0, now - timedelta(days=60 + i))
        
        # Recent scores
        for i in range(5):
            tracker.add_score(90.0, now - timedelta(days=i))
        
        trend = tracker.get_trend(days=30)
        
        # Should only include recent 5 scores
        assert trend['sample_size'] == 5
        assert trend['average_score'] == pytest.approx(90.0)
    
    def test_get_trend_calculates_min_max(self, tracker):
        """Test trend calculates min and max scores."""
        now = datetime.utcnow()
        
        tracker.add_score(70.0, now - timedelta(days=2))
        tracker.add_score(85.0, now - timedelta(days=1))
        tracker.add_score(75.0, now)
        
        trend = tracker.get_trend(days=30)
        
        assert trend['min_score'] == 70.0
        assert trend['max_score'] == 85.0
    
    def test_get_all_scores(self, tracker):
        """Test getting all scores."""
        now = datetime.utcnow()
        
        tracker.add_score(80.0, now - timedelta(days=2))
        tracker.add_score(85.0, now - timedelta(days=1))
        
        all_scores = tracker.get_all_scores()
        
        assert len(all_scores) == 2
        assert all_scores[0]['score'] == 80.0
        assert all_scores[1]['score'] == 85.0
    
    def test_get_all_scores_returns_copy(self, tracker):
        """Test get_all_scores returns copy not reference."""
        tracker.add_score(80.0, datetime.utcnow())
        
        scores = tracker.get_all_scores()
        scores.append({'score': 90.0, 'timestamp': datetime.utcnow()})
        
        # Should not affect original
        assert len(tracker.history) == 1
    
    def test_trend_average_score_calculation(self, tracker):
        """Test average score is calculated correctly."""
        now = datetime.utcnow()
        
        tracker.add_score(80.0, now - timedelta(days=2))
        tracker.add_score(90.0, now - timedelta(days=1))
        
        trend = tracker.get_trend(days=30)
        
        assert trend['average_score'] == pytest.approx(85.0)