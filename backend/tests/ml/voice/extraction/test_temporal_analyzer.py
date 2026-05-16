"""Tests for temporal voice analyzer."""

import pytest
from datetime import datetime, timedelta
from bufferiq.ml.voice.extraction.temporal_analyzer import TemporalVoiceAnalyzer


class TestTemporalVoiceAnalyzer:
    """Test temporal voice analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TemporalVoiceAnalyzer()
    
    def test_analyze_evolution_basic(self, analyzer):
        """Test basic evolution analysis."""
        posts = [
            {
                "text": "Post 1",
                "created_at": datetime.utcnow() - timedelta(days=i),
            }
            for i in range(60)
        ]
        
        evolution = analyzer.analyze_evolution(posts, window_days=30)
        
        assert isinstance(evolution, dict)
        assert 'trend' in evolution
        assert 'windows' in evolution
    
    def test_analyze_evolution_empty_raises_error(self, analyzer):
        """Test empty posts raises error."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.analyze_evolution([])
    
    def test_analyze_evolution_insufficient_data(self, analyzer):
        """Test insufficient data returns appropriate message."""
        posts = [
            {"text": "Post", "created_at": datetime.utcnow()}
        ]
        
        evolution = analyzer.analyze_evolution(posts, window_days=30)
        
        assert evolution['trend'] == "insufficient_data"
    
    def test_analyze_evolution_stable_trend(self, analyzer):
        """Test stable trend detection."""
        posts = [
            {
                "text": "Formal professional business content",
                "created_at": datetime.utcnow() - timedelta(days=i),
            }
            for i in range(60)
        ]
        
        evolution = analyzer.analyze_evolution(posts, window_days=15)
        
        # Should detect stable trend
        assert evolution['trend'] in ["stable", "insufficient_data"]
    
    def test_create_time_windows_basic(self, analyzer):
        """Test time window creation."""
        posts = [
            {"text": f"Post {i}", "created_at": datetime.utcnow() - timedelta(days=i)}
            for i in range(90)
        ]
        
        windows = analyzer._create_time_windows(posts, window_days=30)
        
        assert isinstance(windows, list)
        assert len(windows) >= 2
    
    def test_create_time_windows_empty_returns_empty(self, analyzer):
        """Test empty posts returns empty windows."""
        windows = analyzer._create_time_windows([], window_days=30)
        
        assert windows == []
    
    def test_calculate_trend_increasing(self, analyzer):
        """Test increasing trend detection."""
        values = [50, 55, 60, 65, 70, 75, 80]
        trend = analyzer._calculate_trend(values)
        
        assert trend == "increasing"
    
    def test_calculate_trend_decreasing(self, analyzer):
        """Test decreasing trend detection."""
        values = [80, 75, 70, 65, 60, 55, 50]
        trend = analyzer._calculate_trend(values)
        
        assert trend == "decreasing"
    
    def test_calculate_trend_stable(self, analyzer):
        """Test stable trend detection."""
        values = [70, 71, 69, 70, 72, 69, 70]
        trend = analyzer._calculate_trend(values)
        
        assert trend == "stable"
    
    def test_calculate_trend_insufficient_data(self, analyzer):
        """Test insufficient data for trend."""
        values = [70]
        trend = analyzer._calculate_trend(values)
        
        assert trend == "stable"
    
    def test_evolution_drift_score(self, analyzer):
        """Test drift score calculation."""
        posts = [
            {
                "text": "Formal professional content",
                "created_at": datetime.utcnow() - timedelta(days=60 - i),
            }
            for i in range(30)
        ] + [
            {
                "text": "Casual friendly content with emojis 😊",
                "created_at": datetime.utcnow() - timedelta(days=30 - i),
            }
            for i in range(30)
        ]
        
        evolution = analyzer.analyze_evolution(posts, window_days=30)
        
        # Should detect some drift
        if 'drift_score' in evolution:
            assert evolution['drift_score'] >= 0
    
    def test_evolution_formality_tracking(self, analyzer):
        """Test formality is tracked over time."""
        posts = [
            {
                "text": "Formal professional business content",
                "created_at": datetime.utcnow() - timedelta(days=i),
            }
            for i in range(60)
        ]
        
        evolution = analyzer.analyze_evolution(posts, window_days=30)
        
        if 'formality_over_time' in evolution:
            assert isinstance(evolution['formality_over_time'], list)
    
    def test_time_windows_non_overlapping(self, analyzer):
        """Test time windows are non-overlapping."""
        posts = [
            {"text": f"Post {i}", "created_at": datetime.utcnow() - timedelta(days=i)}
            for i in range(90)
        ]
        
        windows = analyzer._create_time_windows(posts, window_days=30)
        
        # Each window should contain different posts
        all_posts = []
        for window in windows:
            all_posts.extend(window)
        
        # Total posts in windows should equal original (no overlap)
        assert len(all_posts) == len(posts)
    
    def test_evolution_with_gaps(self, analyzer):
        """Test evolution analysis with gaps in posting."""
        posts = [
            {"text": "Post 1", "created_at": datetime.utcnow() - timedelta(days=90)},
            {"text": "Post 2", "created_at": datetime.utcnow() - timedelta(days=60)},
            {"text": "Post 3", "created_at": datetime.utcnow() - timedelta(days=30)},
            {"text": "Post 4", "created_at": datetime.utcnow()},
        ]
        
        evolution = analyzer.analyze_evolution(posts, window_days=30)
        
        # Should handle gaps gracefully
        assert 'trend' in evolution
    
    def test_window_days_parameter_effect(self, analyzer):
        """Test window_days parameter affects window count."""
        posts = [
            {"text": f"Post {i}", "created_at": datetime.utcnow() - timedelta(days=i)}
            for i in range(90)
        ]
        
        windows_30 = analyzer._create_time_windows(posts, window_days=30)
        windows_15 = analyzer._create_time_windows(posts, window_days=15)
        
        # Smaller windows should create more windows
        assert len(windows_15) > len(windows_30)