"""Tests for drift analyzer."""

import pytest
from datetime import datetime, timedelta
from bufferiq.ml.voice.drift.analyzer import DriftAnalyzer


class TestDriftAnalyzer:
    """Test drift analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return DriftAnalyzer()
    
    def test_analyze_drift_basic(self, analyzer):
        """Test basic drift analysis."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 75.0, "complexity": 55.0, "emoji_density": 1.5}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        assert isinstance(analysis, dict)
        assert 'statistical_differences' in analysis
        assert 'primary_dimension' in analysis
    
    def test_analyze_drift_empty_baseline_raises_error(self, analyzer):
        """Test empty baseline raises error."""
        with pytest.raises(ValueError, match="Insufficient"):
            analyzer.analyze_drift([], [{"formality": 70.0}])
    
    def test_analyze_drift_empty_recent_raises_error(self, analyzer):
        """Test empty recent raises error."""
        with pytest.raises(ValueError, match="Insufficient"):
            analyzer.analyze_drift([{"formality": 70.0}], [])
    
    def test_statistical_differences_calculated(self, analyzer):
        """Test statistical differences are calculated."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 80.0, "complexity": 55.0, "emoji_density": 1.0}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        diffs = analysis['statistical_differences']
        assert 'formality' in diffs
        assert 'complexity' in diffs
        assert 'emoji_density' in diffs
    
    def test_primary_dimension_identified(self, analyzer):
        """Test primary drift dimension is identified."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 90.0, "complexity": 51.0, "emoji_density": 1.9}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        # Formality has largest drift
        assert analysis['primary_dimension'] == 'formality'
    
    def test_temporal_pattern_analyzed(self, analyzer):
        """Test temporal pattern is analyzed."""
        baseline = [{"formality": 70.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 75.0, "timestamp": datetime.utcnow()}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        assert 'temporal_pattern' in analysis
    
    def test_root_cause_identified(self, analyzer):
        """Test root cause is identified."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 85.0, "complexity": 50.0, "emoji_density": 2.0}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        assert 'root_cause' in analysis
        assert isinstance(analysis['root_cause'], str)
    
    def test_severity_assessed(self, analyzer):
        """Test severity is assessed."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 75.0, "complexity": 50.0, "emoji_density": 2.0}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        assert 'severity' in analysis
        assert analysis['severity'] in ['low', 'medium', 'high', 'critical']
    
    def test_formality_increase_root_cause(self, analyzer):
        """Test root cause for formality increase."""
        baseline = [{"formality": 60.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 85.0, "complexity": 50.0, "emoji_density": 2.0}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        # Should identify formality as primary
        assert 'formality' in analysis['primary_dimension'].lower() or 'formality' in analysis['root_cause'].lower()
    
    def test_complexity_change_root_cause(self, analyzer):
        """Test root cause for complexity change."""
        baseline = [{"formality": 70.0, "complexity": 40.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 70.0, "complexity": 65.0, "emoji_density": 2.0}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        # Should identify complexity as primary
        assert analysis['primary_dimension'] == 'complexity'
    
    def test_emoji_density_change_root_cause(self, analyzer):
        """Test root cause for emoji density change."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 5.0}] * 30
        recent = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 0.5}] * 15
        
        analysis = analyzer.analyze_drift(baseline, recent)
        
        # Should identify emoji_density as primary
        assert analysis['primary_dimension'] == 'emoji_density'