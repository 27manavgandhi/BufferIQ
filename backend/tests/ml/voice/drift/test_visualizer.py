"""Tests for drift visualizer."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from bufferiq.ml.voice.drift.visualizer import DriftVisualizer


class TestDriftVisualizer:
    """Test drift visualizer."""
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance."""
        return DriftVisualizer()
    
    @pytest.fixture
    def sample_timeline(self):
        """Create sample timeline DataFrame."""
        dates = [datetime.utcnow() - timedelta(days=i) for i in range(30)]
        return pd.DataFrame({
            'timestamp': dates,
            'formality': [70.0 + i * 0.5 for i in range(30)],
            'complexity': [50.0] * 30,
            'emoji_density': [2.0] * 30,
        })
    
    def test_create_timeline_chart_basic(self, visualizer, sample_timeline):
        """Test basic timeline chart creation."""
        chart_data = visualizer.create_timeline_chart(sample_timeline)
        
        assert isinstance(chart_data, dict)
        assert 'labels' in chart_data
        assert 'datasets' in chart_data
    
    def test_create_timeline_chart_empty_raises_error(self, visualizer):
        """Test empty timeline raises error."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty"):
            visualizer.create_timeline_chart(empty_df)
    
    def test_create_timeline_chart_none_raises_error(self, visualizer):
        """Test None timeline raises error."""
        with pytest.raises(ValueError, match="empty"):
            visualizer.create_timeline_chart(None)
    
    def test_timeline_chart_has_datasets(self, visualizer, sample_timeline):
        """Test timeline chart has datasets."""
        chart_data = visualizer.create_timeline_chart(sample_timeline)
        
        datasets = chart_data['datasets']
        assert len(datasets) == 3  # formality, complexity, emoji_density
    
    def test_timeline_chart_dataset_labels(self, visualizer, sample_timeline):
        """Test timeline chart dataset labels."""
        chart_data = visualizer.create_timeline_chart(sample_timeline)
        
        labels = [ds['label'] for ds in chart_data['datasets']]
        assert 'Formality' in labels
        assert 'Complexity' in labels
        assert 'Emoji Density' in labels
    
    def test_timeline_chart_labels_count(self, visualizer, sample_timeline):
        """Test timeline chart has correct number of labels."""
        chart_data = visualizer.create_timeline_chart(sample_timeline)
        
        assert len(chart_data['labels']) == 30
    
    def test_create_comparison_chart_basic(self, visualizer):
        """Test basic comparison chart creation."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 75.0, "complexity": 55.0, "emoji_density": 1.5}] * 15
        
        chart_data = visualizer.create_comparison_chart(baseline, recent)
        
        assert isinstance(chart_data, dict)
        assert 'labels' in chart_data
        assert 'datasets' in chart_data
    
    def test_comparison_chart_has_two_datasets(self, visualizer):
        """Test comparison chart has baseline and recent datasets."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 75.0, "complexity": 55.0, "emoji_density": 1.5}] * 15
        
        chart_data = visualizer.create_comparison_chart(baseline, recent)
        
        assert len(chart_data['datasets']) == 2
    
    def test_comparison_chart_dataset_labels(self, visualizer):
        """Test comparison chart dataset labels."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 75.0, "complexity": 55.0, "emoji_density": 1.5}] * 15
        
        chart_data = visualizer.create_comparison_chart(baseline, recent)
        
        labels = [ds['label'] for ds in chart_data['datasets']]
        assert 'Baseline' in labels
        assert 'Recent' in labels
    
    def test_create_distribution_data_basic(self, visualizer):
        """Test basic distribution data creation."""
        data = [{"formality": 70.0 + i} for i in range(30)]
        
        dist_data = visualizer.create_distribution_data(data, 'formality')
        
        assert isinstance(dist_data, dict)
        assert 'bins' in dist_data
        assert 'counts' in dist_data
        assert 'dimension' in dist_data
    
    def test_distribution_data_dimension_specified(self, visualizer):
        """Test distribution data has correct dimension."""
        data = [{"formality": 70.0}] * 30
        
        dist_data = visualizer.create_distribution_data(data, 'formality')
        
        assert dist_data['dimension'] == 'formality'
    
    def test_distribution_data_bins_count(self, visualizer):
        """Test distribution data has correct number of bins."""
        data = [{"formality": 70.0 + i} for i in range(30)]
        
        dist_data = visualizer.create_distribution_data(data, 'formality')
        
        # Should have 11 bin edges for 10 bins
        assert len(dist_data['bins']) == 11
        assert len(dist_data['counts']) == 10
    
    def test_distribution_counts_sum_to_total(self, visualizer):
        """Test distribution counts sum to data points."""
        data = [{"value": float(i)} for i in range(30)]
        
        dist_data = visualizer.create_distribution_data(data, 'value')
        
        assert sum(dist_data['counts']) == 30