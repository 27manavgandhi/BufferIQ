"""
Drift visualization utilities.

Creates visualizations of voice drift over time.
"""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime


class DriftVisualizer:
    """
    Create drift visualizations.
    
    Generates data structures for visualizing voice drift
    over time (frontend would render actual charts).
    
    Example:
```python
        visualizer = DriftVisualizer()
        chart_data = visualizer.create_timeline_chart(drift_timeline)
```
    """
    
    def __init__(self):
        """Initialize drift visualizer."""
        pass
    
    def create_timeline_chart(self, timeline_df: pd.DataFrame) -> Dict[str, any]:
        """
        Create timeline chart data.
        
        Args:
            timeline_df: DataFrame with drift timeline
        
        Returns:
            Chart data dictionary
        
        Raises:
            ValueError: If DataFrame is empty
        """
        if timeline_df is None or timeline_df.empty:
            raise ValueError("Cannot create chart from empty timeline")
        
        # Convert to chart-ready format
        chart_data = {
            'labels': [
                ts.isoformat() if isinstance(ts, datetime) else str(ts)
                for ts in timeline_df['timestamp'].tolist()
            ],
            'datasets': [
                {
                    'label': 'Formality',
                    'data': timeline_df['formality'].tolist(),
                    'color': '#3B82F6',
                },
                {
                    'label': 'Complexity',
                    'data': timeline_df['complexity'].tolist(),
                    'color': '#10B981',
                },
                {
                    'label': 'Emoji Density',
                    'data': timeline_df['emoji_density'].tolist(),
                    'color': '#F59E0B',
                },
            ],
        }
        
        return chart_data
    
    def create_comparison_chart(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> Dict[str, any]:
        """
        Create baseline vs recent comparison chart.
        
        Args:
            baseline: Baseline data
            recent: Recent data
        
        Returns:
            Comparison chart data
        """
        import statistics
        
        # Calculate averages
        baseline_formality = statistics.mean(d.get('formality', 0) for d in baseline)
        recent_formality = statistics.mean(d.get('formality', 0) for d in recent)
        
        baseline_complexity = statistics.mean(d.get('complexity', 0) for d in baseline)
        recent_complexity = statistics.mean(d.get('complexity', 0) for d in recent)
        
        baseline_emoji = statistics.mean(d.get('emoji_density', 0) for d in baseline)
        recent_emoji = statistics.mean(d.get('emoji_density', 0) for d in recent)
        
        return {
            'labels': ['Formality', 'Complexity', 'Emoji Density'],
            'datasets': [
                {
                    'label': 'Baseline',
                    'data': [baseline_formality, baseline_complexity, baseline_emoji],
                    'color': '#94A3B8',
                },
                {
                    'label': 'Recent',
                    'data': [recent_formality, recent_complexity, recent_emoji],
                    'color': '#EF4444',
                },
            ],
        }
    
    def create_distribution_data(self, data: List[Dict], dimension: str) -> Dict[str, any]:
        """
        Create distribution data for a specific dimension.
        
        Args:
            data: Voice data
            dimension: Dimension to visualize
        
        Returns:
            Distribution data
        """
        values = [d.get(dimension, 0) for d in data]
        
        # Create histogram bins
        min_val = min(values) if values else 0
        max_val = max(values) if values else 100
        
        num_bins = 10
        bin_width = (max_val - min_val) / num_bins
        
        bins = [min_val + i * bin_width for i in range(num_bins + 1)]
        
        # Count values in each bin
        histogram = [0] * num_bins
        for value in values:
            for i in range(num_bins):
                if bins[i] <= value < bins[i + 1]:
                    histogram[i] += 1
                    break
        
        return {
            'bins': bins,
            'counts': histogram,
            'dimension': dimension,
        }