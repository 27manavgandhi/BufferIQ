"""
Drift analysis and root cause identification.

Analyzes drift patterns and identifies root causes.
"""

from typing import List, Dict, Optional
import statistics
from datetime import datetime


class DriftAnalyzer:
    """
    Analyze drift patterns and identify root causes.
    
    Provides detailed analysis of voice drift to help
    understand why drift occurred.
    
    Example:
```python
        analyzer = DriftAnalyzer()
        analysis = analyzer.analyze_drift(baseline, recent)
        print(f"Root cause: {analysis['root_cause']}")
```
    """
    
    def __init__(self):
        """Initialize drift analyzer."""
        pass
    
    def analyze_drift(
        self, baseline_data: List[Dict], recent_data: List[Dict]
    ) -> Dict[str, any]:
        """
        Analyze drift patterns.
        
        Args:
            baseline_data: Baseline voice data
            recent_data: Recent voice data
        
        Returns:
            Analysis results dictionary
        
        Raises:
            ValueError: If data is insufficient
        """
        if not baseline_data or not recent_data:
            raise ValueError("Insufficient data for drift analysis")
        
        # Calculate statistical differences
        stat_diff = self._calculate_statistical_differences(
            baseline_data, recent_data
        )
        
        # Identify primary drift dimension
        primary_dimension = self._identify_primary_dimension(stat_diff)
        
        # Analyze temporal patterns
        temporal_pattern = self._analyze_temporal_pattern(recent_data)
        
        # Identify root cause
        root_cause = self._identify_root_cause(
            primary_dimension, temporal_pattern, stat_diff
        )
        
        return {
            'statistical_differences': stat_diff,
            'primary_dimension': primary_dimension,
            'temporal_pattern': temporal_pattern,
            'root_cause': root_cause,
            'severity': self._assess_severity(stat_diff),
        }
    
    def _calculate_statistical_differences(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> Dict[str, float]:
        """Calculate statistical differences across dimensions."""
        differences = {}
        
        # Formality
        baseline_formality = statistics.mean(d.get('formality', 0) for d in baseline)
        recent_formality = statistics.mean(d.get('formality', 0) for d in recent)
        differences['formality'] = recent_formality - baseline_formality
        
        # Complexity
        baseline_complexity = statistics.mean(d.get('complexity', 0) for d in baseline)
        recent_complexity = statistics.mean(d.get('complexity', 0) for d in recent)
        differences['complexity'] = recent_complexity - baseline_complexity
        
        # Emoji density
        baseline_emoji = statistics.mean(d.get('emoji_density', 0) for d in baseline)
        recent_emoji = statistics.mean(d.get('emoji_density', 0) for d in recent)
        differences['emoji_density'] = recent_emoji - baseline_emoji
        
        return differences
    
    def _identify_primary_dimension(self, differences: Dict[str, float]) -> str:
        """Identify which dimension has the largest drift."""
        max_dimension = 'formality'
        max_diff = 0.0
        
        for dimension, diff in differences.items():
            if abs(diff) > abs(max_diff):
                max_diff = diff
                max_dimension = dimension
        
        return max_dimension
    
    def _analyze_temporal_pattern(self, data: List[Dict]) -> str:
        """Analyze temporal drift pattern."""
        if len(data) < 3:
            return "insufficient_data"
        
        # Sort by timestamp
        sorted_data = sorted(data, key=lambda x: x.get('timestamp', datetime.min))
        
        # Extract formality values
        formality_values = [d.get('formality', 0) for d in sorted_data]
        
        # Check for monotonic trend
        increasing = all(
            formality_values[i] <= formality_values[i + 1]
            for i in range(len(formality_values) - 1)
        )
        decreasing = all(
            formality_values[i] >= formality_values[i + 1]
            for i in range(len(formality_values) - 1)
        )
        
        if increasing:
            return "monotonic_increase"
        elif decreasing:
            return "monotonic_decrease"
        
        # Check for sudden change
        mid = len(formality_values) // 2
        first_half_avg = statistics.mean(formality_values[:mid])
        second_half_avg = statistics.mean(formality_values[mid:])
        
        if abs(second_half_avg - first_half_avg) > 20:
            return "sudden_shift"
        
        return "fluctuating"
    
    def _identify_root_cause(
        self, primary_dimension: str, temporal_pattern: str, differences: Dict[str, float]
    ) -> str:
        """Identify likely root cause of drift."""
        # Formality changes
        if primary_dimension == 'formality':
            if temporal_pattern == "sudden_shift":
                return "Possible change in content creator or editorial policy"
            elif temporal_pattern in ["monotonic_increase", "monotonic_decrease"]:
                return "Gradual shift in target audience or messaging strategy"
            else:
                return "Inconsistent application of brand voice guidelines"
        
        # Complexity changes
        elif primary_dimension == 'complexity':
            if differences['complexity'] > 0:
                return "Content becoming more technical or sophisticated"
            else:
                return "Content becoming simpler and more accessible"
        
        # Emoji usage changes
        elif primary_dimension == 'emoji_density':
            if differences['emoji_density'] > 0:
                return "Increased casual tone or platform adaptation"
            else:
                return "Shift toward more professional communication"
        
        return "Multiple factors contributing to voice drift"
    
    def _assess_severity(self, differences: Dict[str, float]) -> str:
        """Assess overall severity of drift."""
        max_diff = max(abs(d) for d in differences.values())
        
        if max_diff > 30:
            return "critical"
        elif max_diff > 20:
            return "high"
        elif max_diff > 10:
            return "medium"
        else:
            return "low"