"""
Temporal consistency tracking.

Tracks voice consistency over time to identify trends.
"""

from typing import List, Dict
from datetime import datetime, timedelta
import statistics


class ConsistencyTracker:
    """
    Track voice consistency over time.
    
    Maintains history of consistency scores and identifies trends.
    
    Example:
```python
        tracker = ConsistencyTracker()
        tracker.add_score(score, datetime.utcnow())
        trend = tracker.get_trend(days=30)
```
    """
    
    def __init__(self):
        """Initialize consistency tracker."""
        self.history: List[Dict] = []
    
    def add_score(self, score: float, timestamp: datetime) -> None:
        """
        Add consistency score to history.
        
        Args:
            score: Consistency score (0-100)
            timestamp: When score was recorded
        """
        self.history.append({
            'score': score,
            'timestamp': timestamp,
        })
    
    def get_trend(self, days: int = 30) -> Dict[str, any]:
        """
        Get consistency trend for recent period.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Trend analysis dictionary
        """
        if not self.history:
            return {
                'trend': 'insufficient_data',
                'average_score': 0.0,
                'sample_size': 0,
            }
        
        # Filter to recent period
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [
            h for h in self.history
            if h['timestamp'] >= cutoff
        ]
        
        if not recent:
            return {
                'trend': 'insufficient_data',
                'average_score': 0.0,
                'sample_size': 0,
            }
        
        # Calculate statistics
        scores = [h['score'] for h in recent]
        avg_score = statistics.mean(scores)
        
        # Determine trend
        if len(scores) >= 2:
            first_half = scores[:len(scores)//2]
            second_half = scores[len(scores)//2:]
            
            avg_first = statistics.mean(first_half)
            avg_second = statistics.mean(second_half)
            
            if avg_second > avg_first + 5:
                trend = 'improving'
            elif avg_second < avg_first - 5:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'average_score': avg_score,
            'min_score': min(scores),
            'max_score': max(scores),
            'sample_size': len(scores),
        }
    
    def get_all_scores(self) -> List[Dict]:
        """
        Get complete history of scores.
        
        Returns:
            List of score dictionaries
        """
        return self.history.copy()