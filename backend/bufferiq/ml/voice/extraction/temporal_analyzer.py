"""
Temporal voice evolution tracking.

Analyzes how brand voice changes over time.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import statistics

from bufferiq.ml.voice.stylistic.style_detector import StyleDetector


class TemporalVoiceAnalyzer:
    """
    Analyze temporal evolution of brand voice.
    
    Tracks how voice characteristics change over time
    to detect trends and shifts.
    
    Example:
```python
        analyzer = TemporalVoiceAnalyzer()
        evolution = analyzer.analyze_evolution(posts, window_days=30)
        print(f"Trend: {evolution['trend']}")
```
    """
    
    def __init__(self):
        """Initialize temporal analyzer."""
        self.style_detector = StyleDetector()
    
    def analyze_evolution(
        self, posts: List[Dict], window_days: int = 30
    ) -> Dict[str, any]:
        """
        Analyze voice evolution over time.
        
        Args:
            posts: List of posts with timestamps
            window_days: Size of time window for analysis
        
        Returns:
            Evolution metrics
        
        Raises:
            ValueError: If posts is empty
        """
        if not posts:
            raise ValueError("Cannot analyze evolution with no posts")
        
        # Sort by date
        sorted_posts = sorted(posts, key=lambda p: p["created_at"])
        
        # Split into time windows
        windows = self._create_time_windows(sorted_posts, window_days)
        
        if len(windows) < 2:
            return {
                "trend": "insufficient_data",
                "windows": len(windows),
                "drift_score": 0.0,
            }
        
        # Analyze each window
        window_features = []
        for window_posts in windows:
            if window_posts:
                combined_text = " ".join(p["text"] for p in window_posts)
                try:
                    style = self.style_detector.detect(combined_text)
                    window_features.append(style.formality_score)
                except ValueError:
                    continue
        
        if len(window_features) < 2:
            return {
                "trend": "insufficient_data",
                "windows": len(window_features),
                "drift_score": 0.0,
            }
        
        # Calculate trend
        trend = self._calculate_trend(window_features)
        drift_score = abs(window_features[-1] - window_features[0])
        
        return {
            "trend": trend,
            "windows": len(window_features),
            "drift_score": drift_score,
            "initial_formality": window_features[0],
            "current_formality": window_features[-1],
            "formality_over_time": window_features,
        }
    
    def _create_time_windows(
        self, sorted_posts: List[Dict], window_days: int
    ) -> List[List[Dict]]:
        """
        Create time windows from posts.
        
        Args:
            sorted_posts: Posts sorted by date
            window_days: Window size in days
        
        Returns:
            List of post lists (one per window)
        """
        if not sorted_posts:
            return []
        
        windows: List[List[Dict]] = []
        current_window: List[Dict] = []
        window_start = sorted_posts[0]["created_at"]
        window_delta = timedelta(days=window_days)
        
        for post in sorted_posts:
            post_date = post["created_at"]
            
            # Check if post belongs in current window
            if post_date < window_start + window_delta:
                current_window.append(post)
            else:
                # Start new window
                if current_window:
                    windows.append(current_window)
                current_window = [post]
                window_start = post_date
        
        # Add final window
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend from values.
        
        Args:
            values: List of values over time
        
        Returns:
            Trend description
        """
        if len(values) < 2:
            return "stable"
        
        # Calculate linear regression slope (simplified)
        n = len(values)
        x = list(range(n))
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Classify trend
        if slope > 2:
            return "increasing"
        elif slope < -2:
            return "decreasing"
        else:
            return "stable"