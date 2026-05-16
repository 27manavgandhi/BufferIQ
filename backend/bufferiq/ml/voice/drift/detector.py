"""
Statistical voice drift detection.

Uses statistical tests and trend analysis to identify
when brand voice deviates from established profile.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import statistics
import logging

from scipy import stats
import pandas as pd
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class DriftAlert:
    """Voice drift detection alert."""
    
    drift_detected: bool
    drift_score: float  # 0-100, higher = more drift
    drift_type: str  # "gradual", "sudden", "cyclical"
    affected_dimensions: List[str]
    
    # Statistical tests
    t_statistic: float
    p_value: float
    confidence: float
    
    # Root cause
    likely_causes: List[str]
    example_deviations: List[Dict]
    
    # Visualization data
    drift_timeline: Optional[pd.DataFrame]
    severity: str  # "low", "medium", "high", "critical"


class VoiceDriftDetector:
    """
    Detect statistical drift in brand voice over time.
    
    Uses statistical tests and trend analysis to identify
    when brand voice deviates from established profile.
    
    Example:
```python
        detector = VoiceDriftDetector(db_session)
        alert = await detector.detect(
            brand_id="brand123",
            platform="linkedin",
            window_days=30
        )
        if alert.drift_detected:
            print(f"Drift detected! Score: {alert.drift_score:.1f}")
            print(f"Type: {alert.drift_type}")
            print(f"Affected: {alert.affected_dimensions}")
```
    """
    
    def __init__(
        self,
        db_session: Session,
        drift_threshold: float = 0.15,  # 15% deviation
        significance_level: float = 0.05,
    ):
        """
        Initialize drift detector.
        
        Args:
            db_session: Database session
            drift_threshold: Threshold for drift detection
            significance_level: Statistical significance level
        """
        self.db = db_session
        self.threshold = drift_threshold
        self.alpha = significance_level
    
    async def detect(
        self,
        brand_id: str,
        platform: str,
        window_days: int = 30,
        baseline_days: int = 90,
    ) -> DriftAlert:
        """
        Detect voice drift in recent content.
        
        Args:
            brand_id: Brand identifier
            platform: Platform to analyze
            window_days: Recent window to check
            baseline_days: Baseline period for comparison
        
        Returns:
            Drift alert with analysis
        
        Raises:
            ValueError: If platform not supported or insufficient data
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        # Get baseline and recent data
        baseline_data = self._fetch_baseline_data(
            brand_id, platform, baseline_days
        )
        recent_data = self._fetch_recent_data(
            brand_id, platform, window_days
        )
        
        if len(baseline_data) < 20:
            raise ValueError(
                f"Insufficient baseline data. "
                f"Found {len(baseline_data)}, minimum required: 20"
            )
        
        if len(recent_data) < 10:
            raise ValueError(
                f"Insufficient recent data. "
                f"Found {len(recent_data)}, minimum required: 10"
            )
        
        logger.info(
            f"Detecting drift for {brand_id} on {platform}: "
            f"{len(baseline_data)} baseline, {len(recent_data)} recent posts"
        )
        
        # Perform statistical tests
        drift_detected, t_stat, p_value = self._perform_t_test(
            baseline_data, recent_data
        )
        
        # Calculate drift score
        drift_score = self._calculate_drift_score(baseline_data, recent_data)
        
        # Identify drift type
        drift_type = self._identify_drift_type(baseline_data, recent_data)
        
        # Identify affected dimensions
        affected = self._identify_affected_dimensions(baseline_data, recent_data)
        
        # Generate likely causes
        causes = self._identify_likely_causes(affected, drift_type)
        
        # Get example deviations
        examples = self._get_example_deviations(recent_data, baseline_data)
        
        # Create timeline
        timeline = self._create_drift_timeline(baseline_data, recent_data)
        
        # Determine severity
        severity = self._determine_severity(drift_score)
        
        # Calculate confidence
        confidence = 1.0 - p_value if drift_detected else 0.0
        
        logger.info(
            f"Drift detection complete: "
            f"detected={drift_detected}, score={drift_score:.2f}, "
            f"type={drift_type}, severity={severity}"
        )
        
        return DriftAlert(
            drift_detected=drift_detected,
            drift_score=drift_score,
            drift_type=drift_type,
            affected_dimensions=affected,
            t_statistic=t_stat,
            p_value=p_value,
            confidence=confidence,
            likely_causes=causes,
            example_deviations=examples,
            drift_timeline=timeline,
            severity=severity,
        )
    
    def _fetch_baseline_data(
        self, brand_id: str, platform: str, days: int
    ) -> List[Dict]:
        """Fetch baseline historical data."""
        # Mock implementation
        # In production, would query database
        end_date = datetime.utcnow() - timedelta(days=30)
        start_date = end_date - timedelta(days=days)
        
        return [
            {
                'formality': 65.0 + (i % 10),
                'complexity': 50.0 + (i % 8),
                'emoji_density': 2.0,
                'timestamp': start_date + timedelta(days=i),
            }
            for i in range(min(days, 90))
        ]
    
    def _fetch_recent_data(
        self, brand_id: str, platform: str, days: int
    ) -> List[Dict]:
        """Fetch recent data."""
        # Mock implementation
        start_date = datetime.utcnow() - timedelta(days=days)
        
        return [
            {
                'formality': 75.0 + (i % 10),  # Higher formality = drift
                'complexity': 50.0 + (i % 8),
                'emoji_density': 1.0,  # Lower emoji = drift
                'timestamp': start_date + timedelta(days=i),
            }
            for i in range(min(days, 30))
        ]
    
    def _perform_t_test(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> tuple:
        """
        Perform t-test to detect significant differences.
        
        Returns:
            Tuple of (drift_detected, t_statistic, p_value)
        """
        # Extract formality scores
        baseline_formality = [d['formality'] for d in baseline]
        recent_formality = [d['formality'] for d in recent]
        
        # Perform independent t-test
        t_stat, p_value = stats.ttest_ind(baseline_formality, recent_formality)
        
        # Drift detected if p-value < alpha
        drift_detected = p_value < self.alpha
        
        return drift_detected, float(t_stat), float(p_value)
    
    def _calculate_drift_score(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> float:
        """
        Calculate overall drift score (0-100).
        
        Higher score = more drift.
        """
        # Compare means across dimensions
        baseline_formality = statistics.mean(d['formality'] for d in baseline)
        recent_formality = statistics.mean(d['formality'] for d in recent)
        
        baseline_complexity = statistics.mean(d['complexity'] for d in baseline)
        recent_complexity = statistics.mean(d['complexity'] for d in recent)
        
        baseline_emoji = statistics.mean(d['emoji_density'] for d in baseline)
        recent_emoji = statistics.mean(d['emoji_density'] for d in recent)
        
        # Calculate normalized differences
        formality_drift = abs(recent_formality - baseline_formality) / 100
        complexity_drift = abs(recent_complexity - baseline_complexity) / 100
        emoji_drift = abs(recent_emoji - baseline_emoji) / 10  # Scale emoji
        
        # Weighted average
        drift_score = (
            formality_drift * 40 +
            complexity_drift * 30 +
            emoji_drift * 30
        ) * 100
        
        return min(drift_score, 100.0)
    
    def _identify_drift_type(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> str:
        """
        Identify type of drift (gradual, sudden, cyclical).
        
        Returns:
            Drift type string
        """
        # Combine data
        all_data = baseline + recent
        all_data.sort(key=lambda x: x['timestamp'])
        
        # Extract formality over time
        formality_values = [d['formality'] for d in all_data]
        
        if len(formality_values) < 4:
            return "unknown"
        
        # Check for sudden change
        mid_point = len(formality_values) // 2
        first_half = formality_values[:mid_point]
        second_half = formality_values[mid_point:]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        # If big jump between halves = sudden
        if abs(avg_second - avg_first) > 15:
            return "sudden"
        
        # Check for gradual trend
        # Simple linear regression slope
        n = len(formality_values)
        x = list(range(n))
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(formality_values)
        
        numerator = sum((x[i] - x_mean) * (formality_values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if abs(slope) > 0.5:
            return "gradual"
        
        return "stable"
    
    def _identify_affected_dimensions(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> List[str]:
        """Identify which voice dimensions are affected."""
        affected = []
        
        # Check formality
        baseline_formality = statistics.mean(d['formality'] for d in baseline)
        recent_formality = statistics.mean(d['formality'] for d in recent)
        if abs(recent_formality - baseline_formality) > 10:
            affected.append('formality')
        
        # Check complexity
        baseline_complexity = statistics.mean(d['complexity'] for d in baseline)
        recent_complexity = statistics.mean(d['complexity'] for d in recent)
        if abs(recent_complexity - baseline_complexity) > 10:
            affected.append('complexity')
        
        # Check emoji usage
        baseline_emoji = statistics.mean(d['emoji_density'] for d in baseline)
        recent_emoji = statistics.mean(d['emoji_density'] for d in recent)
        if abs(recent_emoji - baseline_emoji) > 1.0:
            affected.append('emoji_usage')
        
        return affected
    
    def _identify_likely_causes(
        self, affected: List[str], drift_type: str
    ) -> List[str]:
        """Identify likely causes of drift."""
        causes = []
        
        if 'formality' in affected:
            if drift_type == 'sudden':
                causes.append("New content creator or change in editorial guidelines")
            else:
                causes.append("Gradual shift in target audience or messaging strategy")
        
        if 'emoji_usage' in affected:
            causes.append("Change in platform norms or audience preferences")
        
        if 'complexity' in affected:
            causes.append("Shift in content topics or target expertise level")
        
        if not causes:
            causes.append("Minor natural variation in content style")
        
        return causes
    
    def _get_example_deviations(
        self, recent: List[Dict], baseline: List[Dict]
    ) -> List[Dict]:
        """Get examples of significant deviations."""
        baseline_formality = statistics.mean(d['formality'] for d in baseline)
        
        examples = []
        for data in recent:
            deviation = abs(data['formality'] - baseline_formality)
            if deviation > 15:  # Significant deviation
                examples.append({
                    'timestamp': data['timestamp'].isoformat(),
                    'formality': data['formality'],
                    'baseline_formality': baseline_formality,
                    'deviation': deviation,
                })
        
        # Return top 5 deviations
        examples.sort(key=lambda x: x['deviation'], reverse=True)
        return examples[:5]
    
    def _create_drift_timeline(
        self, baseline: List[Dict], recent: List[Dict]
    ) -> pd.DataFrame:
        """Create timeline DataFrame for visualization."""
        all_data = baseline + recent
        all_data.sort(key=lambda x: x['timestamp'])
        
        return pd.DataFrame([
            {
                'timestamp': d['timestamp'],
                'formality': d['formality'],
                'complexity': d['complexity'],
                'emoji_density': d['emoji_density'],
                'period': 'baseline' if d in baseline else 'recent',
            }
            for d in all_data
        ])
    
    def _determine_severity(self, drift_score: float) -> str:
        """Determine severity level from drift score."""
        if drift_score < 15:
            return "low"
        elif drift_score < 30:
            return "medium"
        elif drift_score < 50:
            return "high"
        else:
            return "critical"