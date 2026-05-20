"""
Metrics validator.

Validates metric data quality and detects anomalies.

Example:
```python
    validator = MetricsValidator()
    
    is_valid = validator.validate(
        events=events,
        metric_type=MetricType.ENGAGEMENT_RATE
    )
```
"""

from typing import Dict, List, Optional

import numpy as np

from bufferiq.ml.experiments.design.designer import MetricType
from bufferiq.ml.experiments.metrics.tracker import MetricEvent


class MetricsValidator:
    """
    Validate metric data.

    Example:
```python
        validator = MetricsValidator()

        # Validate data quality
        result = validator.validate_metrics(
            events=events,
            metric_type=MetricType.ENGAGEMENT_RATE
        )

        if not result['is_valid']:
            print(f"Issues: {result['issues']}")
```
    """

    def validate_metrics(
        self, events: List[MetricEvent], metric_type: MetricType
    ) -> Dict[str, any]:
        """
        Validate metric events.

        Args:
            events: Metric events
            metric_type: Metric type

        Returns:
            Validation result
        """
        issues = []

        # Filter events
        filtered = [e for e in events if e.metric_type == metric_type]

        if not filtered:
            return {"is_valid": False, "issues": ["No events found"]}

        # Check for missing values
        missing_count = sum(1 for e in filtered if e.value is None)
        if missing_count > 0:
            issues.append(f"{missing_count} events with missing values")

        # Check for outliers
        values = [e.value for e in filtered if e.value is not None]
        if values:
            outlier_count = self._count_outliers(values)
            if outlier_count > len(values) * 0.05:  # > 5% outliers
                issues.append(f"{outlier_count} outliers detected")

        # Check for duplicate timestamps
        timestamps = [e.timestamp for e in filtered]
        if len(timestamps) != len(set(timestamps)):
            issues.append("Duplicate timestamps detected")

        return {"is_valid": len(issues) == 0, "issues": issues, "event_count": len(filtered)}

    def _count_outliers(
        self, values: List[float], z_threshold: float = 3.0
    ) -> int:
        """
        Count outliers using z-score.

        Args:
            values: Values to check
            z_threshold: Z-score threshold

        Returns:
            Number of outliers
        """
        if len(values) < 3:
            return 0

        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)

        if std == 0:
            return 0

        z_scores = np.abs((values_array - mean) / std)
        return int(np.sum(z_scores > z_threshold))

    def check_sample_ratio_mismatch(
        self, variant_counts: Dict[str, int], expected_ratios: Dict[str, float], tolerance: float = 0.05
    ) -> Dict[str, any]:
        """
        Check for sample ratio mismatch.

        Args:
            variant_counts: Actual counts per variant
            expected_ratios: Expected ratios per variant
            tolerance: Allowed deviation

        Returns:
            SRM check result
        """
        total = sum(variant_counts.values())
        issues = []

        for variant_id, expected_ratio in expected_ratios.items():
            actual_count = variant_counts.get(variant_id, 0)
            actual_ratio = actual_count / total if total > 0 else 0.0

            deviation = abs(actual_ratio - expected_ratio)

            if deviation > tolerance:
                issues.append(
                    f"{variant_id}: expected {expected_ratio:.1%}, "
                    f"got {actual_ratio:.1%} (deviation: {deviation:.1%})"
                )

        return {"has_srm": len(issues) > 0, "issues": issues}