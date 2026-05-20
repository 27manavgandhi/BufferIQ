"""
Experiment monitor.

Monitors experiment health and performance.

Example:
```python
    monitor = ExperimentMonitor()
    
    health = monitor.check_health(
        experiment_id="exp_001",
        metrics=metrics
    )
```
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from bufferiq.ml.experiments.monitoring.anomaly_detector import AnomalyDetector
from bufferiq.ml.experiments.monitoring.srm_detector import SRMDetector


@dataclass
class HealthCheck:
    """Experiment health check result."""

    is_healthy: bool
    issues: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class ExperimentMonitor:
    """
    Monitor experiment health.

    Example:
```python
        monitor = ExperimentMonitor()

        health = monitor.check_health(
            variant_counts={"control": 1000, "treatment": 950},
            expected_ratios={"control": 0.5, "treatment": 0.5},
            daily_metrics=daily_data
        )

        if not health.is_healthy:
            print(f"Issues: {health.issues}")
```
    """

    def __init__(self) -> None:
        """Initialize monitor."""
        self.anomaly_detector = AnomalyDetector()
        self.srm_detector = SRMDetector()

    def check_health(
        self,
        variant_counts: Dict[str, int],
        expected_ratios: Dict[str, float],
        daily_metrics: Optional[Dict[str, List[float]]] = None,
    ) -> HealthCheck:
        """
        Check experiment health.

        Args:
            variant_counts: Actual counts per variant
            expected_ratios: Expected traffic ratios
            daily_metrics: Optional daily metrics by variant

        Returns:
            Health check result
        """
        issues = []
        warnings = []
        metrics = {}

        # Check sample ratio mismatch
        srm_result = self.srm_detector.detect_srm(
            variant_counts, expected_ratios
        )

        if srm_result["has_srm"]:
            issues.append(f"Sample ratio mismatch detected: {srm_result['chi2_p_value']:.4f}")

        metrics["srm_p_value"] = srm_result["chi2_p_value"]

        # Check for anomalies in daily metrics
        if daily_metrics:
            for variant, values in daily_metrics.items():
                anomalies = self.anomaly_detector.detect_anomalies(values)

                if anomalies["num_anomalies"] > 0:
                    warnings.append(
                        f"{variant}: {anomalies['num_anomalies']} anomalies detected"
                    )

                metrics[f"{variant}_anomalies"] = anomalies["num_anomalies"]

        # Overall health
        is_healthy = len(issues) == 0

        return HealthCheck(
            is_healthy=is_healthy,
            issues=issues,
            warnings=warnings,
            metrics=metrics,
        )

    def generate_alerts(self, health: HealthCheck) -> List[Dict[str, str]]:
        """
        Generate alerts from health check.

        Args:
            health: Health check result

        Returns:
            List of alerts
        """
        alerts = []

        for issue in health.issues:
            alerts.append({"severity": "critical", "message": issue})

        for warning in health.warnings:
            alerts.append({"severity": "warning", "message": warning})

        return alerts