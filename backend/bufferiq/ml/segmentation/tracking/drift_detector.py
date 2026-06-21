"""Detect segment drift and changes."""

from typing import Any, Dict, List

import numpy as np

from bufferiq.ml.segmentation.types import SegmentSnapshot


class DriftDetector:
    """Detect significant changes or drift in segment characteristics."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize drift detector."""
        self.config = config or {}
        self.drift_threshold = self.config.get("drift_threshold", 0.2)

    def detect_drift(
        self, snapshots: List[SegmentSnapshot]
    ) -> Dict[str, Any]:
        """
        Detect drift in segment characteristics.

        Args:
            snapshots: List of segment snapshots

        Returns:
            Drift analysis
        """
        if len(snapshots) < 2:
            return {
                "has_drift": False,
                "drift_metrics": {},
                "severity": "none",
            }

        # Check centroid drift
        centroid_drift = self._check_centroid_drift(snapshots)

        # Check engagement drift
        engagement_drift = self._check_engagement_drift(snapshots)

        # Check size drift
        size_drift = self._check_size_drift(snapshots)

        # Determine overall severity
        severity = self._determine_severity(
            centroid_drift, engagement_drift, size_drift
        )

        return {
            "has_drift": severity != "none",
            "drift_metrics": {
                "centroid_drift": centroid_drift,
                "engagement_drift": engagement_drift,
                "size_drift": size_drift,
            },
            "severity": severity,
        }

    def _check_centroid_drift(self, snapshots: List[SegmentSnapshot]) -> float:
        """Check for centroid drift."""
        centroids = [s.centroid for s in snapshots if s.centroid is not None]

        if len(centroids) < 2:
            return 0.0

        # Calculate average distance between consecutive centroids
        distances = []
        for i in range(len(centroids) - 1):
            distance = float(np.linalg.norm(centroids[i + 1] - centroids[i]))
            distances.append(distance)

        return float(np.mean(distances)) if distances else 0.0

    def _check_engagement_drift(self, snapshots: List[SegmentSnapshot]) -> float:
        """Check for engagement rate drift."""
        if len(snapshots) < 2:
            return 0.0

        engagement_rates = [s.avg_engagement_rate for s in snapshots]

        # Calculate relative change
        start_engagement = engagement_rates[0]
        end_engagement = engagement_rates[-1]

        if start_engagement == 0:
            return 0.0

        return abs(end_engagement - start_engagement) / start_engagement

    def _check_size_drift(self, snapshots: List[SegmentSnapshot]) -> float:
        """Check for size drift."""
        if len(snapshots) < 2:
            return 0.0

        sizes = [s.size for s in snapshots]

        # Calculate relative change
        start_size = sizes[0]
        end_size = sizes[-1]

        if start_size == 0:
            return 0.0

        return abs(end_size - start_size) / start_size

    def _determine_severity(
        self, centroid_drift: float, engagement_drift: float, size_drift: float
    ) -> str:
        """Determine drift severity."""
        max_drift = max(centroid_drift, engagement_drift, size_drift)

        if max_drift > self.drift_threshold * 2:
            return "high"
        elif max_drift > self.drift_threshold:
            return "medium"
        elif max_drift > 0.0:
            return "low"
        else:
            return "none"