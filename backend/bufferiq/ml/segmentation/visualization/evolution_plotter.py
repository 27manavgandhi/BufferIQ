"""Plot segment evolution over time."""

from typing import Any, Dict, List

from bufferiq.ml.segmentation.types import SegmentSnapshot


class EvolutionPlotter:
    """Plot segment evolution."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize evolution plotter."""
        self.config = config or {}

    def plot_size_evolution(
        self, snapshots: List[SegmentSnapshot]
    ) -> Dict[str, Any]:
        """
        Plot segment size over time.

        Args:
            snapshots: List of snapshots

        Returns:
            Size evolution data
        """
        return {
            "timestamps": [s.timestamp.isoformat() for s in snapshots],
            "sizes": [s.size for s in snapshots],
            "engagement_rates": [float(s.avg_engagement_rate) for s in snapshots],
        }

    def plot_engagement_evolution(
        self, snapshots: List[SegmentSnapshot]
    ) -> Dict[str, Any]:
        """
        Plot engagement evolution.

        Args:
            snapshots: List of snapshots

        Returns:
            Engagement evolution data
        """
        return {
            "timestamps": [s.timestamp.isoformat() for s in snapshots],
            "engagement_rates": [float(s.avg_engagement_rate) for s in snapshots],
            "health_scores": [float(s.health_score) for s in snapshots],
        }

    def plot_comprehensive_evolution(
        self, snapshots: List[SegmentSnapshot]
    ) -> Dict[str, Any]:
        """
        Plot comprehensive segment evolution.

        Args:
            snapshots: List of snapshots

        Returns:
            Comprehensive evolution data
        """
        return {
            "segment_id": snapshots[0].segment_id if snapshots else "unknown",
            "platform": snapshots[0].platform if snapshots else "unknown",
            "timeline": [s.timestamp.isoformat() for s in snapshots],
            "metrics": {
                "size": [s.size for s in snapshots],
                "engagement": [float(s.avg_engagement_rate) for s in snapshots],
                "health": [float(s.health_score) for s in snapshots],
            },
        }