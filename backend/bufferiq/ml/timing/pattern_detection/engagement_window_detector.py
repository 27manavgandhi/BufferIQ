"""Detect high-engagement time windows."""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy import stats


@dataclass
class EngagementWindow:
    """High-engagement time window."""

    hour: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)
    avg_engagement: float
    std_engagement: float
    sample_size: int
    confidence: float
    percentile_rank: float


class EngagementWindowDetector:
    """Detect high-engagement time windows."""

    def __init__(self, percentile_threshold: float = 0.75) -> None:
        """
        Initialize detector.

        Args:
            percentile_threshold: Threshold for "high" engagement (0-1)
        """
        if not 0 < percentile_threshold < 1:
            raise ValueError("percentile_threshold must be between 0 and 1")

        self.percentile_threshold = percentile_threshold

    def detect_windows(self, ts: pd.DataFrame) -> List[EngagementWindow]:
        """
        Detect high-engagement windows.

        Args:
            ts: Time-series with timestamp and engagement_score

        Returns:
            List of EngagementWindow objects sorted by engagement

        Example:
            >>> detector = EngagementWindowDetector(percentile_threshold=0.75)
            >>> windows = detector.detect_windows(ts)
            >>> assert all(0 <= w.hour < 24 for w in windows)
        """
        if ts.empty or "timestamp" not in ts.columns:
            return []

        if "engagement_score" not in ts.columns:
            return []

        # Add temporal features
        df = ts.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek

        # Group by (hour, day_of_week)
        grouped = (
            df.groupby(["hour", "day_of_week"])["engagement_score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        # Calculate percentile threshold
        overall_mean = df["engagement_score"].mean()
        threshold = df["engagement_score"].quantile(self.percentile_threshold)

        # Filter high-engagement windows
        high_windows = grouped[grouped["mean"] >= threshold].copy()

        if high_windows.empty:
            return []

        # Calculate confidence and percentile rank
        windows = []
        for _, row in high_windows.iterrows():
            # Confidence based on sample size and variance
            confidence = self._calculate_confidence(
                float(row["mean"]),
                float(row["std"]) if pd.notna(row["std"]) else 0.0,
                int(row["count"]),
                overall_mean,
            )

            # Percentile rank
            percentile_rank = (
                stats.percentileofscore(grouped["mean"], row["mean"]) / 100.0
            )

            windows.append(
                EngagementWindow(
                    hour=int(row["hour"]),
                    day_of_week=int(row["day_of_week"]),
                    avg_engagement=float(row["mean"]),
                    std_engagement=float(row["std"]) if pd.notna(row["std"]) else 0.0,
                    sample_size=int(row["count"]),
                    confidence=confidence,
                    percentile_rank=percentile_rank,
                )
            )

        # Sort by engagement
        windows.sort(key=lambda w: w.avg_engagement, reverse=True)

        return windows

    def rank_windows(self, windows: List[EngagementWindow]) -> List[EngagementWindow]:
        """
        Rank windows by composite score.

        Score = engagement × confidence × percentile_rank

        Args:
            windows: List of EngagementWindow objects

        Returns:
            Ranked list of windows
        """
        scored_windows = [
            (w, w.avg_engagement * w.confidence * w.percentile_rank) for w in windows
        ]

        # Sort by score descending
        scored_windows.sort(key=lambda x: x[1], reverse=True)

        return [w for w, _ in scored_windows]

    def _calculate_confidence(
        self,
        mean: float,
        std: float,
        n: int,
        baseline: float,
    ) -> float:
        """
        Calculate confidence score (0-1).

        Factors:
        - Sample size (higher is better)
        - Stability (lower std is better)
        - Magnitude (higher mean is better)
        """
        # Sample size factor (0-1)
        size_factor = min(n / 20.0, 1.0)  # Cap at 20 samples

        # Stability factor (0-1)
        if mean > 0:
            cv = std / mean  # Coefficient of variation
            stability_factor = max(0.0, 1.0 - cv)
        else:
            stability_factor = 0.5

        # Magnitude factor (0-1)
        if baseline > 0:
            magnitude_factor = min(mean / (baseline * 2), 1.0)
        else:
            magnitude_factor = 0.5

        # Weighted combination
        confidence = (
            0.4 * size_factor + 0.3 * stability_factor + 0.3 * magnitude_factor
        )

        return float(np.clip(confidence, 0.0, 1.0))