"""
Metrics aggregator.

Aggregates metrics across time periods and variants.

Example:
```python
    aggregator = MetricsAggregator()
    
    daily_metrics = aggregator.aggregate_by_day(
        events=events,
        metric_type=MetricType.ENGAGEMENT_RATE
    )
```
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np

from bufferiq.ml.experiments.design.designer import MetricType
from bufferiq.ml.experiments.metrics.tracker import MetricEvent


class MetricsAggregator:
    """
    Aggregate experiment metrics.

    Example:
```python
        aggregator = MetricsAggregator()

        # Aggregate by variant
        variant_stats = aggregator.aggregate_by_variant(
            events=events,
            metric_type=MetricType.ENGAGEMENT_RATE
        )

        for variant_id, stats in variant_stats.items():
            print(f"{variant_id}: {stats['mean']:.3f}")
```
    """

    def aggregate_by_variant(
        self, events: List[MetricEvent], metric_type: MetricType
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics by variant.

        Args:
            events: Metric events
            metric_type: Metric type to aggregate

        Returns:
            Dictionary mapping variant_id to stats
        """
        # Filter events
        filtered = [e for e in events if e.metric_type == metric_type]

        # Group by variant
        variant_values: Dict[str, List[float]] = defaultdict(list)
        for event in filtered:
            variant_values[event.variant_id].append(event.value)

        # Calculate statistics
        result = {}
        for variant_id, values in variant_values.items():
            if values:
                result[variant_id] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
            else:
                result[variant_id] = {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                }

        return result

    def aggregate_by_day(
        self, events: List[MetricEvent], metric_type: MetricType
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics by day.

        Args:
            events: Metric events
            metric_type: Metric type

        Returns:
            Dictionary mapping date to stats
        """
        # Filter events
        filtered = [e for e in events if e.metric_type == metric_type]

        # Group by day
        daily_values: Dict[str, List[float]] = defaultdict(list)
        for event in filtered:
            day = event.timestamp.strftime("%Y-%m-%d")
            daily_values[day].append(event.value)

        # Calculate statistics
        result = {}
        for day, values in daily_values.items():
            if values:
                result[day] = {
                    "mean": float(np.mean(values)),
                    "count": len(values),
                }
            else:
                result[day] = {"mean": 0.0, "count": 0}

        return result

    def aggregate_by_variant_and_day(
        self, events: List[MetricEvent], metric_type: MetricType
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Aggregate by variant and day.

        Args:
            events: Metric events
            metric_type: Metric type

        Returns:
            Nested dictionary: variant_id -> day -> stats
        """
        # Filter events
        filtered = [e for e in events if e.metric_type == metric_type]

        # Group by variant and day
        data: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for event in filtered:
            day = event.timestamp.strftime("%Y-%m-%d")
            data[event.variant_id][day].append(event.value)

        # Calculate statistics
        result = {}
        for variant_id, daily_data in data.items():
            result[variant_id] = {}
            for day, values in daily_data.items():
                if values:
                    result[variant_id][day] = {
                        "mean": float(np.mean(values)),
                        "count": len(values),
                    }
                else:
                    result[variant_id][day] = {"mean": 0.0, "count": 0}

        return result

    def calculate_cumulative_metrics(
        self, events: List[MetricEvent], metric_type: MetricType, variant_id: str
    ) -> List[Dict[str, float]]:
        """
        Calculate cumulative metrics over time.

        Args:
            events: Metric events
            metric_type: Metric type
            variant_id: Variant ID

        Returns:
            List of cumulative stats by day
        """
        # Filter events
        filtered = [
            e
            for e in events
            if e.metric_type == metric_type and e.variant_id == variant_id
        ]

        # Sort by timestamp
        filtered.sort(key=lambda e: e.timestamp)

        # Calculate cumulative stats
        cumulative = []
        values_so_far = []

        for event in filtered:
            values_so_far.append(event.value)
            cumulative.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "cumulative_mean": float(np.mean(values_so_far)),
                    "cumulative_count": len(values_so_far),
                }
            )

        return cumulative