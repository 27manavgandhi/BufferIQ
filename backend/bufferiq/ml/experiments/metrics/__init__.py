"""
Metrics tracking module.

Tracks experiment metrics, funnels, and conversions.

Components:
    - MetricsTracker: Track metric events
    - FunnelAnalyzer: Conversion funnel analysis
    - MetricsAggregator: Aggregate metrics
    - MetricsValidator: Validate metrics

Example:
```python
    from bufferiq.ml.experiments.metrics import MetricsTracker
    
    tracker = MetricsTracker(db_session)
    
    tracker.track(
        experiment_id="exp_001",
        user_id="user123",
        metric_type=MetricType.ENGAGEMENT_RATE,
        value=1.0
    )
```
"""

from bufferiq.ml.experiments.metrics.tracker import MetricsTracker
from bufferiq.ml.experiments.metrics.funnel import FunnelAnalyzer
from bufferiq.ml.experiments.metrics.aggregator import MetricsAggregator
from bufferiq.ml.experiments.metrics.validator import MetricsValidator

__all__ = [
    "MetricsTracker",
    "FunnelAnalyzer",
    "MetricsAggregator",
    "MetricsValidator",
]