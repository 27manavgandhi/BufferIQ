"""
Metrics tracker.

Tracks metric events for experiments.

Key features:
    - Event tracking
    - Metric validation
    - Timestamp recording
    - User-variant association
    - Batch tracking

Example:
```python
    tracker = MetricsTracker(db_session)
    
    tracker.track(
        experiment_id="exp_001",
        user_id="user123",
        metric_type=MetricType.ENGAGEMENT_RATE,
        value=1.0,
        variant_id="treatment"
    )
```
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from bufferiq.ml.experiments.design.designer import MetricType


@dataclass
class MetricEvent:
    """Metric event data."""

    experiment_id: str
    user_id: str
    variant_id: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None


class MetricsTracker:
    """
    Track experiment metrics.

    Example:
```python
        tracker = MetricsTracker(db_session)

        # Track engagement
        tracker.track(
            experiment_id="exp_001",
            user_id="user123",
            variant_id="treatment",
            metric_type=MetricType.ENGAGEMENT_RATE,
            value=1.0
        )

        # Get metrics
        metrics = tracker.get_metrics(
            experiment_id="exp_001",
            variant_id="treatment"
        )
```
    """

    def __init__(self, db_session: Session) -> None:
        """
        Initialize metrics tracker.

        Args:
            db_session: Database session
        """
        self.db = db_session
        self._events: List[MetricEvent] = []

    def track(
        self,
        experiment_id: str,
        user_id: str,
        variant_id: str,
        metric_type: MetricType,
        value: float,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Track metric event.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            variant_id: Variant ID
            metric_type: Metric type
            value: Metric value
            session_id: Optional session ID
            metadata: Optional metadata
        """
        event = MetricEvent(
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata=metadata,
        )

        self._events.append(event)

        # In production, save to database here

    def track_batch(self, events: List[MetricEvent]) -> None:
        """
        Track multiple events.

        Args:
            events: List of events
        """
        for event in events:
            self._events.append(event)

        # In production, bulk insert to database

    def get_metrics(
        self,
        experiment_id: str,
        variant_id: Optional[str] = None,
        metric_type: Optional[MetricType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MetricEvent]:
        """
        Get tracked metrics.

        Args:
            experiment_id: Experiment ID
            variant_id: Optional variant filter
            metric_type: Optional metric type filter
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            List of metric events
        """
        filtered = [e for e in self._events if e.experiment_id == experiment_id]

        if variant_id:
            filtered = [e for e in filtered if e.variant_id == variant_id]

        if metric_type:
            filtered = [e for e in filtered if e.metric_type == metric_type]

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]

        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]

        return filtered

    def get_metric_values(
        self,
        experiment_id: str,
        variant_id: str,
        metric_type: MetricType,
    ) -> List[float]:
        """
        Get metric values for variant.

        Args:
            experiment_id: Experiment ID
            variant_id: Variant ID
            metric_type: Metric type

        Returns:
            List of metric values
        """
        events = self.get_metrics(experiment_id, variant_id, metric_type)
        return [e.value for e in events]

    def get_user_metrics(
        self, experiment_id: str, user_id: str
    ) -> List[MetricEvent]:
        """
        Get all metrics for user.

        Args:
            experiment_id: Experiment ID
            user_id: User ID

        Returns:
            List of user's metric events
        """
        return [
            e
            for e in self._events
            if e.experiment_id == experiment_id and e.user_id == user_id
        ]