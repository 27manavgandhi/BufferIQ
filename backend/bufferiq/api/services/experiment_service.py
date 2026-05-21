"""
Experiment service.

Business logic for experiment operations.
"""

from typing import Dict, List, Optional, Any

from sqlalchemy.orm import Session

from bufferiq.ml.experiments.intelligence.service import ExperimentIntelligenceService
from bufferiq.ml.experiments.design.designer import Variant, MetricType


class ExperimentService:
    """
    Experiment service layer.

    Wraps ExperimentIntelligenceService for API.
    """

    def __init__(self, db_session: Session):
        """
        Initialize service.

        Args:
            db_session: Database session
        """
        self.intelligence = ExperimentIntelligenceService(db_session)

    async def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Variant],
        platform: str,
        primary_metric: MetricType,
        baseline_rate: float,
        mde: float = 0.10,
        alpha: float = 0.05,
        power: float = 0.80,
        expected_daily_traffic: Optional[int] = None,
        enable_sequential_testing: bool = False,
        enable_early_stopping: bool = False,
    ):
        """Create experiment."""
        return await self.intelligence.create_experiment(
            name=name,
            description=description,
            variants=variants,
            platform=platform,
            primary_metric=primary_metric,
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            expected_daily_traffic=expected_daily_traffic,
            enable_sequential_testing=enable_sequential_testing,
            enable_early_stopping=enable_early_stopping,
        )

    async def assign_user(
        self,
        experiment_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        platform: Optional[str] = None,
    ):
        """Assign user to variant."""
        return await self.intelligence.assign_user(
            experiment_id=experiment_id,
            user_id=user_id,
            session_id=session_id,
            platform=platform,
        )

    async def track_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_type: MetricType,
        value: float,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Track metric."""
        return await self.intelligence.track_metric(
            experiment_id=experiment_id,
            user_id=user_id,
            metric_type=metric_type,
            value=value,
            session_id=session_id,
            metadata=metadata,
        )

    async def analyze_experiment(
        self,
        experiment_id: str,
        alpha: float = 0.05,
        min_sample_size: int = 100,
    ):
        """Analyze experiment."""
        return await self.intelligence.analyze_experiment(
            experiment_id=experiment_id,
            alpha=alpha,
            min_sample_size=min_sample_size,
        )

    def list_experiments(self):
        """List experiments."""
        return self.intelligence.list_experiments()