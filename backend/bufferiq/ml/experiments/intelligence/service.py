"""
Experiment intelligence service.

Main orchestrator for complete experiment lifecycle.

Example:
```python
    service = ExperimentIntelligenceService(db_session)
    
    # Create experiment
    experiment = await service.create_experiment(
        name="Headline Test",
        variants=[control, treatment],
        platform="linkedin"
    )
    
    # Assign user
    assignment = await service.assign_user(
        experiment_id=experiment.experiment_id,
        user_id="user123"
    )
    
    # Track metric
    await service.track_metric(
        experiment_id=experiment.experiment_id,
        user_id="user123",
        metric_type=MetricType.ENGAGEMENT_RATE,
        value=1.0
    )
    
    # Analyze
    results = await service.analyze_experiment(
        experiment_id=experiment.experiment_id
    )
```
"""

from typing import Dict, List, Optional, Any

import numpy as np
from sqlalchemy.orm import Session

from bufferiq.ml.experiments.design.designer import (
    ExperimentDesigner,
    ExperimentConfig,
    Variant,
    MetricType,
    SUPPORTED_PLATFORMS,
)
from bufferiq.ml.experiments.assignment.engine import AssignmentEngine, Assignment
from bufferiq.ml.experiments.statistics.hypothesis_tester import StatisticalAnalyzer
from bufferiq.ml.experiments.power.calculator import PowerAnalyzer
from bufferiq.ml.experiments.metrics.tracker import MetricsTracker
from bufferiq.ml.experiments.sequential.sprt import SequentialTester
from bufferiq.ml.experiments.novelty.detector import NoveltyDetector
from bufferiq.ml.experiments.bandits.thompson_sampling import ThompsonSampling
from bufferiq.ml.experiments.interference.detector import InterferenceDetector
from bufferiq.ml.experiments.results.analyzer import ResultAnalyzer
from bufferiq.ml.experiments.monitoring.monitor import ExperimentMonitor
from bufferiq.ml.experiments.reporting.generator import ReportGenerator


class ExperimentIntelligenceService:
    """
    Main orchestrator for experimentation framework.

    Manages complete experiment lifecycle:
    - Design and creation
    - Variant assignment
    - Metrics tracking
    - Statistical analysis
    - Result reporting

    Example:
```python
        service = ExperimentIntelligenceService(
            db_session=session
        )

        # Create experiment
        experiment = await service.create_experiment(
            name="Headline Test",
            variants=[control, treatment],
            platform="linkedin",
            baseline_rate=0.05,
            mde=0.10
        )

        # Assign user to variant
        assignment = await service.assign_user(
            experiment_id=experiment.experiment_id,
            user_id="user123"
        )

        # Track metric
        await service.track_metric(
            experiment_id=experiment.experiment_id,
            user_id="user123",
            metric_type=MetricType.ENGAGEMENT_RATE,
            value=1.0
        )

        # Analyze results
        results = await service.analyze_experiment(
            experiment_id=experiment.experiment_id
        )

        if results['should_launch']:
            print(f"Winner: {results['winner_variant']}")
```
    """

    def __init__(
        self,
        db_session: Session,
        designer: Optional[ExperimentDesigner] = None,
        assignment_engine: Optional[AssignmentEngine] = None,
        statistical_analyzer: Optional[StatisticalAnalyzer] = None,
        power_analyzer: Optional[PowerAnalyzer] = None,
        metrics_tracker: Optional[MetricsTracker] = None,
        sequential_tester: Optional[SequentialTester] = None,
        novelty_detector: Optional[NoveltyDetector] = None,
        thompson_sampling: Optional[ThompsonSampling] = None,
        interference_detector: Optional[InterferenceDetector] = None,
        result_analyzer: Optional[ResultAnalyzer] = None,
        experiment_monitor: Optional[ExperimentMonitor] = None,
        report_generator: Optional[ReportGenerator] = None,
    ):
        """
        Initialize experiment intelligence service.

        Args:
            db_session: Database session
            designer: Experiment designer
            assignment_engine: Assignment engine
            statistical_analyzer: Statistical analyzer
            power_analyzer: Power analyzer
            metrics_tracker: Metrics tracker
            sequential_tester: Sequential tester
            novelty_detector: Novelty detector
            thompson_sampling: Thompson sampling
            interference_detector: Interference detector
            result_analyzer: Result analyzer
            experiment_monitor: Experiment monitor
            report_generator: Report generator
        """
        self.db = db_session

        # Initialize components
        self.designer = designer or ExperimentDesigner()
        self.assignment_engine = assignment_engine or AssignmentEngine(db_session)
        self.statistical_analyzer = statistical_analyzer or StatisticalAnalyzer()
        self.power_analyzer = power_analyzer or PowerAnalyzer()
        self.metrics_tracker = metrics_tracker or MetricsTracker(db_session)
        self.sequential_tester = sequential_tester or SequentialTester()
        self.novelty_detector = novelty_detector or NoveltyDetector()
        self.thompson_sampling = thompson_sampling or ThompsonSampling()
        self.interference_detector = interference_detector or InterferenceDetector()
        self.result_analyzer = result_analyzer or ResultAnalyzer()
        self.experiment_monitor = experiment_monitor or ExperimentMonitor()
        self.report_generator = report_generator or ReportGenerator()

        # Store active experiments
        self._experiments: Dict[str, ExperimentConfig] = {}

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
        secondary_metrics: Optional[List[MetricType]] = None,
        enable_sequential_testing: bool = False,
        enable_early_stopping: bool = False,
    ) -> ExperimentConfig:
        """
        Create new experiment.

        Args:
            name: Experiment name
            description: Description
            variants: List of variants
            platform: Platform type (linkedin/twitter/bluesky)
            primary_metric: Primary metric
            baseline_rate: Baseline rate
            mde: Minimum detectable effect
            alpha: Type I error rate
            power: Statistical power
            expected_daily_traffic: Expected daily traffic
            secondary_metrics: Secondary metrics
            enable_sequential_testing: Enable sequential testing
            enable_early_stopping: Enable early stopping

        Returns:
            Experiment configuration

        Raises:
            ValueError: If platform not supported
        """
        # Validate platform
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        # Design experiment
        config = self.designer.design(
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
            secondary_metrics=secondary_metrics,
            enable_sequential_testing=enable_sequential_testing,
            enable_early_stopping=enable_early_stopping,
        )

        # Store experiment
        self._experiments[config.experiment_id] = config

        # In production: save to database

        return config

    async def assign_user(
        self,
        experiment_id: str,
        user_id: str,
        session_id: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Assignment:
        """
        Assign user to variant.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            session_id: Optional session ID
            platform: Optional platform

        Returns:
            Assignment

        Raises:
            ValueError: If experiment not found or platform not supported
        """
        # Get experiment config
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        # Validate platform if provided
        if platform and platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        # Assign variant
        assignment = self.assignment_engine.assign(
            experiment_config=config,
            user_id=user_id,
            session_id=session_id,
            platform=platform,
        )

        return assignment

    async def track_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_type: MetricType,
        value: float,
        variant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Track metric event.

        Args:
            experiment_id: Experiment ID
            user_id: User ID
            metric_type: Metric type
            value: Metric value
            variant_id: Optional variant ID (will be looked up if not provided)
            session_id: Optional session ID
            metadata: Optional metadata

        Raises:
            ValueError: If experiment not found
        """
        # Get experiment config
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        # Get variant if not provided
        if not variant_id:
            assignment = self.assignment_engine.get_assignment(experiment_id, user_id)
            if not assignment:
                raise ValueError(f"User '{user_id}' not assigned to experiment")
            variant_id = assignment.variant_id

        # Track metric
        self.metrics_tracker.track(
            experiment_id=experiment_id,
            user_id=user_id,
            variant_id=variant_id,
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
    ) -> Dict[str, Any]:
        """
        Analyze experiment and return results.

        Args:
            experiment_id: Experiment ID
            alpha: Significance level
            min_sample_size: Minimum sample size

        Returns:
            Analysis results

        Raises:
            ValueError: If experiment not found
        """
        # Get experiment config
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        # Get metrics for control and treatment
        control_variant = next(v for v in config.variants if v.is_control)
        treatment_variants = [v for v in config.variants if not v.is_control]

        # For simplicity, analyze first treatment vs control
        if not treatment_variants:
            raise ValueError("No treatment variant found")

        treatment_variant = treatment_variants[0]

        # Get metric data
        control_data = np.array(
            self.metrics_tracker.get_metric_values(
                experiment_id=experiment_id,
                variant_id=control_variant.id,
                metric_type=config.primary_metric,
            )
        )

        treatment_data = np.array(
            self.metrics_tracker.get_metric_values(
                experiment_id=experiment_id,
                variant_id=treatment_variant.id,
                metric_type=config.primary_metric,
            )
        )

        # Check minimum sample size
        if len(control_data) < min_sample_size or len(treatment_data) < min_sample_size:
            return {
                "status": "insufficient_data",
                "n_control": len(control_data),
                "n_treatment": len(treatment_data),
                "required": min_sample_size,
            }

        # Analyze results
        result = self.result_analyzer.analyze_experiment(
            control_data=control_data,
            treatment_data=treatment_data,
            metric_type=config.primary_metric,
            alpha=alpha,
            min_sample_size=min_sample_size,
        )

        # Generate report
        report = self.report_generator.generate_report(
            experiment_config=config,
            experiment_result=result,
        )

        return {
            "status": "complete",
            "has_winner": result.has_winner,
            "winner_variant": result.winner_variant,
            "confidence": result.confidence,
            "should_launch": result.should_launch,
            "recommendation": result.recommendation,
            "statistical_result": {
                "p_value": result.statistical_result.p_value,
                "is_significant": result.statistical_result.is_significant,
                "effect_size": result.statistical_result.effect_size,
                "relative_diff": result.statistical_result.relative_diff,
                "ci_lower": result.statistical_result.ci_lower,
                "ci_upper": result.statistical_result.ci_upper,
            },
            "metrics": {
                "control": result.control_metrics,
                "treatment": result.treatment_metrics,
            },
            "report": report,
        }

    async def check_early_stopping(
        self, experiment_id: str
    ) -> Dict[str, Any]:
        """
        Check if experiment should stop early.

        Args:
            experiment_id: Experiment ID

        Returns:
            Early stopping decision

        Raises:
            ValueError: If experiment not found
        """
        # Get experiment config
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        if not config.enable_early_stopping:
            return {"should_stop": False, "reason": "early_stopping_disabled"}

        # Get metric data
        control_variant = next(v for v in config.variants if v.is_control)
        treatment_variants = [v for v in config.variants if not v.is_control]
        treatment_variant = treatment_variants[0]

        control_data = np.array(
            self.metrics_tracker.get_metric_values(
                experiment_id=experiment_id,
                variant_id=control_variant.id,
                metric_type=config.primary_metric,
            )
        )

        treatment_data = np.array(
            self.metrics_tracker.get_metric_values(
                experiment_id=experiment_id,
                variant_id=treatment_variant.id,
                metric_type=config.primary_metric,
            )
        )

        # Check with sequential tester if enabled
        if config.enable_sequential_testing:
            # For binary metrics
            control_successes = int(np.sum(control_data))
            treatment_successes = int(np.sum(treatment_data))

            sprt_result = self.sequential_tester.test(
                control_successes=control_successes,
                control_trials=len(control_data),
                treatment_successes=treatment_successes,
                treatment_trials=len(treatment_data),
                mde=config.mde,
            )

            if sprt_result.decision == "stop":
                return {
                    "should_stop": True,
                    "reason": "sequential_test",
                    "conclusion": sprt_result.conclusion,
                    "llr": sprt_result.log_likelihood_ratio,
                }

        return {"should_stop": False, "reason": "continue_collecting_data"}

    async def monitor_experiment(
        self, experiment_id: str
    ) -> Dict[str, Any]:
        """
        Monitor experiment health.

        Args:
            experiment_id: Experiment ID

        Returns:
            Health check result

        Raises:
            ValueError: If experiment not found
        """
        # Get experiment config
        config = self._experiments.get(experiment_id)
        if not config:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        # Get variant counts
        variant_counts = {}
        for variant in config.variants:
            metrics = self.metrics_tracker.get_metrics(
                experiment_id=experiment_id,
                variant_id=variant.id,
            )
            variant_counts[variant.id] = len(set(m.user_id for m in metrics))

        # Expected ratios
        expected_ratios = {v.id: v.traffic_allocation for v in config.variants}

        # Check health
        health = self.experiment_monitor.check_health(
            variant_counts=variant_counts,
            expected_ratios=expected_ratios,
        )

        return {
            "is_healthy": health.is_healthy,
            "issues": health.issues,
            "warnings": health.warnings,
            "metrics": health.metrics,
        }

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """
        Get experiment configuration.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment config or None
        """
        return self._experiments.get(experiment_id)

    def list_experiments(self) -> List[ExperimentConfig]:
        """
        List all experiments.

        Returns:
            List of experiment configs
        """
        return list(self._experiments.values())