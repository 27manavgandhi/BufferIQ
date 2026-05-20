"""
Result analyzer.

Analyzes experiment results comprehensively.

Example:
```python
    analyzer = ResultAnalyzer()
    
    result = analyzer.analyze_experiment(
        control_data=control,
        treatment_data=treatment,
        metric_type=MetricType.ENGAGEMENT_RATE
    )
```
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from bufferiq.ml.experiments.design.designer import MetricType
from bufferiq.ml.experiments.statistics.hypothesis_tester import (
    StatisticalAnalyzer,
    HypothesisTestResult,
)
from bufferiq.ml.experiments.results.winner_selector import WinnerSelector


@dataclass
class ExperimentResult:
    """Complete experiment result."""

    # Hypothesis test
    statistical_result: HypothesisTestResult

    # Winner
    has_winner: bool
    winner_variant: Optional[str]
    confidence: float

    # Metrics
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]

    # Recommendations
    recommendation: str
    should_launch: bool

    # Additional analysis
    segments: Optional[Dict] = None
    time_series: Optional[Dict] = None


class ResultAnalyzer:
    """
    Analyze experiment results.

    Example:
```python
        analyzer = ResultAnalyzer()

        result = analyzer.analyze_experiment(
            control_data=np.array([0, 1, 0, 1, 1]),
            treatment_data=np.array([1, 1, 1, 1, 0]),
            metric_type=MetricType.ENGAGEMENT_RATE,
            alpha=0.05
        )

        if result.has_winner:
            print(f"Winner: {result.winner_variant}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Recommendation: {result.recommendation}")
```
    """

    def __init__(self) -> None:
        """Initialize result analyzer."""
        self.statistical_analyzer = StatisticalAnalyzer()
        self.winner_selector = WinnerSelector()

    def analyze_experiment(
        self,
        control_data: np.ndarray,
        treatment_data: np.ndarray,
        metric_type: MetricType,
        alpha: float = 0.05,
        min_sample_size: int = 100,
        min_improvement: float = 0.01,
    ) -> ExperimentResult:
        """
        Analyze experiment and determine results.

        Args:
            control_data: Control data
            treatment_data: Treatment data
            metric_type: Metric type
            alpha: Significance level
            min_sample_size: Minimum sample size
            min_improvement: Minimum practical improvement

        Returns:
            Experiment result
        """
        # Validate sample sizes
        if len(control_data) < min_sample_size or len(treatment_data) < min_sample_size:
            return self._insufficient_data_result(
                len(control_data), len(treatment_data), min_sample_size
            )

        # Run statistical test
        stat_result = self.statistical_analyzer.analyze(
            control_data=control_data,
            treatment_data=treatment_data,
            metric_type=metric_type,
            alpha=alpha,
        )

        # Calculate metrics
        control_metrics = self._calculate_metrics(control_data)
        treatment_metrics = self._calculate_metrics(treatment_data)

        # Determine winner
        winner_result = self.winner_selector.select_winner(
            stat_result=stat_result,
            min_improvement=min_improvement,
            alpha=alpha,
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            stat_result=stat_result,
            winner_result=winner_result,
            min_improvement=min_improvement,
        )

        # Decide if should launch
        should_launch = (
            winner_result["has_winner"]
            and winner_result["winner"] == "treatment"
            and stat_result.relative_diff >= min_improvement
        )

        return ExperimentResult(
            statistical_result=stat_result,
            has_winner=winner_result["has_winner"],
            winner_variant=winner_result["winner"],
            confidence=winner_result["confidence"],
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics,
            recommendation=recommendation,
            should_launch=should_launch,
        )

    def _calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate summary metrics.

        Args:
            data: Data array

        Returns:
            Dictionary of metrics
        """
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "count": len(data),
            "sum": float(np.sum(data)),
        }

    def _insufficient_data_result(
        self, n_control: int, n_treatment: int, min_required: int
    ) -> ExperimentResult:
        """
        Create result for insufficient data.

        Args:
            n_control: Control sample size
            n_treatment: Treatment sample size
            min_required: Minimum required

        Returns:
            Experiment result
        """
        from bufferiq.ml.experiments.statistics.hypothesis_tester import HypothesisTestResult

        # Dummy statistical result
        stat_result = HypothesisTestResult(
            test_type="insufficient_data",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=0.05,
            effect_size=0.0,
            effect_size_type="none",
            ci_lower=0.0,
            ci_upper=0.0,
            confidence_level=0.95,
            n_control=n_control,
            n_treatment=n_treatment,
            control_mean=0.0,
            treatment_mean=0.0,
            absolute_diff=0.0,
            relative_diff=0.0,
        )

        return ExperimentResult(
            statistical_result=stat_result,
            has_winner=False,
            winner_variant=None,
            confidence=0.0,
            control_metrics={"count": n_control},
            treatment_metrics={"count": n_treatment},
            recommendation=f"Insufficient data. Need at least {min_required} samples per variant.",
            should_launch=False,
        )

    def _generate_recommendation(
        self,
        stat_result: HypothesisTestResult,
        winner_result: Dict,
        min_improvement: float,
    ) -> str:
        """
        Generate recommendation.

        Args:
            stat_result: Statistical result
            winner_result: Winner result
            min_improvement: Minimum improvement

        Returns:
            Recommendation string
        """
        if not stat_result.is_significant:
            return (
                "No significant difference detected. "
                "Consider running longer or increasing sample size."
            )

        if not winner_result["has_winner"]:
            return "Results inconclusive. Continue monitoring."

        winner = winner_result["winner"]
        improvement = stat_result.relative_diff

        if winner == "treatment":
            if improvement >= min_improvement:
                return (
                    f"✓ Launch treatment. "
                    f"Significant improvement of {improvement:.1%} detected. "
                    f"Confidence: {winner_result['confidence']:.1%}"
                )
            else:
                return (
                    f"Treatment wins but improvement ({improvement:.1%}) "
                    f"is below minimum threshold ({min_improvement:.1%}). "
                    "Consider business impact before launching."
                )
        else:
            return (
                f"Control performs better. "
                f"Do not launch treatment. "
                f"Difference: {abs(improvement):.1%}"
            )