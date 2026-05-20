"""
Winner selector.

Determines experiment winner based on statistical results.

Example:
```python
    selector = WinnerSelector()
    
    winner = selector.select_winner(
        stat_result=test_result,
        min_improvement=0.05
    )
```
"""

from typing import Dict, Optional

from bufferiq.ml.experiments.statistics.hypothesis_tester import HypothesisTestResult


class WinnerSelector:
    """
    Select experiment winner.

    Example:
```python
        selector = WinnerSelector()

        winner = selector.select_winner(
            stat_result=statistical_result,
            min_improvement=0.01,
            alpha=0.05
        )

        if winner['has_winner']:
            print(f"Winner: {winner['winner']}")
            print(f"Confidence: {winner['confidence']:.1%}")
```
    """

    def select_winner(
        self,
        stat_result: HypothesisTestResult,
        min_improvement: float = 0.01,
        alpha: float = 0.05,
    ) -> Dict[str, any]:
        """
        Select winner based on statistical result.

        Args:
            stat_result: Statistical test result
            min_improvement: Minimum practical improvement
            alpha: Significance level

        Returns:
            Winner selection result
        """
        # No winner if not significant
        if not stat_result.is_significant:
            return {
                "has_winner": False,
                "winner": None,
                "confidence": 0.0,
                "reason": "not_significant",
            }

        # Determine winner
        if stat_result.treatment_mean > stat_result.control_mean:
            winner = "treatment"
            improvement = stat_result.relative_diff
        else:
            winner = "control"
            improvement = -stat_result.relative_diff

        # Check minimum improvement
        meets_minimum = abs(improvement) >= min_improvement

        # Calculate confidence (1 - p_value)
        confidence = 1 - stat_result.p_value

        return {
            "has_winner": True,
            "winner": winner,
            "confidence": confidence,
            "improvement": improvement,
            "meets_minimum": meets_minimum,
            "p_value": stat_result.p_value,
            "effect_size": stat_result.effect_size,
        }

    def select_winner_bayesian(
        self,
        probability_beat_control: float,
        expected_loss: float,
        threshold: float = 0.95,
        loss_threshold: float = 0.01,
    ) -> Dict[str, any]:
        """
        Select winner using Bayesian analysis.

        Args:
            probability_beat_control: P(treatment > control)
            expected_loss: Expected loss of choosing treatment
            threshold: Probability threshold
            loss_threshold: Maximum acceptable loss

        Returns:
            Winner selection result
        """
        # Treatment wins if high probability and low loss
        if (
            probability_beat_control >= threshold
            and expected_loss <= loss_threshold
        ):
            return {
                "has_winner": True,
                "winner": "treatment",
                "confidence": probability_beat_control,
                "expected_loss": expected_loss,
            }

        # Control wins if treatment unlikely to win
        if probability_beat_control <= (1 - threshold):
            return {
                "has_winner": True,
                "winner": "control",
                "confidence": 1 - probability_beat_control,
                "expected_loss": 0.0,
            }

        # No clear winner
        return {
            "has_winner": False,
            "winner": None,
            "confidence": max(probability_beat_control, 1 - probability_beat_control),
            "reason": "inconclusive",
        }