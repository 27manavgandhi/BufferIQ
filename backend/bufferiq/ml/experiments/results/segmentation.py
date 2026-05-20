"""
Segmentation analyzer.

Analyzes results across different user segments.

Example:
```python
    analyzer = SegmentationAnalyzer()
    
    segments = analyzer.analyze_by_segment(
        data=data,
        segment_key="user_type"
    )
```
"""

from typing import Dict, List, Any

import numpy as np

from bufferiq.ml.experiments.design.designer import MetricType
from bufferiq.ml.experiments.statistics.hypothesis_tester import StatisticalAnalyzer


class SegmentationAnalyzer:
    """
    Analyze results by segments.

    Example:
```python
        analyzer = SegmentationAnalyzer()

        results = analyzer.analyze_by_segments(
            control_data_by_segment={
                "new_users": [0, 1, 0, 1],
                "returning_users": [1, 1, 1, 0]
            },
            treatment_data_by_segment={
                "new_users": [1, 1, 1, 1],
                "returning_users": [1, 1, 0, 1]
            },
            metric_type=MetricType.ENGAGEMENT_RATE
        )

        for segment, result in results.items():
            print(f"{segment}: p={result['p_value']:.3f}")
```
    """

    def __init__(self) -> None:
        """Initialize segmentation analyzer."""
        self.statistical_analyzer = StatisticalAnalyzer()

    def analyze_by_segments(
        self,
        control_data_by_segment: Dict[str, List[float]],
        treatment_data_by_segment: Dict[str, List[float]],
        metric_type: MetricType,
        alpha: float = 0.05,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze results for each segment.

        Args:
            control_data_by_segment: Control data by segment
            treatment_data_by_segment: Treatment data by segment
            metric_type: Metric type
            alpha: Significance level

        Returns:
            Dictionary of results by segment
        """
        results = {}

        for segment in control_data_by_segment.keys():
            if segment not in treatment_data_by_segment:
                continue

            control = np.array(control_data_by_segment[segment])
            treatment = np.array(treatment_data_by_segment[segment])

            if len(control) < 10 or len(treatment) < 10:
                results[segment] = {
                    "status": "insufficient_data",
                    "n_control": len(control),
                    "n_treatment": len(treatment),
                }
                continue

            # Run test
            stat_result = self.statistical_analyzer.analyze(
                control_data=control,
                treatment_data=treatment,
                metric_type=metric_type,
                alpha=alpha,
            )

            results[segment] = {
                "status": "analyzed",
                "is_significant": stat_result.is_significant,
                "p_value": stat_result.p_value,
                "control_mean": stat_result.control_mean,
                "treatment_mean": stat_result.treatment_mean,
                "relative_diff": stat_result.relative_diff,
                "effect_size": stat_result.effect_size,
                "n_control": stat_result.n_control,
                "n_treatment": stat_result.n_treatment,
            }

        return results

    def identify_best_segments(
        self, segment_results: Dict[str, Dict[str, Any]], top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Identify segments with best treatment performance.

        Args:
            segment_results: Results by segment
            top_n: Number of top segments to return

        Returns:
            List of top segments
        """
        # Filter valid segments
        valid = [
            {
                "segment": seg,
                "relative_diff": res["relative_diff"],
                "is_significant": res["is_significant"],
                "p_value": res["p_value"],
            }
            for seg, res in segment_results.items()
            if res.get("status") == "analyzed"
        ]

        # Sort by relative difference
        sorted_segments = sorted(
            valid, key=lambda x: x["relative_diff"], reverse=True
        )

        return sorted_segments[:top_n]

    def calculate_heterogeneous_treatment_effects(
        self, segment_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate heterogeneous treatment effects.

        Args:
            segment_results: Results by segment

        Returns:
            HTE analysis
        """
        # Extract effects
        effects = []
        segments = []

        for segment, result in segment_results.items():
            if result.get("status") == "analyzed":
                effects.append(result["relative_diff"])
                segments.append(segment)

        if not effects:
            return {"has_hte": False, "reason": "no_valid_segments"}

        effects = np.array(effects)

        # Calculate variance
        effect_variance = float(np.var(effects))
        effect_std = float(np.std(effects))
        mean_effect = float(np.mean(effects))

        # High variance indicates HTE
        has_hte = effect_std > 0.05  # 5% threshold

        return {
            "has_hte": has_hte,
            "mean_effect": mean_effect,
            "effect_std": effect_std,
            "effect_variance": effect_variance,
            "min_effect": float(np.min(effects)),
            "max_effect": float(np.max(effects)),
            "num_segments": len(effects),
            "segments": segments,
            "effects": effects.tolist(),
        }