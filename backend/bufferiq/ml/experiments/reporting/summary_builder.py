"""
Summary builder.

Builds executive summaries and insights.

Example:
```python
    builder = SummaryBuilder()
    
    summary = builder.build_executive_summary(result)
```
"""

from bufferiq.ml.experiments.results.analyzer import ExperimentResult


class SummaryBuilder:
    """
    Build experiment summaries.

    Example:
```python
        builder = SummaryBuilder()

        summary = builder.build_executive_summary(
            experiment_result=result
        )

        print(summary)
```
    """

    def build_executive_summary(self, result: ExperimentResult) -> str:
        """
        Build executive summary.

        Args:
            result: Experiment result

        Returns:
            Executive summary text
        """
        stat = result.statistical_result

        if result.should_launch:
            summary = (
                f"✓ **RECOMMENDATION: LAUNCH TREATMENT**\n\n"
                f"The treatment variant shows a statistically significant improvement "
                f"of **{stat.relative_diff:.1%}** over the control "
                f"(p-value: {stat.p_value:.4f}). "
                f"Based on {stat.n_treatment:,} samples in treatment and "
                f"{stat.n_control:,} in control, we can conclude with "
                f"**{result.confidence:.1%} confidence** that the treatment performs better. "
                f"The effect size ({stat.effect_size_type}) is {abs(stat.effect_size):.3f}, "
                f"indicating a meaningful practical difference."
            )
        elif result.has_winner and result.winner_variant == "control":
            summary = (
                f"✗ **RECOMMENDATION: DO NOT LAUNCH**\n\n"
                f"The control variant performs significantly better than treatment "
                f"by **{abs(stat.relative_diff):.1%}** "
                f"(p-value: {stat.p_value:.4f}). "
                f"Based on the statistical analysis of {stat.n_treatment:,} treatment samples "
                f"and {stat.n_control:,} control samples, launching the treatment "
                f"would likely harm performance."
            )
        else:
            summary = (
                f"⚠ **RECOMMENDATION: CONTINUE MONITORING**\n\n"
                f"The experiment has not yet produced statistically significant results "
                f"(p-value: {stat.p_value:.4f}). "
                f"Current sample sizes: {stat.n_treatment:,} treatment, "
                f"{stat.n_control:,} control. "
                f"Observed difference: {stat.relative_diff:.1%}. "
                f"Consider running longer to collect more data or re-evaluating the hypothesis."
            )

        return summary

    def build_insights(self, result: ExperimentResult) -> list[str]:
        """
        Build key insights.

        Args:
            result: Experiment result

        Returns:
            List of insights
        """
        insights = []

        stat = result.statistical_result

        # Statistical significance
        if stat.is_significant:
            insights.append(
                f"Statistically significant difference detected (p={stat.p_value:.4f})"
            )
        else:
            insights.append(
                f"No significant difference (p={stat.p_value:.4f})"
            )

        # Effect size
        if abs(stat.effect_size) > 0.8:
            insights.append("Large effect size - strong treatment impact")
        elif abs(stat.effect_size) > 0.5:
            insights.append("Medium effect size - moderate treatment impact")
        elif abs(stat.effect_size) > 0.2:
            insights.append("Small effect size - weak treatment impact")

        # Practical significance
        if abs(stat.relative_diff) > 0.10:
            insights.append(
                f"Large practical improvement: {abs(stat.relative_diff):.1%}"
            )
        elif abs(stat.relative_diff) > 0.05:
            insights.append(
                f"Moderate practical improvement: {abs(stat.relative_diff):.1%}"
            )

        return insights