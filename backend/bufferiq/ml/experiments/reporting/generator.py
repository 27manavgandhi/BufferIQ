"""
Report generator.

Generates comprehensive experiment reports.

Example:
```python
    generator = ReportGenerator()
    
    report = generator.generate_report(
        experiment_config=config,
        experiment_result=result
    )
```
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from bufferiq.ml.experiments.design.designer import ExperimentConfig
from bufferiq.ml.experiments.results.analyzer import ExperimentResult
from bufferiq.ml.experiments.reporting.summary_builder import SummaryBuilder


@dataclass
class ExperimentReport:
    """Complete experiment report."""

    # Metadata
    experiment_id: str
    experiment_name: str
    generated_at: datetime

    # Configuration
    config_summary: Dict

    # Results
    result_summary: Dict

    # Executive summary
    executive_summary: str

    # Detailed analysis
    statistical_details: Dict
    metrics_breakdown: Dict

    # Recommendations
    recommendations: list[str]
    next_steps: list[str]


class ReportGenerator:
    """
    Generate experiment reports.

    Example:
```python
        generator = ReportGenerator()

        report = generator.generate_report(
            experiment_config=config,
            experiment_result=result
        )

        print(report.executive_summary)
        print(f"Recommendations: {report.recommendations}")
```
    """

    def __init__(self) -> None:
        """Initialize report generator."""
        self.summary_builder = SummaryBuilder()

    def generate_report(
        self,
        experiment_config: ExperimentConfig,
        experiment_result: ExperimentResult,
    ) -> ExperimentReport:
        """
        Generate comprehensive report.

        Args:
            experiment_config: Experiment configuration
            experiment_result: Experiment result

        Returns:
            Complete report
        """
        # Build summaries
        config_summary = self._summarize_config(experiment_config)
        result_summary = self._summarize_result(experiment_result)

        # Executive summary
        executive_summary = self.summary_builder.build_executive_summary(
            experiment_result
        )

        # Statistical details
        stat_details = self._build_statistical_details(experiment_result)

        # Metrics breakdown
        metrics = self._build_metrics_breakdown(experiment_result)

        # Recommendations
        recommendations = self._generate_recommendations(experiment_result)
        next_steps = self._generate_next_steps(experiment_result)

        return ExperimentReport(
            experiment_id=experiment_config.experiment_id,
            experiment_name=experiment_config.name,
            generated_at=datetime.now(),
            config_summary=config_summary,
            result_summary=result_summary,
            executive_summary=executive_summary,
            statistical_details=stat_details,
            metrics_breakdown=metrics,
            recommendations=recommendations,
            next_steps=next_steps,
        )

    def _summarize_config(self, config: ExperimentConfig) -> Dict:
        """Summarize experiment configuration."""
        return {
            "name": config.name,
            "description": config.description,
            "type": config.type.value,
            "platform": config.platform,
            "num_variants": len(config.variants),
            "primary_metric": config.primary_metric.value,
            "alpha": config.alpha,
            "power": config.power,
            "mde": config.mde,
            "required_sample_size": config.required_sample_size,
        }

    def _summarize_result(self, result: ExperimentResult) -> Dict:
        """Summarize experiment result."""
        return {
            "has_winner": result.has_winner,
            "winner": result.winner_variant,
            "confidence": result.confidence,
            "should_launch": result.should_launch,
            "is_significant": result.statistical_result.is_significant,
            "p_value": result.statistical_result.p_value,
            "relative_improvement": result.statistical_result.relative_diff,
        }

    def _build_statistical_details(self, result: ExperimentResult) -> Dict:
        """Build statistical details."""
        stat = result.statistical_result

        return {
            "test_type": stat.test_type,
            "statistic": stat.statistic,
            "p_value": stat.p_value,
            "effect_size": stat.effect_size,
            "effect_size_type": stat.effect_size_type,
            "confidence_interval": {
                "lower": stat.ci_lower,
                "upper": stat.ci_upper,
                "level": stat.confidence_level,
            },
            "sample_sizes": {
                "control": stat.n_control,
                "treatment": stat.n_treatment,
            },
        }

    def _build_metrics_breakdown(self, result: ExperimentResult) -> Dict:
        """Build metrics breakdown."""
        return {
            "control": result.control_metrics,
            "treatment": result.treatment_metrics,
            "absolute_difference": result.statistical_result.absolute_diff,
            "relative_difference": result.statistical_result.relative_diff,
        }

    def _generate_recommendations(self, result: ExperimentResult) -> list[str]:
        """Generate recommendations."""
        recommendations = []

        if result.should_launch:
            recommendations.append("✓ Launch treatment variant to all users")
            recommendations.append(
                f"Expected improvement: {result.statistical_result.relative_diff:.1%}"
            )
        elif result.has_winner and result.winner_variant == "control":
            recommendations.append("✗ Do not launch treatment - control performs better")
        else:
            recommendations.append("Continue monitoring or run longer experiment")

        return recommendations

    def _generate_next_steps(self, result: ExperimentResult) -> list[str]:
        """Generate next steps."""
        next_steps = []

        if result.should_launch:
            next_steps.append("Prepare launch plan")
            next_steps.append("Monitor post-launch metrics")
            next_steps.append("Document learnings")
        else:
            next_steps.append("Analyze why treatment didn't win")
            next_steps.append("Consider alternative treatments")
            next_steps.append("Review hypothesis")

        return next_steps

    def export_to_markdown(self, report: ExperimentReport) -> str:
        """
        Export report to Markdown.

        Args:
            report: Experiment report

        Returns:
            Markdown string
        """
        md = f"""# Experiment Report: {report.experiment_name}

**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}  
**Experiment ID:** {report.experiment_id}

---

## Executive Summary

{report.executive_summary}

---

## Configuration

- **Type:** {report.config_summary['type']}
- **Platform:** {report.config_summary['platform']}
- **Primary Metric:** {report.config_summary['primary_metric']}
- **Number of Variants:** {report.config_summary['num_variants']}
- **Significance Level (α):** {report.config_summary['alpha']}
- **Statistical Power:** {report.config_summary['power']}
- **Minimum Detectable Effect:** {report.config_summary['mde']:.1%}
- **Required Sample Size:** {report.config_summary['required_sample_size']:,} per variant

---

## Results

**Winner:** {report.result_summary['winner'] or 'No clear winner'}  
**Confidence:** {report.result_summary['confidence']:.1%}  
**P-value:** {report.result_summary['p_value']:.4f}  
**Relative Improvement:** {report.result_summary['relative_improvement']:.2%}  
**Launch Recommended:** {'✓ Yes' if report.result_summary['should_launch'] else '✗ No'}

---

## Statistical Details

- **Test Type:** {report.statistical_details['test_type']}
- **Test Statistic:** {report.statistical_details['statistic']:.4f}
- **Effect Size ({report.statistical_details['effect_size_type']}):** {report.statistical_details['effect_size']:.4f}
- **Confidence Interval ({report.statistical_details['confidence_interval']['level']:.0%}):** [{report.statistical_details['confidence_interval']['lower']:.4f}, {report.statistical_details['confidence_interval']['upper']:.4f}]

---

## Metrics Breakdown

### Control
- **Mean:** {report.metrics_breakdown['control']['mean']:.4f}
- **Count:** {report.metrics_breakdown['control']['count']:,}

### Treatment
- **Mean:** {report.metrics_breakdown['treatment']['mean']:.4f}
- **Count:** {report.metrics_breakdown['treatment']['count']:,}

### Difference
- **Absolute:** {report.metrics_breakdown['absolute_difference']:.4f}
- **Relative:** {report.metrics_breakdown['relative_difference']:.2%}

---

## Recommendations

"""
        for rec in report.recommendations:
            md += f"- {rec}\n"

        md += "\n---\n\n## Next Steps\n\n"

        for step in report.next_steps:
            md += f"1. {step}\n"

        return md