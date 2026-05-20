"""
Funnel analyzer.

Analyzes conversion funnels for experiments.

Example:
```python
    analyzer = FunnelAnalyzer()
    
    funnel = analyzer.analyze(
        events=events,
        steps=["view", "click", "convert"]
    )
    
    print(f"Overall conversion: {funnel['overall_conversion']:.1%}")
```
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from bufferiq.ml.experiments.metrics.tracker import MetricEvent


@dataclass
class FunnelStep:
    """Funnel step data."""

    step_name: str
    user_count: int
    conversion_rate: float
    drop_off_rate: float


@dataclass
class FunnelAnalysis:
    """Funnel analysis result."""

    steps: List[FunnelStep]
    overall_conversion: float
    total_users: int


class FunnelAnalyzer:
    """
    Analyze conversion funnels.

    Example:
```python
        analyzer = FunnelAnalyzer()

        # Define funnel steps
        steps = ["impression", "click", "engagement", "conversion"]

        # Analyze funnel
        result = analyzer.analyze_funnel(
            events=events,
            steps=steps
        )

        for step in result.steps:
            print(f"{step.step_name}: {step.conversion_rate:.1%}")
```
    """

    def analyze_funnel(
        self, events: List[MetricEvent], steps: List[str]
    ) -> FunnelAnalysis:
        """
        Analyze conversion funnel.

        Args:
            events: List of metric events
            steps: Ordered list of funnel steps

        Returns:
            Funnel analysis
        """
        # Count users at each step
        step_users: Dict[str, set] = {step: set() for step in steps}

        for event in events:
            step_name = event.metadata.get("step") if event.metadata else None
            if step_name and step_name in step_users:
                step_users[step_name].add(event.user_id)

        # Calculate funnel metrics
        total_users = len(step_users[steps[0]]) if steps else 0
        funnel_steps: List[FunnelStep] = []

        for i, step in enumerate(steps):
            count = len(step_users[step])

            if i == 0:
                conversion_rate = 1.0 if count > 0 else 0.0
                drop_off_rate = 0.0
            else:
                prev_count = len(step_users[steps[i - 1]])
                conversion_rate = count / prev_count if prev_count > 0 else 0.0
                drop_off_rate = 1.0 - conversion_rate

            funnel_steps.append(
                FunnelStep(
                    step_name=step,
                    user_count=count,
                    conversion_rate=conversion_rate,
                    drop_off_rate=drop_off_rate,
                )
            )

        # Overall conversion
        overall_conversion = (
            len(step_users[steps[-1]]) / total_users if total_users > 0 else 0.0
        )

        return FunnelAnalysis(
            steps=funnel_steps,
            overall_conversion=overall_conversion,
            total_users=total_users,
        )

    def compare_funnels(
        self,
        control_events: List[MetricEvent],
        treatment_events: List[MetricEvent],
        steps: List[str],
    ) -> Dict[str, FunnelAnalysis]:
        """
        Compare funnels between variants.

        Args:
            control_events: Control events
            treatment_events: Treatment events
            steps: Funnel steps

        Returns:
            Dictionary with control and treatment funnels
        """
        control_funnel = self.analyze_funnel(control_events, steps)
        treatment_funnel = self.analyze_funnel(treatment_events, steps)

        return {"control": control_funnel, "treatment": treatment_funnel}