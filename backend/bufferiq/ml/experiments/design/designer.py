"""
Experiment designer.

Designs A/B tests and multivariate experiments with proper
statistical configuration and validation.

Key features:
    - A/B test design
    - Multivariate test design
    - Sample size calculation
    - Traffic allocation
    - Platform validation
    - Duration estimation

Example:
```python
    designer = ExperimentDesigner()
    
    variants = [
        Variant(
            id="control",
            name="Original",
            traffic_allocation=0.5,
            changes={},
            is_control=True
        ),
        Variant(
            id="treatment",
            name="New Headline",
            traffic_allocation=0.5,
            changes={"headline": "AI-powered"}
        )
    ]
    
    config = designer.design(
        name="Headline Test",
        variants=variants,
        platform="linkedin",
        primary_metric=MetricType.ENGAGEMENT_RATE,
        baseline_rate=0.05,
        mde=0.10
    )
```
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from bufferiq.ml.experiments.design.sample_size_calculator import (
    SampleSizeCalculator,
)

# Supported platforms
SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


class ExperimentType(Enum):
    """Experiment types."""

    AB_TEST = "ab_test"  # 2 variants
    MULTIVARIATE = "multivariate"  # Multiple factors
    SEQUENTIAL = "sequential"  # Sequential testing
    BANDIT = "bandit"  # Multi-armed bandit


class MetricType(Enum):
    """Metric types."""

    ENGAGEMENT_RATE = "engagement_rate"
    CLICK_THROUGH_RATE = "click_through_rate"
    CONVERSION_RATE = "conversion_rate"
    TIME_ON_PAGE = "time_on_page"
    REVENUE = "revenue"
    CUSTOM = "custom"


@dataclass
class Variant:
    """Experiment variant configuration."""

    id: str
    name: str
    description: str
    traffic_allocation: float  # 0-1, must sum to 1 across variants

    # Treatment configuration
    changes: Dict[str, Any]  # What's different in this variant

    # Control flag
    is_control: bool = False

    def __post_init__(self) -> None:
        """Validate variant configuration."""
        if not 0 <= self.traffic_allocation <= 1:
            raise ValueError(
                f"Traffic allocation must be between 0 and 1, "
                f"got {self.traffic_allocation}"
            )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_id: str
    name: str
    description: str
    type: ExperimentType

    # Variants
    variants: List[Variant]

    # Target
    platform: str
    target_audience: Optional[str] = None

    # Metrics
    primary_metric: MetricType = MetricType.ENGAGEMENT_RATE
    secondary_metrics: List[MetricType] = field(default_factory=list)

    # Statistical parameters
    alpha: float = 0.05  # Type I error rate
    power: float = 0.80  # Statistical power (1 - Type II error)
    mde: float = 0.10  # Minimum detectable effect (10% relative change)

    # Sample size
    required_sample_size: int = 0  # Calculated

    # Duration
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    estimated_duration_days: int = 0

    # Advanced options
    enable_sequential_testing: bool = False
    enable_early_stopping: bool = False
    stratification_key: Optional[str] = None  # e.g., "user_type"

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate experiment configuration."""
        if self.platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{self.platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )


class ExperimentDesigner:
    """
    Design A/B tests and multivariate experiments.

    Handles experiment configuration, sample size calculation,
    and statistical parameter validation.

    Example:
```python
        designer = ExperimentDesigner()

        variants = [
            Variant(
                id="control",
                name="Original",
                description="Current headline",
                traffic_allocation=0.5,
                changes={},
                is_control=True
            ),
            Variant(
                id="treatment",
                name="New Headline",
                description="AI-powered headline",
                traffic_allocation=0.5,
                changes={"headline": "New AI headline"}
            )
        ]

        config = designer.design(
            name="Headline Test",
            description="Test AI-generated headlines",
            variants=variants,
            platform="linkedin",
            primary_metric=MetricType.ENGAGEMENT_RATE,
            baseline_rate=0.05,
            mde=0.10
        )

        print(f"Required sample size: {config.required_sample_size:,}")
        print(f"Estimated duration: {config.estimated_duration_days} days")
```
    """

    def __init__(self) -> None:
        """Initialize experiment designer."""
        self.sample_size_calculator = SampleSizeCalculator()

    def design(
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
        target_audience: Optional[str] = None,
        enable_sequential_testing: bool = False,
        enable_early_stopping: bool = False,
        stratification_key: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> ExperimentConfig:
        """
        Design experiment with statistical parameters.

        Args:
            name: Experiment name
            description: Experiment description
            variants: List of variants
            platform: Platform type (linkedin/twitter/bluesky)
            primary_metric: Primary success metric
            baseline_rate: Baseline conversion/engagement rate
            mde: Minimum detectable effect (relative change)
            alpha: Type I error rate
            power: Statistical power
            expected_daily_traffic: Expected traffic per day
            secondary_metrics: Optional secondary metrics
            target_audience: Optional audience filter
            enable_sequential_testing: Enable sequential testing
            enable_early_stopping: Enable early stopping
            stratification_key: Optional stratification key
            created_by: Optional creator ID

        Returns:
            Complete experiment configuration

        Raises:
            ValueError: If platform not supported or invalid config
        """
        # Validate platform
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        # Validate variants
        self._validate_variants(variants)

        # Calculate sample size
        sample_size = self.sample_size_calculator.calculate(
            baseline_rate=baseline_rate,
            mde=mde,
            alpha=alpha,
            power=power,
            num_variants=len(variants),
        )

        # Estimate duration
        duration_days = 0
        if expected_daily_traffic:
            duration_days = self._estimate_duration(
                sample_size=sample_size,
                daily_traffic=expected_daily_traffic,
                num_variants=len(variants),
            )

        # Determine experiment type
        exp_type = (
            ExperimentType.AB_TEST
            if len(variants) == 2
            else ExperimentType.MULTIVARIATE
        )

        # Create config
        config = ExperimentConfig(
            experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=name,
            description=description,
            type=exp_type,
            variants=variants,
            platform=platform,
            target_audience=target_audience,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            alpha=alpha,
            power=power,
            mde=mde,
            required_sample_size=sample_size,
            estimated_duration_days=duration_days,
            enable_sequential_testing=enable_sequential_testing,
            enable_early_stopping=enable_early_stopping,
            stratification_key=stratification_key,
            created_by=created_by,
        )

        return config

    def _validate_variants(self, variants: List[Variant]) -> None:
        """
        Validate variant configuration.

        Args:
            variants: List of variants

        Raises:
            ValueError: If variants invalid
        """
        if len(variants) < 2:
            raise ValueError("At least 2 variants required")

        # Check traffic allocation sums to 1
        total_allocation = sum(v.traffic_allocation for v in variants)
        if not np.isclose(total_allocation, 1.0, atol=0.01):
            raise ValueError(
                f"Traffic allocation must sum to 1.0, got {total_allocation:.4f}"
            )

        # Check exactly one control
        control_count = sum(1 for v in variants if v.is_control)
        if control_count != 1:
            raise ValueError("Exactly one variant must be control")

        # Check unique IDs
        ids = [v.id for v in variants]
        if len(ids) != len(set(ids)):
            raise ValueError("Variant IDs must be unique")

    def _estimate_duration(
        self, sample_size: int, daily_traffic: int, num_variants: int
    ) -> int:
        """
        Estimate experiment duration in days.

        Args:
            sample_size: Required sample size per variant
            daily_traffic: Expected daily traffic
            num_variants: Number of variants

        Returns:
            Estimated duration in days
        """
        if daily_traffic <= 0:
            return 0

        traffic_per_variant = daily_traffic / num_variants
        days = int(np.ceil(sample_size / traffic_per_variant))

        # Minimum 7 days for valid results
        return max(7, days)

    def create_ab_test(
        self,
        name: str,
        description: str,
        control_name: str,
        treatment_name: str,
        treatment_changes: Dict[str, Any],
        platform: str,
        primary_metric: MetricType,
        baseline_rate: float,
        mde: float = 0.10,
        traffic_split: float = 0.5,
        **kwargs: Any,
    ) -> ExperimentConfig:
        """
        Create simple A/B test (2 variants).

        Args:
            name: Experiment name
            description: Description
            control_name: Control variant name
            treatment_name: Treatment variant name
            treatment_changes: Changes in treatment
            platform: Platform
            primary_metric: Primary metric
            baseline_rate: Baseline rate
            mde: Minimum detectable effect
            traffic_split: Traffic to treatment (0-1)
            **kwargs: Additional design parameters

        Returns:
            Experiment configuration
        """
        variants = [
            Variant(
                id="control",
                name=control_name,
                description="Control variant",
                traffic_allocation=1 - traffic_split,
                changes={},
                is_control=True,
            ),
            Variant(
                id="treatment",
                name=treatment_name,
                description="Treatment variant",
                traffic_allocation=traffic_split,
                changes=treatment_changes,
                is_control=False,
            ),
        ]

        return self.design(
            name=name,
            description=description,
            variants=variants,
            platform=platform,
            primary_metric=primary_metric,
            baseline_rate=baseline_rate,
            mde=mde,
            **kwargs,
        )