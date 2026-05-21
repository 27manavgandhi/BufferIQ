"""
API models for experiments.

Pydantic models for request/response validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator

from bufferiq.ml.experiments.design.designer import (
    MetricType,
    ExperimentType,
    SUPPORTED_PLATFORMS,
)


class VariantCreate(BaseModel):
    """Create variant request."""

    id: str = Field(..., description="Variant ID")
    name: str = Field(..., description="Variant name")
    description: str = Field(..., description="Variant description")
    traffic_allocation: float = Field(
        ..., ge=0.0, le=1.0, description="Traffic allocation (0-1)"
    )
    changes: Dict[str, Any] = Field(default_factory=dict, description="Changes")
    is_control: bool = Field(default=False, description="Is control variant")


class ExperimentCreate(BaseModel):
    """Create experiment request."""

    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Description")
    variants: List[VariantCreate] = Field(..., min_items=2, description="Variants")
    platform: str = Field(..., description="Platform (linkedin/twitter/bluesky)")
    primary_metric: MetricType = Field(..., description="Primary metric")
    baseline_rate: float = Field(..., gt=0, lt=1, description="Baseline rate")
    mde: float = Field(default=0.10, gt=0, lt=1, description="Minimum detectable effect")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Type I error rate")
    power: float = Field(default=0.80, gt=0, lt=1, description="Statistical power")
    expected_daily_traffic: Optional[int] = Field(None, gt=0, description="Daily traffic")
    enable_sequential_testing: bool = Field(default=False, description="Enable sequential testing")
    enable_early_stopping: bool = Field(default=False, description="Enable early stopping")

    @validator("platform")
    def validate_platform(cls, v):
        """Validate platform."""
        if v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{v}' not supported. Supported: {SUPPORTED_PLATFORMS}"
            )
        return v


class ExperimentResponse(BaseModel):
    """Experiment response."""

    experiment_id: str
    name: str
    description: str
    type: str
    platform: str
    primary_metric: str
    num_variants: int
    required_sample_size: int
    estimated_duration_days: int
    created_at: datetime

    class Config:
        from_attributes = True


class AssignmentRequest(BaseModel):
    """Assignment request."""

    experiment_id: str = Field(..., description="Experiment ID")
    user_id: str = Field(..., description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    platform: Optional[str] = Field(None, description="Platform")

    @validator("platform")
    def validate_platform(cls, v):
        """Validate platform."""
        if v and v not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{v}' not supported. Supported: {SUPPORTED_PLATFORMS}"
            )
        return v


class AssignmentResponse(BaseModel):
    """Assignment response."""

    experiment_id: str
    user_id: str
    variant_id: str
    variant_name: str
    assigned_at: datetime
    is_new_assignment: bool


class MetricTrackRequest(BaseModel):
    """Track metric request."""

    experiment_id: str = Field(..., description="Experiment ID")
    user_id: str = Field(..., description="User ID")
    metric_type: MetricType = Field(..., description="Metric type")
    value: float = Field(..., description="Metric value")
    session_id: Optional[str] = Field(None, description="Session ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class AnalysisRequest(BaseModel):
    """Analysis request."""

    experiment_id: str = Field(..., description="Experiment ID")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance level")
    min_sample_size: int = Field(default=100, gt=0, description="Minimum sample size")


class AnalysisResponse(BaseModel):
    """Analysis response."""

    status: str
    has_winner: Optional[bool] = None
    winner_variant: Optional[str] = None
    confidence: Optional[float] = None
    should_launch: Optional[bool] = None
    recommendation: Optional[str] = None
    statistical_result: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None