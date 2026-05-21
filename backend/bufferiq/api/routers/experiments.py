"""
Experiments API router.

REST API endpoints for experiment management.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from bufferiq.api.models.experiments import (
    ExperimentCreate,
    ExperimentResponse,
    AssignmentRequest,
    AssignmentResponse,
    MetricTrackRequest,
    AnalysisRequest,
    AnalysisResponse,
)
from bufferiq.api.services.experiment_service import ExperimentService
from bufferiq.api.dependencies.experiments import get_experiment_service

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])


@router.post("/create", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    request: ExperimentCreate,
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    Create new experiment.

    Args:
        request: Experiment creation request
        service: Experiment service

    Returns:
        Created experiment

    Raises:
        HTTPException: If platform not supported
    """
    try:
        # Convert variants
        from bufferiq.ml.experiments.design.designer import Variant

        variants = [
            Variant(
                id=v.id,
                name=v.name,
                description=v.description,
                traffic_allocation=v.traffic_allocation,
                changes=v.changes,
                is_control=v.is_control,
            )
            for v in request.variants
        ]

        # Create experiment
        config = await service.create_experiment(
            name=request.name,
            description=request.description,
            variants=variants,
            platform=request.platform,
            primary_metric=request.primary_metric,
            baseline_rate=request.baseline_rate,
            mde=request.mde,
            alpha=request.alpha,
            power=request.power,
            expected_daily_traffic=request.expected_daily_traffic,
            enable_sequential_testing=request.enable_sequential_testing,
            enable_early_stopping=request.enable_early_stopping,
        )

        return ExperimentResponse(
            experiment_id=config.experiment_id,
            name=config.name,
            description=config.description,
            type=config.type.value,
            platform=config.platform,
            primary_metric=config.primary_metric.value,
            num_variants=len(config.variants),
            required_sample_size=config.required_sample_size,
            estimated_duration_days=config.estimated_duration_days,
            created_at=config.created_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/assign", response_model=AssignmentResponse)
async def assign_user(
    request: AssignmentRequest,
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    Assign user to variant.

    Args:
        request: Assignment request
        service: Experiment service

    Returns:
        Assignment result

    Raises:
        HTTPException: If experiment not found or platform not supported
    """
    try:
        assignment = await service.assign_user(
            experiment_id=request.experiment_id,
            user_id=request.user_id,
            session_id=request.session_id,
            platform=request.platform,
        )

        return AssignmentResponse(
            experiment_id=assignment.experiment_id,
            user_id=assignment.user_id,
            variant_id=assignment.variant_id,
            variant_name=assignment.variant_name,
            assigned_at=assignment.assigned_at,
            is_new_assignment=assignment.is_new_assignment,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/track", status_code=status.HTTP_204_NO_CONTENT)
async def track_metric(
    request: MetricTrackRequest,
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    Track metric event.

    Args:
        request: Metric track request
        service: Experiment service

    Raises:
        HTTPException: If experiment not found
    """
    try:
        await service.track_metric(
            experiment_id=request.experiment_id,
            user_id=request.user_id,
            metric_type=request.metric_type,
            value=request.value,
            session_id=request.session_id,
            metadata=request.metadata,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{experiment_id}/results", response_model=AnalysisResponse)
async def get_results(
    experiment_id: str,
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    Get experiment results.

    Args:
        experiment_id: Experiment ID
        service: Experiment service

    Returns:
        Analysis results

    Raises:
        HTTPException: If experiment not found
    """
    try:
        results = await service.analyze_experiment(experiment_id=experiment_id)
        return AnalysisResponse(**results)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{experiment_id}/analyze", response_model=AnalysisResponse)
async def analyze_experiment(
    experiment_id: str,
    request: AnalysisRequest,
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    Analyze experiment.

    Args:
        experiment_id: Experiment ID
        request: Analysis request
        service: Experiment service

    Returns:
        Analysis results

    Raises:
        HTTPException: If experiment not found
    """
    try:
        results = await service.analyze_experiment(
            experiment_id=experiment_id,
            alpha=request.alpha,
            min_sample_size=request.min_sample_size,
        )
        return AnalysisResponse(**results)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/{experiment_id}/stop", status_code=status.HTTP_204_NO_CONTENT)
async def stop_experiment(
    experiment_id: str,
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    Stop experiment.

    Args:
        experiment_id: Experiment ID
        service: Experiment service

    Raises:
        HTTPException: If experiment not found
    """
    try:
        # Implementation: mark experiment as stopped
        pass

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/list")
async def list_experiments(
    service: ExperimentService = Depends(get_experiment_service),
):
    """
    List all experiments.

    Args:
        service: Experiment service

    Returns:
        List of experiments
    """
    experiments = service.list_experiments()

    return [
        {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "platform": exp.platform,
            "created_at": exp.created_at.isoformat(),
        }
        for exp in experiments
    ]