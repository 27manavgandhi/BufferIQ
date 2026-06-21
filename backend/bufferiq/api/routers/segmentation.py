"""API router for segmentation endpoints."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from bufferiq.api.models.segmentation import (
    SegmentAudienceRequest,
    SegmentationResponse,
    PersonaResponse,
    RecommendationResponse,
)
from bufferiq.api.dependencies.segmentation import (
    get_segmentation_service,
    get_intelligence_service,
)
from bufferiq.database import get_db
from bufferiq.ml.segmentation.types import AudienceDataPoint, SUPPORTED_PLATFORMS
from bufferiq.ml.segmentation.exceptions import (
    UnsupportedPlatformError,
    InsufficientDataError,
)

router = APIRouter(prefix="/api/v1/segmentation", tags=["segmentation"])


@router.post("/segment", response_model=SegmentationResponse)
async def segment_audience(
    request: SegmentAudienceRequest,
    service=Depends(get_segmentation_service),
):
    """
    Segment audience into clusters with personas.

    Platform must be one of: linkedin, twitter, bluesky
    Requires minimum 10 audience members.
    """
    if request.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{request.platform}' not supported. "
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}",
        )

    try:
        # Convert request models to domain models
        audience_data = [
            AudienceDataPoint(
                user_id=item.user_id,
                platform=item.platform,
                follower_count=item.follower_count,
                following_count=item.following_count,
                post_count=item.post_count,
                avg_engagement_rate=item.avg_engagement_rate,
                engagement_history=item.engagement_history,
                interaction_types=item.interaction_types,
                active_hours=item.active_hours,
                active_days=item.active_days,
                topics_engaged=item.topics_engaged,
                content_types_engaged=item.content_types_engaged,
                account_age_days=item.account_age_days,
                verified=item.verified,
                bio_keywords=item.bio_keywords,
                location=item.location,
                language=item.language,
            )
            for item in request.audience_data
        ]

        result = await service.segment_audience(
            audience_data=audience_data,
            platform=request.platform,
            historical_snapshots=request.historical_snapshots,
        )

        return SegmentationResponse(**result)

    except UnsupportedPlatformError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InsufficientDataError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.get("/platforms")
async def get_supported_platforms():
    """Get list of supported platforms."""
    return SUPPORTED_PLATFORMS


@router.get("/personas/{platform}", response_model=List[PersonaResponse])
async def get_personas(
    platform: str,
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    """Get all personas for a platform."""
    if platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. "
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}",
        )

    try:
        from bufferiq.domain.repositories.segment_repository import PersonaRepository

        repo = PersonaRepository(db)
        personas = repo.get_personas_by_platform(platform)

        return [PersonaResponse(**p.to_dict()) for p in personas]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations/{segment_id}", response_model=RecommendationResponse)
async def get_recommendations(
    segment_id: str,
    platform: str = Query(...),
    service=Depends(get_segmentation_service),
):
    """Get content recommendations for a specific segment."""
    if platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. "
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}",
        )

    try:
        rec = await service.get_recommendations(segment_id, platform)
        return RecommendationResponse(**rec)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evolution/{segment_id}")
async def get_segment_evolution(
    segment_id: str,
    platform: str = Query(...),
    days: int = Query(default=30, ge=7, le=365),
    db: Session = Depends(get_db),
):
    """Get segment evolution over time."""
    if platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. "
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}",
        )

    try:
        from bufferiq.domain.repositories.segment_repository import EvolutionRepository

        repo = EvolutionRepository(db)
        history = repo.get_segment_history(segment_id, limit=days)

        return {
            "segment_id": segment_id,
            "platform": platform,
            "history": [r.to_dict() for r in history],
            "days": len(history),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/segments/{platform}")
async def get_segments(
    platform: str,
    db: Session = Depends(get_db),
):
    """Get all segments for a platform."""
    if platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. "
            f"Supported: {', '.join(SUPPORTED_PLATFORMS)}",
        )

    try:
        from bufferiq.domain.repositories.segment_repository import SegmentRepository

        repo = SegmentRepository(db)
        segments = repo.get_segments_by_platform(platform)

        return {"platform": platform, "segments": [s.to_dict() for s in segments]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))