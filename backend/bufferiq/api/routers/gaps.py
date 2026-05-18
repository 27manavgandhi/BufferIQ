"""API router for gap analysis endpoints."""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from bufferiq.api.models.gaps import (
    GapAnalysisRequest,
    GapAnalysisResponse,
    CalendarRequest,
    CalendarResponse,
    RecommendationsRequest,
    CompetitorAnalysisRequest,
    BatchAnalysisRequest,
)
from bufferiq.api.dependencies.gaps import get_gap_service
from bufferiq.api.services.gap_service import GapService

router = APIRouter(prefix="/api/v1/gaps", tags=["gaps"])


@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_gaps(
    request: GapAnalysisRequest,
    gap_service: GapService = Depends(get_gap_service),
) -> Dict[str, Any]:
    """
    Analyze content gaps.

    Performs comprehensive gap analysis including:
    - Topic extraction
    - Coverage analysis
    - Gap detection
    - Competitor analysis (if provided)
    - Content recommendations

    Args:
        request: Gap analysis request
        gap_service: Gap service dependency

    Returns:
        Gap analysis report

    Raises:
        HTTPException: If analysis fails
    """
    try:
        report = await gap_service.analyze_gaps(
            user_id=request.user_id,
            platform=request.platform,
            competitor_ids=request.competitor_ids,
            industry=request.industry,
            lookback_days=request.lookback_days,
            include_recommendations=request.include_recommendations,
        )

        return report

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gap analysis failed: {str(e)}",
        )


@router.post("/recommendations", response_model=Dict[str, Any])
async def get_recommendations(
    request: RecommendationsRequest,
    gap_service: GapService = Depends(get_gap_service),
) -> Dict[str, Any]:
    """
    Get content recommendations.

    Args:
        request: Recommendations request
        gap_service: Gap service dependency

    Returns:
        Content recommendations

    Raises:
        HTTPException: If generation fails
    """
    try:
        recommendations = await gap_service.get_recommendations(
            user_id=request.user_id,
            platform=request.platform,
            count=request.count,
        )

        return {
            "user_id": request.user_id,
            "platform": request.platform,
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation generation failed: {str(e)}",
        )


@router.post("/calendar", response_model=Dict[str, Any])
async def generate_calendar(
    request: CalendarRequest,
    gap_service: GapService = Depends(get_gap_service),
) -> Dict[str, Any]:
    """
    Generate content calendar.

    Args:
        request: Calendar request
        gap_service: Gap service dependency

    Returns:
        Content calendar

    Raises:
        HTTPException: If generation fails
    """
    try:
        calendar = await gap_service.generate_calendar(
            user_id=request.user_id,
            platform=request.platform,
            weeks=request.weeks,
            posts_per_week=request.posts_per_week,
            start_date=request.start_date,
        )

        return calendar

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Calendar generation failed: {str(e)}",
        )


@router.post("/competitors", response_model=Dict[str, Any])
async def analyze_competitors(
    request: CompetitorAnalysisRequest,
    gap_service: GapService = Depends(get_gap_service),
) -> Dict[str, Any]:
    """
    Analyze competitor strategy.

    Args:
        request: Competitor analysis request
        gap_service: Gap service dependency

    Returns:
        Competitive analysis

    Raises:
        HTTPException: If analysis fails
    """
    try:
        analysis = await gap_service.benchmark_competitors(
            user_id=request.user_id,
            competitor_ids=request.competitor_ids,
            platform=request.platform,
        )

        return analysis

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Competitor analysis failed: {str(e)}",
        )


@router.get("/report/{brand_id}", response_model=Dict[str, Any])
async def get_gap_report(
    brand_id: str,
    platform: str = "linkedin",
    gap_service: GapService = Depends(get_gap_service),
) -> Dict[str, Any]:
    """
    Get comprehensive gap report.

    Args:
        brand_id: Brand identifier
        platform: Platform (default: linkedin)
        gap_service: Gap service dependency

    Returns:
        Comprehensive gap report

    Raises:
        HTTPException: If report generation fails
    """
    try:
        # Validate platform
        if platform not in ["linkedin", "twitter", "bluesky"]:
            raise ValueError(f"Invalid platform: {platform}")

        report = await gap_service.get_quick_insights(
            user_id=brand_id,
            platform=platform,
        )

        return report

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )


@router.post("/batch", response_model=Dict[str, Any])
async def batch_analyze(
    request: BatchAnalysisRequest,
    gap_service: GapService = Depends(get_gap_service),
) -> Dict[str, Any]:
    """
    Batch analyze multiple users.

    Args:
        request: Batch analysis request
        gap_service: Gap service dependency

    Returns:
        Batch analysis results

    Raises:
        HTTPException: If batch analysis fails
    """
    try:
        results = []

        for user_id in request.user_ids:
            try:
                report = await gap_service.analyze_gaps(
                    user_id=user_id,
                    platform=request.platform,
                    lookback_days=request.lookback_days,
                    include_recommendations=False,  # Skip for batch
                )

                results.append({
                    "user_id": user_id,
                    "status": "success",
                    "coverage_score": report["coverage_score"],
                    "total_gaps": report["total_gaps"],
                })

            except Exception as e:
                results.append({
                    "user_id": user_id,
                    "status": "error",
                    "error": str(e),
                })

        return {
            "total_users": len(request.user_ids),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )