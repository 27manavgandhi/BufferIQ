"""
Content analysis API endpoints.

Provides REST API for content intelligence features.
"""

from typing import List

from fastapi import APIRouter, HTTPException, status

from bufferiq.api.models.content import (
    ContentAnalysisRequest,
    ContentAnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
)
from bufferiq.ml.content.intelligence.service import ContentIntelligenceService

router = APIRouter(prefix="/content", tags=["content"])

# Initialize service
content_service = ContentIntelligenceService()


@router.post(
    "/analyze",
    response_model=ContentAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_content(request: ContentAnalysisRequest) -> ContentAnalysisResponse:
    """
    Analyze content.

    Performs comprehensive content analysis including:
    - Text preprocessing
    - Sentiment analysis
    - Readability scoring
    - Quality checking
    - Optimization suggestions

    Args:
        request: Content analysis request

    Returns:
        Complete analysis results

    Raises:
        HTTPException: If analysis fails
    """
    try:
        result = content_service.analyze_content(
            text=request.text,
            platform=request.platform,
            user_id=request.user_id,
            include_optimization=request.include_optimization,
        )

        return ContentAnalysisResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
)
async def batch_analyze(request: BatchAnalysisRequest) -> BatchAnalysisResponse:
    """
    Batch analyze multiple posts.

    Analyzes multiple posts in a single request.

    Args:
        request: Batch analysis request

    Returns:
        Batch analysis results

    Raises:
        HTTPException: If analysis fails
    """
    try:
        results = content_service.analyze_batch(
            posts=request.posts, platform=request.platform
        )

        # Count errors
        errors = sum(1 for r in results if "error" in r)

        # Convert to response models
        response_results = []
        for result in results:
            if "error" not in result:
                response_results.append(ContentAnalysisResponse(**result))

        return BatchAnalysisResponse(
            results=response_results,
            total_analyzed=len(results) - errors,
            total_errors=errors,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )


@router.post("/optimize", status_code=status.HTTP_200_OK)
async def optimize_content(request: ContentAnalysisRequest) -> dict:
    """
    Get content optimization suggestions.

    Provides actionable recommendations for improving content.

    Args:
        request: Content analysis request

    Returns:
        Optimization suggestions

    Raises:
        HTTPException: If optimization fails
    """
    try:
        # Analyze content first
        analysis = content_service.analyze_content(
            text=request.text,
            platform=request.platform,
            user_id=request.user_id,
            include_optimization=True,
        )

        # Return optimization section
        if "optimization" in analysis:
            return analysis["optimization"]
        else:
            return {"message": "No optimization suggestions available"}

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}",
        )
