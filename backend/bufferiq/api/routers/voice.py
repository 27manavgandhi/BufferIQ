"""
Voice analysis API router.

Endpoints for voice profile management and analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import logging
from datetime import datetime

from bufferiq.api.models.voice import (
    VoiceExtractionRequest,
    VoiceAnalysisRequest,
    BatchAnalysisRequest,
    VoiceValidationRequest,
    DriftDetectionRequest,
    VoiceAnalysisResponse,
    BatchAnalysisResponse,
    VoiceProfileResponse,
    DriftAlertResponse,
    ValidationResponse,
    ErrorResponse,
    ConsistencyScoreResponse,
    RecommendationResponse,
)
from bufferiq.api.dependencies.voice import get_voice_service
from bufferiq.ml.voice.intelligence.service import VoiceIntelligenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


@router.post(
    "/extract",
    response_model=VoiceProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract voice profile from historical content",
)
async def extract_voice_profile(
    request: VoiceExtractionRequest,
    service: VoiceIntelligenceService = Depends(get_voice_service),
) -> VoiceProfileResponse:
    """
    Extract and build voice profile from historical posts.
    
    Analyzes past content to create comprehensive brand voice profile
    with linguistic, syntactic, and stylistic characteristics.
    
    Args:
        request: Voice extraction request
        service: Voice intelligence service
    
    Returns:
        Voice profile
    
    Raises:
        HTTPException: If extraction fails
    """
    try:
        logger.info(
            f"Extracting voice profile: brand={request.brand_id}, "
            f"platform={request.platform}"
        )
        
        profile = await service.build_voice_profile(
            brand_id=request.brand_id,
            platform=request.platform,
            lookback_days=request.lookback_days,
        )
        
        return VoiceProfileResponse(
            profile_id=profile.profile_id,
            brand_id=profile.brand_id,
            version=profile.version,
            created_at=profile.created_at.isoformat(),
            confidence=profile.confidence,
            sample_size=profile.sample_size,
            signature=profile.signature,
            platform_profiles=profile.platform_profiles,
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Voice extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice extraction failed: {str(e)}",
        )


@router.post(
    "/analyze",
    response_model=VoiceAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze content voice alignment",
)
async def analyze_content_voice(
    request: VoiceAnalysisRequest,
    service: VoiceIntelligenceService = Depends(get_voice_service),
) -> VoiceAnalysisResponse:
    """
    Analyze content for voice consistency with brand profile.
    
    Provides comprehensive analysis including:
    - Consistency scores (lexical, syntactic, stylistic)
    - Similarity metrics
    - Recommendations for improvement
    - Validation results
    
    Args:
        request: Voice analysis request
        service: Voice intelligence service
    
    Returns:
        Voice analysis results
    
    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(
            f"Analyzing content: brand={request.brand_id}, "
            f"platform={request.platform}"
        )
        
        analysis = await service.analyze_content(
            text=request.text,
            brand_id=request.brand_id,
            platform=request.platform,
            return_recommendations=request.return_recommendations,
            return_validation=request.return_validation,
        )
        
        # Convert to response model
        response_data = {
            "text": analysis["text"],
            "brand_id": analysis["brand_id"],
            "platform": analysis["platform"],
            "profile_id": analysis["profile_id"],
            "consistency_score": ConsistencyScoreResponse(**analysis["consistency_score"]),
            "metrics": analysis["metrics"],
            "analyzed_at": analysis["analyzed_at"],
        }
        
        # Add recommendations if present
        if "recommendations" in analysis:
            response_data["recommendations"] = [
                RecommendationResponse(**rec) for rec in analysis["recommendations"]
            ]
        
        # Add validation if present
        if "validation" in analysis:
            response_data["validation"] = ValidationResponse(**analysis["validation"])
        
        return VoiceAnalysisResponse(**response_data)
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Voice analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice analysis failed: {str(e)}",
        )


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze multiple content pieces",
)
async def analyze_batch_content(
    request: BatchAnalysisRequest,
    service: VoiceIntelligenceService = Depends(get_voice_service),
) -> BatchAnalysisResponse:
    """
    Analyze multiple pieces of content in batch.
    
    Args:
        request: Batch analysis request
        service: Voice intelligence service
    
    Returns:
        Batch analysis results
    
    Raises:
        HTTPException: If batch analysis fails
    """
    try:
        logger.info(
            f"Batch analyzing {len(request.contents)} items: "
            f"brand={request.brand_id}, platform={request.platform}"
        )
        
        results = await service.analyze_batch(
            contents=request.contents,
            brand_id=request.brand_id,
            platform=request.platform,
        )
        
        # Convert to response models
        analysis_responses = []
        successful = 0
        failed = 0
        
        for result in results:
            if "error" in result:
                failed += 1
                continue
            
            successful += 1
            response_data = {
                "text": result["text"],
                "brand_id": result["brand_id"],
                "platform": result["platform"],
                "profile_id": result["profile_id"],
                "consistency_score": ConsistencyScoreResponse(**result["consistency_score"]),
                "metrics": result["metrics"],
                "analyzed_at": result["analyzed_at"],
            }
            
            if "recommendations" in result:
                response_data["recommendations"] = [
                    RecommendationResponse(**rec) for rec in result["recommendations"]
                ]
            
            analysis_responses.append(VoiceAnalysisResponse(**response_data))
        
        return BatchAnalysisResponse(
            results=analysis_responses,
            total_analyzed=len(request.contents),
            successful=successful,
            failed=failed,
        )
    
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}",
        )


@router.post(
    "/validate",
    response_model=ValidationResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate content against brand voice",
)
async def validate_content(
    request: VoiceValidationRequest,
    service: VoiceIntelligenceService = Depends(get_voice_service),
) -> ValidationResponse:
    """
    Validate content against brand voice profile.
    
    Performs pre-publish validation to ensure content
    meets voice consistency standards.
    
    Args:
        request: Validation request
        service: Voice intelligence service
    
    Returns:
        Validation result
    
    Raises:
        HTTPException: If validation fails
    """
    try:
        logger.info(
            f"Validating content: brand={request.brand_id}, "
            f"platform={request.platform}"
        )
        
        analysis = await service.analyze_content(
            text=request.text,
            brand_id=request.brand_id,
            platform=request.platform,
            return_recommendations=True,
            return_validation=True,
        )
        
        if "validation" not in analysis:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Validation not performed",
            )
        
        return ValidationResponse(**analysis["validation"])
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}",
        )


@router.post(
    "/drift",
    response_model=DriftAlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Detect voice drift",
)
async def detect_voice_drift(
    request: DriftDetectionRequest,
    service: VoiceIntelligenceService = Depends(get_voice_service),
) -> DriftAlertResponse:
    """
    Detect statistical drift in brand voice.
    
    Analyzes recent content to identify deviations
    from established brand voice profile.
    
    Args:
        request: Drift detection request
        service: Voice intelligence service
    
    Returns:
        Drift alert
    
    Raises:
        HTTPException: If drift detection fails
    """
    try:
        logger.info(
            f"Detecting drift: brand={request.brand_id}, "
            f"platform={request.platform}"
        )
        
        drift_result = await service.detect_drift(
            brand_id=request.brand_id,
            platform=request.platform,
            window_days=request.window_days,
        )
        
        return DriftAlertResponse(**drift_result)
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}",
        )


@router.get(
    "/profile/{brand_id}/{platform}",
    response_model=VoiceProfileResponse,
    status_code=status.HTTP_200_OK,
    summary="Get voice profile",
)
async def get_voice_profile(
    brand_id: str,
    platform: str,
    service: VoiceIntelligenceService = Depends(get_voice_service),
) -> VoiceProfileResponse:
    """
    Get existing voice profile for a brand.
    
    Args:
        brand_id: Brand identifier
        platform: Platform (linkedin/twitter/bluesky)
        service: Voice intelligence service
    
    Returns:
        Voice profile
    
    Raises:
        HTTPException: If profile not found
    """
    try:
        logger.info(f"Getting profile: brand={brand_id}, platform={platform}")
        
        profile = service.get_cached_profile(brand_id, platform)
        
        if profile is None:
            # Try to build it
            profile = await service.build_voice_profile(brand_id, platform)
        
        return VoiceProfileResponse(
            profile_id=profile.profile_id,
            brand_id=profile.brand_id,
            version=profile.version,
            created_at=profile.created_at.isoformat(),
            confidence=profile.confidence,
            sample_size=profile.sample_size,
            signature=profile.signature,
            platform_profiles=profile.platform_profiles,
        )
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile: {str(e)}",
        )