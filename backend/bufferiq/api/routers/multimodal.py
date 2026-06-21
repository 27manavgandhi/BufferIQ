"""Multi-modal analysis API router."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from typing import List, Optional

from bufferiq.api.models.multimodal import (
    ImageAnalysisRequest,
    VideoAnalysisRequest,
    LinkAnalysisRequest,
    MultiModalAnalysisRequest,
    MultiModalAnalysisResponse,
)
from bufferiq.ml.multimodal.intelligence.service import MultiModalIntelligenceService
from bufferiq.ml.multimodal.types import SUPPORTED_PLATFORMS
from bufferiq.ml.multimodal.exceptions import UnsupportedPlatformError

router = APIRouter(prefix="/api/v1/multimodal", tags=["multimodal"])


def get_multimodal_service() -> MultiModalIntelligenceService:
    """Dependency to get multi-modal service."""
    return MultiModalIntelligenceService()


@router.post("/analyze/image", response_model=MultiModalAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    platform: str = Query(..., description="Platform type (linkedin, twitter, or bluesky)"),
    service: MultiModalIntelligenceService = Depends(get_multimodal_service)
):
    """
    Analyze uploaded image.
    
    Platform must be one of: linkedin, twitter, bluesky
    
    Args:
        image: Uploaded image file
        platform: Platform type
        service: Multi-modal service
        
    Returns:
        Image analysis results
        
    Raises:
        HTTPException: If platform not supported or analysis fails
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{platform}' not supported. Supported: {', '.join(SUPPORTED_PLATFORMS)}"
        )
    
    try:
        # Read image bytes
        image_bytes = await image.read()
        
        # Analyze image
        result = await service.image_analyzer.analyze(
            image_bytes,
            platform  # type: ignore
        )
        
        return MultiModalAnalysisResponse(
            analysis_type="image",
            platform=platform,
            results=result.to_dict(),
            processing_time_ms=result.processing_time_ms
        )
        
    except UnsupportedPlatformError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")


@router.post("/analyze/video", response_model=MultiModalAnalysisResponse)
async def analyze_video(
    request: VideoAnalysisRequest,
    service: MultiModalIntelligenceService = Depends(get_multimodal_service)
):
    """
    Analyze video from URL.
    
    Platform must be one of: linkedin, twitter, bluesky
    
    Args:
        request: Video analysis request
        service: Multi-modal service
        
    Returns:
        Video analysis results
        
    Raises:
        HTTPException: If platform not supported or analysis fails
    """
    if request.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{request.platform}' not supported. Supported: {', '.join(SUPPORTED_PLATFORMS)}"
        )
    
    try:
        result = await service.video_analyzer.analyze(
            str(request.video_url),
            request.platform,
            extract_keyframes=request.extract_keyframes,
            detect_scenes=request.detect_scenes,
            analyze_audio=request.analyze_audio
        )
        
        return MultiModalAnalysisResponse(
            analysis_type="video",
            platform=request.platform,
            results=result.to_dict(),
            processing_time_ms=result.processing_time_ms
        )
        
    except UnsupportedPlatformError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@router.post("/analyze/link", response_model=MultiModalAnalysisResponse)
async def analyze_link(
    request: LinkAnalysisRequest,
    service: MultiModalIntelligenceService = Depends(get_multimodal_service)
):
    """
    Analyze link preview.
    
    Platform must be one of: linkedin, twitter, bluesky
    
    Args:
        request: Link analysis request
        service: Multi-modal service
        
    Returns:
        Link preview analysis results
        
    Raises:
        HTTPException: If platform not supported or analysis fails
    """
    if request.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{request.platform}' not supported. Supported: {', '.join(SUPPORTED_PLATFORMS)}"
        )
    
    try:
        result = await service.link_analyzer.analyze(
            str(request.url),
            request.platform
        )
        
        return MultiModalAnalysisResponse(
            analysis_type="link",
            platform=request.platform,
            results=result.to_dict(),
            processing_time_ms=result.processing_time_ms
        )
        
    except UnsupportedPlatformError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Link analysis failed: {str(e)}")


@router.post("/analyze/post", response_model=MultiModalAnalysisResponse)
async def analyze_post(
    request: MultiModalAnalysisRequest,
    service: MultiModalIntelligenceService = Depends(get_multimodal_service)
):
    """
    Analyze complete post with all media.
    
    Platform must be one of: linkedin, twitter, bluesky
    
    Args:
        request: Multi-modal analysis request
        service: Multi-modal service
        
    Returns:
        Complete multi-modal analysis results
        
    Raises:
        HTTPException: If platform not supported or analysis fails
    """
    if request.platform not in SUPPORTED_PLATFORMS:
        raise HTTPException(
            status_code=400,
            detail=f"Platform '{request.platform}' not supported. Supported: {', '.join(SUPPORTED_PLATFORMS)}"
        )
    
    try:
        # Convert URLs to strings
        image_urls = [str(url) for url in request.image_urls] if request.image_urls else None
        video_urls = [str(url) for url in request.video_urls] if request.video_urls else None
        link_urls = [str(url) for url in request.link_urls] if request.link_urls else None
        
        result = await service.analyze_post(
            post_id=request.post_id,
            text=request.text,
            image_urls=image_urls,
            video_urls=video_urls,
            link_urls=link_urls,
            platform=request.platform
        )
        
        return MultiModalAnalysisResponse(
            analysis_type="multi_modal",
            platform=request.platform,
            results=result,
            processing_time_ms=result.get("processing_time_ms")
        )
        
    except UnsupportedPlatformError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-modal analysis failed: {str(e)}")


@router.get("/platforms", response_model=List[str])
async def get_supported_platforms():
    """
    Get list of supported platforms.
    
    Returns:
        List of supported platform names
    """
    return SUPPORTED_PLATFORMS