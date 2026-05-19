"""
Hashtag API router.

Endpoints for hashtag analysis and recommendations.
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from bufferiq.api.models.hashtags import (
    HashtagAnalyzeRequest,
    HashtagAnalyzeResponse,
    HashtagRecommendRequest,
    HashtagRecommendResponse,
    TrendingHashtagRequest,
    TrendingHashtagResponse,
    TrendingHashtagItem,
    HashtagDiscoverRequest,
    HashtagDiscoverResponse,
    DiscoveredHashtagItem,
    HashtagValidateRequest,
    HashtagValidateResponse,
    HashtagValidationItem,
    HashtagInsightsResponse,
)
from bufferiq.api.dependencies.hashtags import get_hashtag_service
from bufferiq.ml.hashtags.intelligence.service import HashtagIntelligenceService
from bufferiq.core.database import get_db

router = APIRouter(prefix="/api/v1/hashtags", tags=["hashtags"])


@router.post("/analyze", response_model=HashtagAnalyzeResponse)
async def analyze_hashtag(
    request: HashtagAnalyzeRequest,
    service: HashtagIntelligenceService = Depends(get_hashtag_service),
) -> HashtagAnalyzeResponse:
    """
    Analyze hashtag performance and characteristics.

    Returns:
    - Performance metrics (engagement, lift, ROI)
    - Risk assessment
    - Related hashtags
    """
    try:
        analysis = await service.analyze_hashtag(
            hashtag=request.hashtag,
            platform=request.platform,
            user_id=request.user_id,
        )

        return HashtagAnalyzeResponse(**analysis)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/recommend", response_model=HashtagRecommendResponse)
async def recommend_hashtags(
    request: HashtagRecommendRequest,
    service: HashtagIntelligenceService = Depends(get_hashtag_service),
) -> HashtagRecommendResponse:
    """
    Get hashtag recommendations for content.

    Returns optimized hashtag suggestions based on:
    - Content topic
    - Platform best practices
    - Performance data
    """
    try:
        recommendations = await service.recommend_hashtags(
            content=request.content,
            platform=request.platform,
            user_id=request.user_id,
            count=request.count,
        )

        return HashtagRecommendResponse(
            hashtags=recommendations,
            platform=request.platform,
            count=len(recommendations),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Recommendation failed: {str(e)}"
        )


@router.post("/trends", response_model=TrendingHashtagResponse)
async def get_trending(
    request: TrendingHashtagRequest,
    service: HashtagIntelligenceService = Depends(get_hashtag_service),
) -> TrendingHashtagResponse:
    """
    Get currently trending hashtags.

    Returns hashtags that are:
    - Gaining momentum
    - High engagement
    - Relevant to category (if specified)
    """
    try:
        trending = await service.get_trending(
            platform=request.platform,
            category=request.category,
            limit=request.limit,
        )

        items = [
            TrendingHashtagItem(
                hashtag=t.hashtag,
                stage=t.stage.value,
                momentum_score=t.momentum_score,
                current_volume=t.current_volume,
                volume_change=t.volume_change,
                recommendation=t.recommendation,
            )
            for t in trending
        ]

        return TrendingHashtagResponse(
            platform=request.platform,
            trending=items,
            count=len(items),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend detection failed: {str(e)}")


@router.post("/discover", response_model=HashtagDiscoverResponse)
async def discover_hashtags(
    request: HashtagDiscoverRequest,
    service: HashtagIntelligenceService = Depends(get_hashtag_service),
) -> HashtagDiscoverResponse:
    """
    Discover related and niche hashtags.

    Returns:
    - Synonyms
    - Related hashtags
    - Complementary hashtags
    - Niche opportunities
    """
    try:
        discovery = await service.discovery_engine.discover(
            seed_hashtag=request.seed_hashtag,
            platform=request.platform,
            include_trending=False,
            max_results=request.max_results,
        )

        def to_items(hashtags: List) -> List[DiscoveredHashtagItem]:
            return [
                DiscoveredHashtagItem(
                    hashtag=h.hashtag,
                    similarity_score=h.similarity_score,
                    relationship_type=h.relationship_type,
                    effectiveness_score=h.effectiveness_score,
                )
                for h in hashtags
            ]

        return HashtagDiscoverResponse(
            seed_hashtag=request.seed_hashtag,
            platform=request.platform,
            synonyms=to_items(discovery.synonyms),
            related=to_items(discovery.related),
            complementary=to_items(discovery.complementary),
            niche=to_items(discovery.niche_hashtags),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.post("/validate", response_model=HashtagValidateResponse)
async def validate_hashtags(
    request: HashtagValidateRequest,
    service: HashtagIntelligenceService = Depends(get_hashtag_service),
) -> HashtagValidateResponse:
    """
    Validate hashtag safety and brand compliance.

    Checks for:
    - Banned hashtags
    - Spam patterns
    - Controversial content
    - Brand safety issues
    """
    try:
        validation = await service.validate_hashtags(
            hashtags=request.hashtags,
            platform=request.platform,
        )

        results = {
            hashtag: HashtagValidationItem(
                hashtag=hashtag,
                risk_level=risk.risk_level,
                is_safe=risk.risk_level in ["none", "low"],
                recommendation=risk.recommendation,
                reasons=risk.risk_reasons,
            )
            for hashtag, risk in validation.items()
        }

        return HashtagValidateResponse(
            platform=request.platform,
            results=results,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/insights/{hashtag}", response_model=HashtagInsightsResponse)
async def get_hashtag_insights(
    hashtag: str,
    platform: str,
    service: HashtagIntelligenceService = Depends(get_hashtag_service),
) -> HashtagInsightsResponse:
    """
    Get comprehensive insights for a hashtag.

    Includes:
    - Performance metrics
    - Risk assessment
    - Lifecycle stage
    - Related hashtags
    - Opportunities
    """
    try:
        # Get comprehensive analysis
        analysis = await service.analyze_hashtag(
            hashtag=hashtag,
            platform=platform,
        )

        # Mock lifecycle data (would come from lifecycle tracker)
        lifecycle_stage = "mature"
        age_days = 180

        # Mock opportunities
        opportunities = [
            "High engagement on weekdays",
            "Pair with #innovation for 15% lift",
            "Peak usage at 10 AM EST",
        ]

        return HashtagInsightsResponse(
            hashtag=hashtag,
            platform=platform,
            performance=analysis["performance"],
            risk=analysis["risk"],
            lifecycle_stage=lifecycle_stage,
            age_days=age_days,
            related_hashtags=[h["hashtag"] for h in analysis["related"]["synonyms"]],
            opportunities=opportunities,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get insights: {str(e)}"
        )