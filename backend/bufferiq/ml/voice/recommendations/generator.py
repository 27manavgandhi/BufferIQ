"""
Voice-aligned recommendation generation.

Generates specific suggestions to align content
with brand voice profile.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile
from bufferiq.ml.voice.consistency.scorer import ConsistencyScore

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class VoiceRecommendation:
    """Voice alignment recommendation."""
    
    type: str  # "vocabulary", "style", "tone", "structure"
    priority: str  # "high", "medium", "low"
    current_value: str
    suggested_value: str
    reason: str
    impact_score: float  # Expected consistency improvement
    examples: List[str]


@dataclass
class VoiceOptimizationResult:
    """Content optimization for voice alignment."""
    
    original_text: str
    original_score: float
    
    optimized_text: str
    optimized_score: float
    improvement: float
    
    recommendations: List[VoiceRecommendation]
    rewrite_suggestions: List[str]
    
    changes_made: Dict[str, List[str]]


class VoiceRecommendationEngine:
    """
    Generate voice-aligned content recommendations.
    
    Provides specific suggestions to align content
    with brand voice profile.
    
    Example:
```python
        engine = VoiceRecommendationEngine()
        result = engine.optimize(
            text="Check this out!",
            profile=brand_voice,
            platform="linkedin"
        )
        print(f"Improvement: +{result.improvement:.1f} points")
        print(f"Optimized: {result.optimized_text}")
        for rec in result.recommendations:
            print(f"  {rec.type}: {rec.reason}")
```
    """
    
    def __init__(self):
        """Initialize recommendation engine."""
        pass
    
    def generate_recommendations(
        self,
        consistency_score: ConsistencyScore,
        profile: VoiceProfile,
        platform: str,
    ) -> List[VoiceRecommendation]:
        """
        Generate recommendations from consistency score.
        
        Args:
            consistency_score: Consistency analysis
            profile: Voice profile
            platform: Target platform
        
        Returns:
            List of recommendations
        
        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        recommendations = []
        
        # Lexical recommendations
        if consistency_score.lexical_consistency < 75:
            recommendations.extend(
                self._generate_lexical_recommendations(
                    consistency_score, profile
                )
            )
        
        # Syntactic recommendations
        if consistency_score.syntactic_consistency < 75:
            recommendations.extend(
                self._generate_syntactic_recommendations(
                    consistency_score, profile
                )
            )
        
        # Stylistic recommendations
        if consistency_score.stylistic_consistency < 75:
            recommendations.extend(
                self._generate_stylistic_recommendations(
                    consistency_score, profile
                )
            )
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order[r.priority])
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
    
    def optimize(
        self,
        text: str,
        profile: VoiceProfile,
        platform: str,
        target_score: float = 85.0,
    ) -> VoiceOptimizationResult:
        """
        Optimize content for voice alignment.
        
        Args:
            text: Content to optimize
            profile: Brand voice profile
            platform: Target platform
            target_score: Target consistency score
        
        Returns:
            Optimization result with suggestions
        
        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        # For now, return basic optimization
        # In production, this would use more sophisticated rewriting
        
        from bufferiq.ml.voice.consistency.scorer import VoiceConsistencyScorer
        
        scorer = VoiceConsistencyScorer()
        original_score_obj = scorer.score(text, profile, platform)
        original_score = original_score_obj.overall_score
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            original_score_obj, profile, platform
        )
        
        # Apply simple optimizations
        optimized_text = text
        changes_made: Dict[str, List[str]] = {}
        
        # Example optimization: adjust formality
        if original_score_obj.stylistic_consistency < 75:
            if "formality" in original_score_obj.feature_deviations:
                # Simple transformation (in production, use NLP models)
                if profile.stylistic_fingerprint.get('formality_score', 50) > 70:
                    # Make more formal
                    optimized_text = optimized_text.replace("!", ".")
                    changes_made['punctuation'] = ['Reduced exclamation marks']
        
        # Re-score optimized text
        optimized_score_obj = scorer.score(optimized_text, profile, platform)
        optimized_score = optimized_score_obj.overall_score
        
        improvement = optimized_score - original_score
        
        # Generate rewrite suggestions
        rewrite_suggestions = self._generate_rewrite_suggestions(
            text, recommendations
        )
        
        return VoiceOptimizationResult(
            original_text=text,
            original_score=original_score,
            optimized_text=optimized_text,
            optimized_score=optimized_score,
            improvement=improvement,
            recommendations=recommendations,
            rewrite_suggestions=rewrite_suggestions,
            changes_made=changes_made,
        )
    
    def _generate_lexical_recommendations(
        self, score: ConsistencyScore, profile: VoiceProfile
    ) -> List[VoiceRecommendation]:
        """Generate lexical recommendations."""
        recommendations = []
        
        # Check vocabulary complexity
        if 'lexical_complexity' in score.feature_deviations:
            target_complexity = profile.lexical_fingerprint.get('complexity', 50)
            
            recommendations.append(VoiceRecommendation(
                type="vocabulary",
                priority="medium",
                current_value="Current complexity",
                suggested_value=f"Target complexity: {target_complexity:.1f}",
                reason="Adjust vocabulary complexity to match brand voice",
                impact_score=15.0,
                examples=[
                    "Use simpler words" if target_complexity < 50 else "Use more sophisticated vocabulary"
                ],
            ))
        
        return recommendations
    
    def _generate_syntactic_recommendations(
        self, score: ConsistencyScore, profile: VoiceProfile
    ) -> List[VoiceRecommendation]:
        """Generate syntactic recommendations."""
        recommendations = []
        
        # Check sentence length
        if 'syntactic_avg_sentence_length' in score.feature_deviations:
            target_length = profile.syntactic_fingerprint.get('avg_sentence_length', 15)
            
            recommendations.append(VoiceRecommendation(
                type="structure",
                priority="low",
                current_value="Current sentence structure",
                suggested_value=f"Target avg length: {target_length:.1f} words",
                reason="Adjust sentence length to match brand patterns",
                impact_score=10.0,
                examples=[
                    "Break long sentences" if target_length < 15 else "Combine short sentences"
                ],
            ))
        
        return recommendations
    
    def _generate_stylistic_recommendations(
        self, score: ConsistencyScore, profile: VoiceProfile
    ) -> List[VoiceRecommendation]:
        """Generate stylistic recommendations."""
        recommendations = []
        
        # Check formality
        if 'stylistic_formality_score' in score.feature_deviations:
            target_formality = profile.stylistic_fingerprint.get('formality_score', 50)
            deviation = score.feature_deviations['stylistic_formality_score']
            
            if deviation > 0:
                direction = "more casual" if target_formality < 50 else "more formal"
            else:
                direction = "more formal" if target_formality > 50 else "more casual"
            
            recommendations.append(VoiceRecommendation(
                type="tone",
                priority="high",
                current_value="Current tone",
                suggested_value=f"{direction} (target: {target_formality:.0f})",
                reason=f"Adjust formality to match brand voice",
                impact_score=20.0,
                examples=[
                    "Use contractions" if direction == "more casual" else "Avoid contractions",
                    "Add emojis" if direction == "more casual" else "Remove emojis",
                ],
            ))
        
        return recommendations
    
    def _generate_rewrite_suggestions(
        self, text: str, recommendations: List[VoiceRecommendation]
    ) -> List[str]:
        """Generate complete rewrite suggestions."""
        suggestions = []
        
        # Generate variations based on recommendations
        if any(r.type == "tone" and "casual" in r.suggested_value for r in recommendations):
            suggestions.append(f"More casual version: {text.replace('.', '!').replace('However', 'But')}")
        
        if any(r.type == "tone" and "formal" in r.suggested_value for r in recommendations):
            suggestions.append(f"More formal version: {text.replace('!', '.').replace('But', 'However')}")
        
        # Add generic suggestion
        suggestions.append(f"Alternative: Consider rephrasing to better match brand voice")
        
        return suggestions[:3]  # Return top 3