"""
Voice consistency scoring.

Measures alignment between content and voice profiles
using cosine similarity, KL divergence, and feature comparison.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile
from bufferiq.ml.voice.linguistic.lexical_analyzer import LexicalAnalyzer
from bufferiq.ml.voice.linguistic.syntactic_analyzer import SyntacticAnalyzer
from bufferiq.ml.voice.stylistic.style_detector import StyleDetector
from bufferiq.ml.voice.consistency.metrics import ConsistencyMetrics

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class ConsistencyScore:
    """Voice consistency measurement."""
    
    overall_score: float  # 0-100
    lexical_consistency: float  # 0-100
    syntactic_consistency: float  # 0-100
    stylistic_consistency: float  # 0-100
    
    # Detailed metrics
    cosine_similarity: float  # 0-1
    kl_divergence: float  # 0-∞, lower = more consistent
    feature_deviations: Dict[str, float]
    
    # Thresholds
    is_consistent: bool
    severity: str  # "none", "minor", "moderate", "severe"
    
    # Recommendations
    alignment_suggestions: List[str]


class VoiceConsistencyScorer:
    """
    Score voice consistency between content and profile.
    
    Measures alignment using multiple metrics including
    cosine similarity, KL divergence, and feature comparison.
    
    Example:
```python
        scorer = VoiceConsistencyScorer()
        score = scorer.score(
            content="Your new post here",
            profile=brand_voice_profile,
            platform="linkedin"
        )
        print(f"Consistency: {score.overall_score:.1f}/100")
        print(f"Is consistent: {score.is_consistent}")
        if not score.is_consistent:
            for suggestion in score.alignment_suggestions:
                print(f"  - {suggestion}")
```
    """
    
    def __init__(self, consistency_threshold: float = 75.0):
        """
        Initialize consistency scorer.
        
        Args:
            consistency_threshold: Minimum score for consistency (0-100)
        """
        self.threshold = consistency_threshold
        self.lexical_analyzer = LexicalAnalyzer()
        self.syntactic_analyzer = SyntacticAnalyzer()
        self.style_detector = StyleDetector()
        self.metrics = ConsistencyMetrics()
    
    def score(
        self,
        content: str,
        profile: VoiceProfile,
        platform: str,
    ) -> ConsistencyScore:
        """
        Score content consistency with voice profile.
        
        Args:
            content: Content to score
            profile: Brand voice profile
            platform: Target platform
        
        Returns:
            Consistency score and recommendations
        
        Raises:
            ValueError: If platform not supported or content too short
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        if not content or len(content.strip()) < 10:
            raise ValueError("Content too short for consistency scoring")
        
        # Analyze content
        try:
            lexical = self.lexical_analyzer.analyze(content)
            syntactic = self.syntactic_analyzer.analyze(content)
            stylistic = self.style_detector.detect(content)
        except ValueError as e:
            logger.warning(f"Analysis failed: {e}")
            raise ValueError(f"Content analysis failed: {e}")
        
        # Create content fingerprints
        content_lexical = self._create_lexical_fingerprint(lexical)
        content_syntactic = self._create_syntactic_fingerprint(syntactic)
        content_stylistic = self._create_stylistic_fingerprint(stylistic)
        
        # Calculate consistency scores
        lexical_score = self._score_fingerprint_match(
            content_lexical, profile.lexical_fingerprint
        )
        syntactic_score = self._score_fingerprint_match(
            content_syntactic, profile.syntactic_fingerprint
        )
        stylistic_score = self._score_fingerprint_match(
            content_stylistic, profile.stylistic_fingerprint
        )
        
        # Calculate overall score (weighted)
        overall = (
            lexical_score * 0.30 +
            syntactic_score * 0.30 +
            stylistic_score * 0.40
        )
        
        # Calculate cosine similarity
        cosine_sim = self.calculate_cosine_similarity(
            {**content_lexical, **content_syntactic, **content_stylistic},
            {**profile.lexical_fingerprint, **profile.syntactic_fingerprint, **profile.stylistic_fingerprint}
        )
        
        # Calculate KL divergence (simplified)
        kl_div = self._calculate_kl_divergence(
            content_stylistic, profile.stylistic_fingerprint
        )
        
        # Identify deviations
        deviations = self._identify_deviations(
            content_lexical, content_syntactic, content_stylistic,
            profile
        )
        
        # Determine severity
        severity = self._determine_severity(overall)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(
            deviations, lexical_score, syntactic_score, stylistic_score
        )
        
        logger.info(
            f"Consistency score: {overall:.1f} "
            f"(lexical: {lexical_score:.1f}, "
            f"syntactic: {syntactic_score:.1f}, "
            f"stylistic: {stylistic_score:.1f})"
        )
        
        return ConsistencyScore(
            overall_score=overall,
            lexical_consistency=lexical_score,
            syntactic_consistency=syntactic_score,
            stylistic_consistency=stylistic_score,
            cosine_similarity=cosine_sim,
            kl_divergence=kl_div,
            feature_deviations=deviations,
            is_consistent=overall >= self.threshold,
            severity=severity,
            alignment_suggestions=suggestions,
        )
    
    def calculate_cosine_similarity(
        self,
        features_a: Dict[str, float],
        features_b: Dict[str, float],
    ) -> float:
        """
        Calculate cosine similarity between feature vectors.
        
        Args:
            features_a: First feature vector
            features_b: Second feature vector
        
        Returns:
            Cosine similarity (0-1)
        """
        return self.metrics.cosine_similarity(features_a, features_b)
    
    def _create_lexical_fingerprint(self, lexical: any) -> Dict[str, float]:
        """Create lexical fingerprint from analysis."""
        return {
            "type_token_ratio": lexical.type_token_ratio,
            "hapax_ratio": lexical.hapax_legomena_ratio,
            "avg_word_length": lexical.average_word_length,
            "lexical_density": lexical.lexical_density,
            "complexity": lexical.complexity_score,
        }
    
    def _create_syntactic_fingerprint(self, syntactic: any) -> Dict[str, float]:
        """Create syntactic fingerprint from analysis."""
        fingerprint = {
            "avg_sentence_length": syntactic.average_sentence_length,
            "sentence_complexity": syntactic.sentence_complexity,
            "dependency_depth": syntactic.dependency_depth,
            "clause_density": syntactic.clause_density,
            "syntactic_variety": syntactic.syntactic_variety,
        }
        
        # Add POS distribution
        for pos, ratio in syntactic.pos_distribution.items():
            fingerprint[f"pos_{pos.lower()}"] = ratio
        
        return fingerprint
    
    def _create_stylistic_fingerprint(self, stylistic: any) -> Dict[str, float]:
        """Create stylistic fingerprint from analysis."""
        fingerprint = {
            "formality_score": stylistic.formality_score,
            "emoji_density": stylistic.emoji_density,
            "contraction_ratio": stylistic.contraction_ratio,
            "question_ratio": stylistic.question_ratio,
            "exclamation_ratio": stylistic.exclamation_ratio,
        }
        
        # Add punctuation density
        for punct, density in stylistic.punctuation_density.items():
            fingerprint[f"punct_{punct}"] = density
        
        return fingerprint
    
    def _score_fingerprint_match(
        self, content_fp: Dict[str, float], profile_fp: Dict[str, float]
    ) -> float:
        """
        Score how well content fingerprint matches profile.
        
        Returns score 0-100.
        """
        if not content_fp or not profile_fp:
            return 0.0
        
        # Get common keys
        common_keys = set(content_fp.keys()) & set(profile_fp.keys())
        
        if not common_keys:
            return 0.0
        
        # Calculate average similarity across features
        total_similarity = 0.0
        for key in common_keys:
            content_val = content_fp[key]
            profile_val = profile_fp[key]
            
            # Calculate normalized difference
            max_val = max(abs(content_val), abs(profile_val), 1.0)
            diff = abs(content_val - profile_val)
            similarity = 1.0 - min(diff / max_val, 1.0)
            
            total_similarity += similarity
        
        # Average and convert to 0-100 scale
        avg_similarity = total_similarity / len(common_keys)
        return avg_similarity * 100
    
    def _calculate_kl_divergence(
        self, content_fp: Dict[str, float], profile_fp: Dict[str, float]
    ) -> float:
        """
        Calculate KL divergence (simplified).
        
        Lower values = more similar.
        """
        return self.metrics.kl_divergence(content_fp, profile_fp)
    
    def _identify_deviations(
        self,
        content_lexical: Dict[str, float],
        content_syntactic: Dict[str, float],
        content_stylistic: Dict[str, float],
        profile: VoiceProfile,
    ) -> Dict[str, float]:
        """Identify significant deviations from profile."""
        deviations = {}
        
        # Check lexical deviations
        for key, content_val in content_lexical.items():
            profile_val = profile.lexical_fingerprint.get(key, content_val)
            deviation = abs(content_val - profile_val)
            if deviation > 0.2:  # Threshold
                deviations[f"lexical_{key}"] = deviation
        
        # Check stylistic deviations
        for key, content_val in content_stylistic.items():
            profile_val = profile.stylistic_fingerprint.get(key, content_val)
            deviation = abs(content_val - profile_val)
            if deviation > 10.0:  # Threshold for formality score
                deviations[f"stylistic_{key}"] = deviation
        
        return deviations
    
    def _determine_severity(self, overall_score: float) -> str:
        """Determine severity level from score."""
        if overall_score >= 85:
            return "none"
        elif overall_score >= 70:
            return "minor"
        elif overall_score >= 50:
            return "moderate"
        else:
            return "severe"
    
    def _generate_suggestions(
        self,
        deviations: Dict[str, float],
        lexical_score: float,
        syntactic_score: float,
        stylistic_score: float,
    ) -> List[str]:
        """Generate alignment suggestions."""
        suggestions = []
        
        if lexical_score < 75:
            suggestions.append(
                "Consider adjusting vocabulary complexity to match brand voice"
            )
        
        if syntactic_score < 75:
            suggestions.append(
                "Adjust sentence structure to align with typical brand patterns"
            )
        
        if stylistic_score < 75:
            if "stylistic_formality_score" in deviations:
                deviation = deviations["stylistic_formality_score"]
                if deviation > 0:
                    suggestions.append(
                        "Content is more formal than brand voice - consider a more casual tone"
                    )
                else:
                    suggestions.append(
                        "Content is less formal than brand voice - consider a more professional tone"
                    )
            
            if "stylistic_emoji_density" in deviations:
                suggestions.append(
                    "Emoji usage differs from brand voice patterns"
                )
        
        if not suggestions:
            suggestions.append("Content aligns well with brand voice")
        
        return suggestions