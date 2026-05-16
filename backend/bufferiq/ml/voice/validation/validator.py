"""
Pre-publish voice validation.

Validates content against brand voice before publishing.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import logging

from bufferiq.ml.voice.profiler.builder import VoiceProfile
from bufferiq.ml.voice.consistency.scorer import VoiceConsistencyScorer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class ValidationResult:
    """Voice validation result."""
    
    passed: bool
    score: float
    threshold: float
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]


class VoiceValidator:
    """
    Validate content against brand voice.
    
    Performs pre-publish validation to ensure content
    aligns with brand voice guidelines.
    
    Example:
```python
        validator = VoiceValidator(threshold=75.0)
        result = validator.validate(
            text="Your post content",
            profile=brand_voice,
            platform="linkedin"
        )
        
        if result.passed:
            print("✓ Content approved")
        else:
            print("✗ Content needs revision")
            for issue in result.issues:
                print(f"  - {issue}")
```
    """
    
    def __init__(
        self,
        threshold: float = 75.0,
        scorer: Optional[VoiceConsistencyScorer] = None,
    ):
        """
        Initialize voice validator.
        
        Args:
            threshold: Minimum consistency score to pass (0-100)
            scorer: Optional consistency scorer
        """
        self.threshold = threshold
        self.scorer = scorer or VoiceConsistencyScorer(consistency_threshold=threshold)
    
    def validate(
        self,
        text: str,
        profile: VoiceProfile,
        platform: str,
    ) -> ValidationResult:
        """
        Validate content against voice profile.
        
        Args:
            text: Content to validate
            profile: Brand voice profile
            platform: Target platform
        
        Returns:
            Validation result
        
        Raises:
            ValueError: If platform not supported
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        logger.info(f"Validating content for {profile.brand_id} on {platform}")
        
        # Score consistency
        try:
            consistency = self.scorer.score(text, profile, platform)
        except ValueError as e:
            return ValidationResult(
                passed=False,
                score=0.0,
                threshold=self.threshold,
                issues=[f"Validation failed: {str(e)}"],
                warnings=[],
                suggestions=[],
            )
        
        # Determine pass/fail
        passed = consistency.overall_score >= self.threshold
        
        # Collect issues
        issues = []
        if consistency.severity == "severe":
            issues.append("Major voice inconsistency detected")
        elif consistency.severity == "moderate":
            issues.append("Moderate voice inconsistency detected")
        
        # Collect warnings
        warnings = []
        if consistency.lexical_consistency < 70:
            warnings.append("Vocabulary doesn't match typical brand usage")
        if consistency.stylistic_consistency < 70:
            warnings.append("Tone differs from established brand voice")
        
        # Get suggestions
        suggestions = consistency.alignment_suggestions
        
        logger.info(
            f"Validation complete: passed={passed}, score={consistency.overall_score:.1f}"
        )
        
        return ValidationResult(
            passed=passed,
            score=consistency.overall_score,
            threshold=self.threshold,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
        )
    
    def validate_batch(
        self,
        contents: List[str],
        profile: VoiceProfile,
        platform: str,
    ) -> List[ValidationResult]:
        """
        Validate multiple pieces of content.
        
        Args:
            contents: List of content to validate
            profile: Brand voice profile
            platform: Target platform
        
        Returns:
            List of validation results
        """
        results = []
        
        for content in contents:
            try:
                result = self.validate(content, profile, platform)
                results.append(result)
            except Exception as e:
                logger.error(f"Validation error: {e}")
                results.append(ValidationResult(
                    passed=False,
                    score=0.0,
                    threshold=self.threshold,
                    issues=[f"Validation error: {str(e)}"],
                    warnings=[],
                    suggestions=[],
                ))
        
        return results