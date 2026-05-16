"""
Voice profile building and management.

Creates multi-dimensional voice representations
with versioning and signature generation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any
import json
import logging

from bufferiq.ml.voice.extraction.extractor import VoiceFeatures
from bufferiq.ml.voice.profiler.signature_generator import VoiceSignatureGenerator

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class VoiceProfile:
    """Complete brand voice profile."""
    
    profile_id: str
    brand_id: str
    version: int
    created_at: datetime
    
    # Core characteristics
    lexical_fingerprint: Dict[str, float]
    syntactic_fingerprint: Dict[str, float]
    stylistic_fingerprint: Dict[str, float]
    
    # Voice signature (hash)
    signature: str
    
    # Metadata
    confidence: float
    sample_size: int
    platform_profiles: Dict[str, Dict]
    
    # Evolution tracking
    previous_version: Optional[str] = None
    drift_from_previous: Optional[float] = None


class VoiceProfileBuilder:
    """
    Build comprehensive voice profiles from features.
    
    Creates multi-dimensional voice representations
    with versioning and signature generation.
    
    Example:
```python
        builder = VoiceProfileBuilder()
        profile = builder.build(
            brand_id="brand123",
            voice_features=extracted_features,
            platform="linkedin"
        )
        print(f"Profile ID: {profile.profile_id}")
        print(f"Signature: {profile.signature}")
```
    """
    
    def __init__(self, signature_generator: Optional[VoiceSignatureGenerator] = None):
        """Initialize profile builder."""
        self.signature_generator = signature_generator or VoiceSignatureGenerator()
    
    def build(
        self,
        brand_id: str,
        voice_features: VoiceFeatures,
        platform: str,
        previous_profile: Optional[VoiceProfile] = None,
    ) -> VoiceProfile:
        """
        Build voice profile from extracted features.
        
        Args:
            brand_id: Brand identifier
            voice_features: Extracted voice features
            platform: Primary platform
            previous_profile: Optional previous version
        
        Returns:
            Voice profile
        
        Raises:
            ValueError: If platform not supported or features insufficient
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported platforms: {SUPPORTED_PLATFORMS}"
            )
        
        if voice_features.sample_size < 10:
            raise ValueError(
                f"Insufficient sample size for profile building. "
                f"Found {voice_features.sample_size}, minimum required: 10"
            )
        
        # Create fingerprints
        lexical_fp = self._create_lexical_fingerprint(voice_features)
        syntactic_fp = self._create_syntactic_fingerprint(voice_features)
        stylistic_fp = self._create_stylistic_fingerprint(voice_features)
        
        # Generate signature
        profile_data = {
            "lexical": lexical_fp,
            "syntactic": syntactic_fp,
            "stylistic": stylistic_fp,
        }
        signature = self.signature_generator.generate(profile_data)
        
        # Calculate version and drift
        version = 1 if previous_profile is None else previous_profile.version + 1
        drift = (
            self._calculate_drift(previous_profile, profile_data)
            if previous_profile
            else None
        )
        
        # Generate profile ID
        profile_id = f"{brand_id}_{platform}_v{version}_{datetime.utcnow().strftime('%Y%m%d')}"
        
        logger.info(f"Built voice profile {profile_id} with signature {signature[:16]}...")
        
        return VoiceProfile(
            profile_id=profile_id,
            brand_id=brand_id,
            version=version,
            created_at=datetime.utcnow(),
            lexical_fingerprint=lexical_fp,
            syntactic_fingerprint=syntactic_fp,
            stylistic_fingerprint=stylistic_fp,
            signature=signature,
            confidence=voice_features.confidence_score,
            sample_size=voice_features.sample_size,
            platform_profiles=voice_features.platform_variations,
            previous_version=previous_profile.profile_id if previous_profile else None,
            drift_from_previous=drift,
        )
    
    def _create_lexical_fingerprint(self, features: VoiceFeatures) -> Dict[str, float]:
        """Create lexical fingerprint from features."""
        return {
            "type_token_ratio": features.lexical_profile.type_token_ratio,
            "hapax_ratio": features.lexical_profile.hapax_legomena_ratio,
            "avg_word_length": features.lexical_profile.average_word_length,
            "lexical_density": features.lexical_profile.lexical_density,
            "complexity": features.lexical_profile.complexity_score,
            "vocabulary_size": float(features.lexical_profile.vocabulary_size),
        }
    
    def _create_syntactic_fingerprint(self, features: VoiceFeatures) -> Dict[str, float]:
        """Create syntactic fingerprint from features."""
        fingerprint = {
            "avg_sentence_length": features.syntactic_profile.average_sentence_length,
            "sentence_complexity": features.syntactic_profile.sentence_complexity,
            "dependency_depth": features.syntactic_profile.dependency_depth,
            "clause_density": features.syntactic_profile.clause_density,
            "syntactic_variety": features.syntactic_profile.syntactic_variety,
        }
        
        # Add POS distribution
        for pos, ratio in features.syntactic_profile.pos_distribution.items():
            fingerprint[f"pos_{pos.lower()}"] = ratio
        
        return fingerprint
    
    def _create_stylistic_fingerprint(self, features: VoiceFeatures) -> Dict[str, float]:
        """Create stylistic fingerprint from features."""
        fingerprint = {
            "formality_score": features.stylistic_profile.formality_score,
            "emoji_density": features.stylistic_profile.emoji_density,
            "contraction_ratio": features.stylistic_profile.contraction_ratio,
            "question_ratio": features.stylistic_profile.question_ratio,
            "exclamation_ratio": features.stylistic_profile.exclamation_ratio,
            "avg_paragraph_length": features.stylistic_profile.average_paragraph_length,
            "style_confidence": features.stylistic_profile.style_confidence,
        }
        
        # Add punctuation density
        for punct, density in features.stylistic_profile.punctuation_density.items():
            fingerprint[f"punct_{punct}"] = density
        
        return fingerprint
    
    def _calculate_drift(
        self, previous: VoiceProfile, current_data: Dict[str, Dict[str, float]]
    ) -> float:
        """
        Calculate drift from previous profile.
        
        Args:
            previous: Previous voice profile
            current_data: Current profile data
        
        Returns:
            Drift score (0-1)
        """
        # Compare fingerprints
        lexical_drift = self._compare_fingerprints(
            previous.lexical_fingerprint, current_data["lexical"]
        )
        syntactic_drift = self._compare_fingerprints(
            previous.syntactic_fingerprint, current_data["syntactic"]
        )
        stylistic_drift = self._compare_fingerprints(
            previous.stylistic_fingerprint, current_data["stylistic"]
        )
        
        # Average drift
        avg_drift = (lexical_drift + syntactic_drift + stylistic_drift) / 3
        return avg_drift
    
    def _compare_fingerprints(
        self, fp1: Dict[str, float], fp2: Dict[str, float]
    ) -> float:
        """
        Compare two fingerprints (returns drift/distance, not similarity).
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
        
        Returns:
            Drift score (0-1)
        """
        # Get all keys
        all_keys = set(fp1.keys()) | set(fp2.keys())
        
        # Calculate squared differences
        total_diff = 0.0
        for key in all_keys:
            v1 = fp1.get(key, 0.0)
            v2 = fp2.get(key, 0.0)
            total_diff += (v1 - v2) ** 2
        
        # Normalize
        drift = (total_diff / len(all_keys)) ** 0.5 if all_keys else 0.0
        return min(drift, 1.0)