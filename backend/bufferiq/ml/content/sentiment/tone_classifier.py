"""
Tone classification for text.

Classifies text into professional, casual, urgent, or friendly tones.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class Tone(str, Enum):
    """Tone categories."""

    PROFESSIONAL = "professional"
    CASUAL = "casual"
    URGENT = "urgent"
    FRIENDLY = "friendly"


@dataclass
class ToneResult:
    """Tone classification result."""

    tone: Tone
    confidence: float
    scores: Dict[str, float]


class ToneClassifier:
    """
        Classify tone of text.

        Uses keyword-based approach with tone indicators.

        Example:
    ```python
            classifier = ToneClassifier()
            result = classifier.classify("Please review this ASAP!")
            print(result.tone)  # Tone.URGENT
            print(result.confidence)  # 0.75
    ```
    """

    def __init__(self) -> None:
        """Initialize tone classifier with indicators."""
        self.tone_indicators: Dict[Tone, Set[str]] = {
            Tone.PROFESSIONAL: {
                "please",
                "kindly",
                "regards",
                "sincerely",
                "professional",
                "formal",
                "respectfully",
            },
            Tone.CASUAL: {
                "hey",
                "hi",
                "cool",
                "awesome",
                "yeah",
                "yep",
                "gonna",
                "wanna",
            },
            Tone.URGENT: {
                "urgent",
                "asap",
                "immediately",
                "now",
                "quick",
                "fast",
                "hurry",
                "critical",
            },
            Tone.FRIENDLY: {
                "thanks",
                "thank you",
                "appreciate",
                "great",
                "wonderful",
                "lovely",
                "nice",
            },
        }

    def classify(self, text: str) -> ToneResult:
        """
        Classify tone of text.

        Args:
            text: Text to classify

        Returns:
            Tone classification result

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text_lower = text.lower()

        # Count tone indicators
        tone_counts: Dict[Tone, int] = {tone: 0 for tone in Tone}

        for tone, indicators in self.tone_indicators.items():
            for indicator in indicators:
                tone_counts[tone] += text_lower.count(indicator)

        # Calculate scores
        total_matches = sum(tone_counts.values())
        if total_matches == 0:
            # Default to professional if no indicators
            return ToneResult(
                tone=Tone.PROFESSIONAL,
                confidence=0.5,
                scores={t.value: 0.25 for t in Tone},
            )

        scores = {
            tone.value: count / total_matches for tone, count in tone_counts.items()
        }

        # Determine dominant tone
        dominant_tone = max(tone_counts.items(), key=lambda x: x[1])
        tone = dominant_tone[0]
        confidence = scores[tone.value]

        return ToneResult(tone=tone, confidence=confidence, scores=scores)
