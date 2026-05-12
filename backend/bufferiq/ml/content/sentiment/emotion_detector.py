"""
Emotion detection in text.

Detects basic emotions using keyword-based approach.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set


class Emotion(str, Enum):
    """Emotion categories."""

    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    NEUTRAL = "neutral"


@dataclass
class EmotionResult:
    """Emotion detection result."""

    emotion: Emotion
    confidence: float
    scores: Dict[str, float]


class EmotionDetector:
    """
        Detect emotions in text.

        Uses keyword-based approach with emotion lexicons.

        Example:
    ```python
            detector = EmotionDetector()
            result = detector.detect("I'm so happy and excited!")
            print(result.emotion)  # Emotion.JOY
            print(result.confidence)  # 0.85
    ```
    """

    def __init__(self) -> None:
        """Initialize emotion detector with lexicons."""
        self.emotion_keywords: Dict[Emotion, Set[str]] = {
            Emotion.JOY: {
                "happy",
                "joy",
                "excited",
                "great",
                "amazing",
                "wonderful",
                "love",
                "fantastic",
                "excellent",
                "brilliant",
            },
            Emotion.ANGER: {
                "angry",
                "mad",
                "furious",
                "hate",
                "terrible",
                "awful",
                "annoyed",
                "frustrated",
                "outraged",
            },
            Emotion.SADNESS: {
                "sad",
                "unhappy",
                "depressed",
                "disappointed",
                "miserable",
                "sorry",
                "regret",
                "crying",
            },
            Emotion.FEAR: {
                "afraid",
                "scared",
                "fear",
                "worried",
                "anxious",
                "nervous",
                "terrified",
                "panic",
            },
            Emotion.SURPRISE: {
                "surprised",
                "shocked",
                "amazed",
                "astonished",
                "wow",
                "unexpected",
                "sudden",
            },
        }

    def detect(self, text: str) -> EmotionResult:
        """
        Detect emotion in text.

        Args:
            text: Text to analyze

        Returns:
            Emotion detection result

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        text_lower = text.lower()
        words = set(text_lower.split())

        # Count emotion keywords
        emotion_counts: Dict[Emotion, int] = {
            emotion: 0 for emotion in Emotion if emotion != Emotion.NEUTRAL
        }

        for emotion, keywords in self.emotion_keywords.items():
            emotion_counts[emotion] = len(words.intersection(keywords))

        # Calculate scores
        total_matches = sum(emotion_counts.values())
        if total_matches == 0:
            return EmotionResult(
                emotion=Emotion.NEUTRAL,
                confidence=1.0,
                scores={e.value: 0.0 for e in Emotion},
            )

        scores = {
            emotion.value: count / total_matches
            for emotion, count in emotion_counts.items()
        }
        scores[Emotion.NEUTRAL.value] = 0.0

        # Determine dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])
        if dominant_emotion[1] == 0:
            emotion = Emotion.NEUTRAL
            confidence = 1.0
        else:
            emotion = dominant_emotion[0]
            confidence = scores[emotion.value]

        return EmotionResult(emotion=emotion, confidence=confidence, scores=scores)
