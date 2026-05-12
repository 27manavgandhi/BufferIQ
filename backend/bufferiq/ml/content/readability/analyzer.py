"""
Readability analysis.

Combines multiple readability metrics.
"""

from dataclasses import dataclass

from bufferiq.ml.content.readability.metrics import ReadabilityMetrics


@dataclass
class ReadabilityScores:
    """Readability metrics for text."""

    flesch_reading_ease: float
    flesch_kincaid_grade: float
    gunning_fog: float
    smog_index: float
    coleman_liau: float
    automated_readability: float
    average_grade_level: float
    reading_difficulty: str


class ReadabilityAnalyzer:
    """
        Calculate readability scores for text.

        Implements multiple readability formulas and provides
        an overall assessment of text difficulty.

        Example:
    ```python
            analyzer = ReadabilityAnalyzer()
            scores = analyzer.analyze("This is a simple sentence.")
            print(scores.flesch_reading_ease)  # 90.5 (very easy)
            print(scores.reading_difficulty)   # "easy"
    ```
    """

    def __init__(self) -> None:
        """Initialize readability analyzer."""
        self.metrics = ReadabilityMetrics()

    def analyze(self, text: str) -> ReadabilityScores:
        """
        Calculate all readability metrics.

        Args:
            text: Text to analyze

        Returns:
            Readability scores

        Raises:
            ValueError: If text is too short
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text is too short for readability analysis")

        # Calculate all metrics
        flesch_ease = self.metrics.flesch_reading_ease(text)
        flesch_grade = self.metrics.flesch_kincaid_grade(text)
        gunning = self.metrics.gunning_fog_index(text)
        smog = self.metrics.smog_index(text)
        coleman = self.metrics.coleman_liau_index(text)

        # Automated Readability Index (simplified)
        ari = flesch_grade  # Use FK grade as proxy

        # Average grade level
        avg_grade = (flesch_grade + gunning + smog + coleman + ari) / 5

        # Difficulty classification
        difficulty = self.get_difficulty_level(avg_grade)

        return ReadabilityScores(
            flesch_reading_ease=flesch_ease,
            flesch_kincaid_grade=flesch_grade,
            gunning_fog=gunning,
            smog_index=smog,
            coleman_liau=coleman,
            automated_readability=ari,
            average_grade_level=avg_grade,
            reading_difficulty=difficulty,
        )

    def get_difficulty_level(self, avg_grade: float) -> str:
        """
        Get difficulty classification.

        Args:
            avg_grade: Average grade level

        Returns:
            "easy", "medium", or "hard"
        """
        if avg_grade < 6:
            return "easy"
        elif avg_grade < 12:
            return "medium"
        else:
            return "hard"
