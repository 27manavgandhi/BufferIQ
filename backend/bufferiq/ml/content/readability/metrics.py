"""
Readability metrics calculation.

Implements various readability formulas.
"""

import re
from typing import Optional


class ReadabilityMetrics:
    """
        Calculate readability metrics.

        Implements multiple readability formulas:
        - Flesch Reading Ease
        - Flesch-Kincaid Grade Level
        - Gunning Fog Index
        - SMOG Index
        - Coleman-Liau Index

        Example:
    ```python
            metrics = ReadabilityMetrics()
            text = "This is a simple sentence."
            flesch = metrics.flesch_reading_ease(text)
            print(f"Flesch: {flesch:.1f}")
    ```
    """

    def __init__(self) -> None:
        """Initialize readability metrics calculator."""
        pass

    def flesch_reading_ease(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score.

        Score ranges from 0-100, higher means easier to read.
        90-100: Very Easy
        80-89: Easy
        70-79: Fairly Easy
        60-69: Standard
        50-59: Fairly Difficult
        30-49: Difficult
        0-29: Very Confusing

        Args:
            text: Text to analyze

        Returns:
            Flesch Reading Ease score

        Raises:
            ValueError: If text is too short
        """
        words = self._count_words(text)
        sentences = self._count_sentences(text)
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0.0

        asl = words / sentences  # Average sentence length
        asw = syllables / words  # Average syllables per word

        score = 206.835 - (1.015 * asl) - (84.6 * asw)
        return max(0.0, min(100.0, score))

    def flesch_kincaid_grade(self, text: str) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.

        Returns the US grade level required to understand the text.

        Args:
            text: Text to analyze

        Returns:
            Grade level (e.g., 8.0 = 8th grade)

        Raises:
            ValueError: If text is too short
        """
        words = self._count_words(text)
        sentences = self._count_sentences(text)
        syllables = self._count_syllables(text)

        if sentences == 0 or words == 0:
            return 0.0

        asl = words / sentences
        asw = syllables / words

        grade = (0.39 * asl) + (11.8 * asw) - 15.59
        return max(0.0, grade)

    def gunning_fog_index(self, text: str) -> float:
        """
        Calculate Gunning Fog Index.

        Estimates years of education needed to understand the text.

        Args:
            text: Text to analyze

        Returns:
            Years of education required

        Raises:
            ValueError: If text is too short
        """
        words = self._count_words(text)
        sentences = self._count_sentences(text)
        complex_words = self._count_complex_words(text)

        if sentences == 0 or words == 0:
            return 0.0

        asl = words / sentences
        pcw = (complex_words / words) * 100  # Percentage complex words

        fog = 0.4 * (asl + pcw)
        return max(0.0, fog)

    def smog_index(self, text: str) -> float:
        """
        Calculate SMOG Index.

        Simple Measure of Gobbledygook estimates years of education.

        Args:
            text: Text to analyze

        Returns:
            Years of education required

        Raises:
            ValueError: If text is too short
        """
        sentences = self._count_sentences(text)
        polysyllables = self._count_polysyllables(text)

        if sentences < 3:
            # SMOG requires at least 30 sentences, but we'll adapt
            return 0.0

        # Simplified SMOG
        smog = 1.0430 * ((polysyllables * (30 / sentences)) ** 0.5) + 3.1291
        return max(0.0, smog)

    def coleman_liau_index(self, text: str) -> float:
        """
        Calculate Coleman-Liau Index.

        Uses characters instead of syllables for grade level.

        Args:
            text: Text to analyze

        Returns:
            Grade level

        Raises:
            ValueError: If text is too short
        """
        words = self._count_words(text)
        sentences = self._count_sentences(text)
        characters = self._count_characters(text)

        if words == 0:
            return 0.0

        l = (characters / words) * 100  # Letters per 100 words
        s = (sentences / words) * 100  # Sentences per 100 words

        cli = (0.0588 * l) - (0.296 * s) - 15.8
        return max(0.0, cli)

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        words = re.findall(r"\b\w+\b", text)
        return len(words)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r"[.!?]+", text)
        count = len([s for s in sentences if s.strip()])
        return max(1, count)

    def _count_characters(self, text: str) -> int:
        """Count alphanumeric characters."""
        return len(re.findall(r"[a-zA-Z0-9]", text))

    def _count_syllables(self, text: str) -> int:
        """Count syllables (simplified)."""
        words = re.findall(r"\b\w+\b", text.lower())
        total = 0
        for word in words:
            total += self._syllables_in_word(word)
        return max(1, total)

    def _syllables_in_word(self, word: str) -> int:
        """Count syllables in a word (simplified)."""
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            syllables -= 1

        return max(1, syllables)

    def _count_complex_words(self, text: str) -> int:
        """Count words with 3+ syllables."""
        words = re.findall(r"\b\w+\b", text.lower())
        complex_count = 0
        for word in words:
            if self._syllables_in_word(word) >= 3:
                complex_count += 1
        return complex_count

    def _count_polysyllables(self, text: str) -> int:
        """Count words with 3+ syllables (polysyllables)."""
        return self._count_complex_words(text)
