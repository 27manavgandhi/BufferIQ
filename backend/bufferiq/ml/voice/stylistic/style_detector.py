"""
Writing style detection and classification.

Identifies formal vs casual, technical vs conversational,
and other stylistic dimensions of brand voice.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict
import re
from collections import Counter

from nltk import word_tokenize, sent_tokenize


class WritingStyle(Enum):
    """Writing style categories."""
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL = "professional"


@dataclass
class StylisticFeatures:
    """Stylistic characteristics of text."""
    
    style: WritingStyle
    style_confidence: float
    formality_score: float  # 0-100, higher = more formal
    punctuation_density: Dict[str, float]
    emoji_density: float  # Emojis per 100 words
    capitalization_pattern: str  # "standard", "title", "all_caps", "mixed"
    contraction_ratio: float  # Contractions per 100 words
    question_ratio: float  # Questions per sentence
    exclamation_ratio: float  # Exclamations per sentence
    average_paragraph_length: float


class StyleDetector:
    """
    Detect and classify writing style.
    
    Identifies formal vs casual, technical vs conversational,
    and other stylistic dimensions of brand voice.
    
    Example:
```python
        detector = StyleDetector()
        features = detector.detect("Check out our new product! 🚀")
        print(f"Style: {features.style}")
        print(f"Formality: {features.formality_score:.1f}")
```
    """
    
    def __init__(self):
        """Initialize style detector."""
        self.formal_indicators = {
            'furthermore', 'moreover', 'consequently', 'nevertheless',
            'accordingly', 'henceforth', 'heretofore', 'wherein',
            'therefore', 'thus', 'hence', 'indeed', 'however'
        }
        self.casual_indicators = {
            'yeah', 'yep', 'nope', 'gonna', 'wanna', 'kinda',
            'sorta', 'hey', 'wow', 'cool', 'awesome', 'lol',
            'omg', 'btw', 'fyi', 'imho'
        }
        self.contractions = {
            "n't", "'ll", "'ve", "'re", "'m", "'d", "'s"
        }
        self.technical_indicators = {
            'algorithm', 'implementation', 'architecture', 'framework',
            'methodology', 'optimization', 'configuration', 'parameter',
            'function', 'variable', 'instance', 'interface'
        }
    
    def detect(self, text: str) -> StylisticFeatures:
        """
        Detect writing style features.
        
        Args:
            text: Text to analyze
        
        Returns:
            Stylistic features
        
        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) < 5:
            raise ValueError("Text too short for style detection")
        
        # Calculate all features
        formality = self.calculate_formality_score(text)
        punct_density = self._calculate_punctuation_density(text)
        emoji_density = self._calculate_emoji_density(text)
        cap_pattern = self._detect_capitalization_pattern(text)
        contraction_ratio = self._calculate_contraction_ratio(text)
        question_ratio = self._calculate_question_ratio(text)
        exclamation_ratio = self._calculate_exclamation_ratio(text)
        avg_para_length = self._calculate_avg_paragraph_length(text)
        
        # Determine primary style
        style, confidence = self._classify_style(text, formality)
        
        return StylisticFeatures(
            style=style,
            style_confidence=confidence,
            formality_score=formality,
            punctuation_density=punct_density,
            emoji_density=emoji_density,
            capitalization_pattern=cap_pattern,
            contraction_ratio=contraction_ratio,
            question_ratio=question_ratio,
            exclamation_ratio=exclamation_ratio,
            average_paragraph_length=avg_para_length,
        )
    
    def calculate_formality_score(self, text: str) -> float:
        """
        Calculate formality score (0-100).
        
        Args:
            text: Text to score
        
        Returns:
            Formality score
        """
        words = word_tokenize(text.lower())
        if not words:
            return 50.0
        
        # Count formal vs casual indicators
        formal_count = sum(1 for w in words if w in self.formal_indicators)
        casual_count = sum(1 for w in words if w in self.casual_indicators)
        
        # Count contractions
        contraction_count = sum(1 for w in words if any(c in w for c in self.contractions))
        
        # Count exclamations and emojis
        exclamations = text.count('!')
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE
        )
        emoji_count = len(emoji_pattern.findall(text))
        
        # Calculate score
        formality = 50.0
        formality += formal_count * 5
        formality -= casual_count * 5
        formality -= contraction_count * 2
        formality -= exclamations * 1
        formality -= emoji_count * 3
        
        # Clamp to 0-100
        return max(0.0, min(100.0, formality))
    
    def _calculate_punctuation_density(self, text: str) -> Dict[str, float]:
        """Calculate punctuation mark densities."""
        if not text:
            return {}
        
        word_count = len(word_tokenize(text))
        if word_count == 0:
            return {}
        
        return {
            'period': text.count('.') / word_count * 100,
            'comma': text.count(',') / word_count * 100,
            'exclamation': text.count('!') / word_count * 100,
            'question': text.count('?') / word_count * 100,
            'semicolon': text.count(';') / word_count * 100,
            'colon': text.count(':') / word_count * 100,
        }
    
    def _calculate_emoji_density(self, text: str) -> float:
        """Calculate emoji density (emojis per 100 words)."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        emoji_count = len(emoji_pattern.findall(text))
        word_count = len(word_tokenize(text))
        
        if word_count == 0:
            return 0.0
        
        return emoji_count / word_count * 100
    
    def _detect_capitalization_pattern(self, text: str) -> str:
        """Detect capitalization pattern."""
        if not text:
            return "unknown"
        
        # Remove punctuation for analysis
        clean_text = re.sub(r'[^\w\s]', '', text)
        words = clean_text.split()
        
        if not words:
            return "unknown"
        
        all_caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        title_case_count = sum(1 for w in words if w.istitle())
        
        all_caps_ratio = all_caps_count / len(words)
        title_ratio = title_case_count / len(words)
        
        if all_caps_ratio > 0.5:
            return "all_caps"
        elif title_ratio > 0.7:
            return "title"
        elif title_ratio < 0.3 and all_caps_ratio < 0.1:
            return "lowercase"
        else:
            return "standard"
    
    def _calculate_contraction_ratio(self, text: str) -> float:
        """Calculate contraction ratio (per 100 words)."""
        words = word_tokenize(text)
        if not words:
            return 0.0
        
        contraction_count = sum(
            1 for w in words if any(c in w for c in self.contractions)
        )
        
        return contraction_count / len(words) * 100
    
    def _calculate_question_ratio(self, text: str) -> float:
        """Calculate question ratio (per sentence)."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        question_count = sum(1 for s in sentences if '?' in s)
        return question_count / len(sentences)
    
    def _calculate_exclamation_ratio(self, text: str) -> float:
        """Calculate exclamation ratio (per sentence)."""
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        exclamation_count = sum(1 for s in sentences if '!' in s)
        return exclamation_count / len(sentences)
    
    def _calculate_avg_paragraph_length(self, text: str) -> float:
        """Calculate average paragraph length (in sentences)."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        total_sentences = 0
        for para in paragraphs:
            sentences = sent_tokenize(para)
            total_sentences += len(sentences)
        
        return total_sentences / len(paragraphs)
    
    def _classify_style(self, text: str, formality: float) -> tuple:
        """
        Classify writing style based on features.
        
        Returns:
            Tuple of (WritingStyle, confidence)
        """
        words = word_tokenize(text.lower())
        
        # Count technical words
        technical_count = sum(1 for w in words if w in self.technical_indicators)
        technical_ratio = technical_count / len(words) if words else 0
        
        # Determine style
        if technical_ratio > 0.05:
            return WritingStyle.TECHNICAL, 0.8
        elif formality > 70:
            return WritingStyle.FORMAL, 0.85
        elif formality > 55:
            return WritingStyle.PROFESSIONAL, 0.75
        elif formality > 40:
            return WritingStyle.CONVERSATIONAL, 0.7
        else:
            return WritingStyle.CASUAL, 0.8