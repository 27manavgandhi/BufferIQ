"""
Lexical diversity and complexity analysis.

Analyzes vocabulary richness, word choice patterns,
and lexical complexity for voice profiling.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import re
from collections import Counter

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class LexicalMetrics:
    """Lexical diversity and complexity metrics."""
    
    type_token_ratio: float  # Vocabulary richness (0-1)
    hapax_legomena_ratio: float  # Unique words ratio
    average_word_length: float
    vocabulary_size: int
    unique_words: int
    word_frequency_dist: Dict[str, int]
    lexical_density: float  # Content words / total words
    complexity_score: float  # 0-100


class LexicalAnalyzer:
    """
    Analyze lexical characteristics of text.
    
    Measures vocabulary richness, word choice patterns,
    and lexical complexity for voice profiling.
    
    Example:
```python
        analyzer = LexicalAnalyzer()
        metrics = analyzer.analyze("Your brand content here")
        print(f"TTR: {metrics.type_token_ratio:.3f}")
        print(f"Complexity: {metrics.complexity_score:.1f}")
```
    """
    
    def __init__(self):
        """Initialize lexical analyzer."""
        self.stop_words = set(stopwords.words('english'))
    
    def analyze(self, text: str) -> LexicalMetrics:
        """
        Analyze lexical characteristics.
        
        Args:
            text: Text to analyze
        
        Returns:
            Lexical metrics
        
        Raises:
            ValueError: If text is empty or too short
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for lexical analysis (minimum 10 characters)")
        
        # Tokenize
        tokens = self._tokenize(text)
        
        if len(tokens) < 3:
            raise ValueError("Text too short for lexical analysis (minimum 3 tokens)")
        
        # Get POS tags
        pos_tags = pos_tag(tokens)
        
        # Calculate metrics
        ttr = self.calculate_ttr(tokens)
        hapax_ratio = self._calculate_hapax_ratio(tokens)
        avg_word_length = sum(len(word) for word in tokens) / len(tokens)
        vocab_size = len(set(tokens))
        unique_words = vocab_size
        word_freq = dict(Counter(tokens).most_common(50))
        lexical_density = self.calculate_lexical_density(tokens, pos_tags)
        complexity = self._calculate_complexity(
            ttr, lexical_density, avg_word_length, vocab_size
        )
        
        return LexicalMetrics(
            type_token_ratio=ttr,
            hapax_legomena_ratio=hapax_ratio,
            average_word_length=avg_word_length,
            vocabulary_size=vocab_size,
            unique_words=unique_words,
            word_frequency_dist=word_freq,
            lexical_density=lexical_density,
            complexity_score=complexity,
        )
    
    def calculate_ttr(self, tokens: List[str]) -> float:
        """
        Calculate Type-Token Ratio.
        
        Args:
            tokens: List of word tokens
        
        Returns:
            TTR score (0-1)
        """
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)
    
    def calculate_lexical_density(
        self, tokens: List[str], pos_tags: List[tuple]
    ) -> float:
        """
        Calculate lexical density (content words ratio).
        
        Args:
            tokens: Word tokens
            pos_tags: POS tag tuples
        
        Returns:
            Lexical density (0-1)
        """
        if not tokens:
            return 0.0
        
        content_tags = {
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'JJ', 'JJR', 'JJS',  # Adjectives
            'RB', 'RBR', 'RBS'  # Adverbs
        }
        content_words = sum(1 for _, tag in pos_tags if tag in content_tags)
        return content_words / len(tokens)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of word tokens
        """
        # Use NLTK tokenizer
        tokens = word_tokenize(text.lower())
        # Filter out punctuation and non-alphabetic tokens
        tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
        return tokens
    
    def _calculate_hapax_ratio(self, tokens: List[str]) -> float:
        """
        Calculate hapax legomena ratio (words appearing only once).
        
        Args:
            tokens: Word tokens
        
        Returns:
            Hapax ratio (0-1)
        """
        if not tokens:
            return 0.0
        
        freq = Counter(tokens)
        hapax_count = sum(1 for count in freq.values() if count == 1)
        return hapax_count / len(tokens)
    
    def _calculate_complexity(
        self,
        ttr: float,
        lexical_density: float,
        avg_word_length: float,
        vocab_size: int
    ) -> float:
        """
        Calculate overall lexical complexity score (0-100).
        
        Args:
            ttr: Type-token ratio
            lexical_density: Lexical density
            avg_word_length: Average word length
            vocab_size: Vocabulary size
        
        Returns:
            Complexity score (0-100)
        """
        # Normalize vocab size (cap at 200)
        vocab_score = min(vocab_size / 200, 1.0)
        
        # Normalize word length (cap at 7)
        length_score = min(avg_word_length / 7, 1.0)
        
        # Weighted combination
        complexity = (
            ttr * 25 +
            lexical_density * 25 +
            vocab_score * 25 +
            length_score * 25
        )
        
        return min(complexity, 100.0)