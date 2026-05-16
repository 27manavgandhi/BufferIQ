"""
Vocabulary fingerprinting and analysis.

Analyzes vocabulary usage patterns and word choice
for brand voice characterization.
"""

from typing import Dict, List, Set
from collections import Counter
import re

from nltk import word_tokenize
from nltk.corpus import stopwords


class VocabularyAnalyzer:
    """
    Analyze vocabulary usage and word choice patterns.
    
    Creates vocabulary fingerprints for brand voice identification.
    
    Example:
```python
        analyzer = VocabularyAnalyzer()
        fingerprint = analyzer.create_fingerprint(texts)
        similarity = analyzer.compare_fingerprints(fp1, fp2)
```
    """
    
    def __init__(self):
        """Initialize vocabulary analyzer."""
        self.stop_words = set(stopwords.words('english'))
    
    def create_fingerprint(
        self, texts: List[str], top_n: int = 100
    ) -> Dict[str, float]:
        """
        Create vocabulary fingerprint from texts.
        
        Args:
            texts: List of texts
            top_n: Number of top words to include
        
        Returns:
            Vocabulary fingerprint (word -> normalized frequency)
        
        Raises:
            ValueError: If texts is empty
        """
        if not texts:
            raise ValueError("Cannot create fingerprint from empty text list")
        
        # Collect all words
        all_words: List[str] = []
        for text in texts:
            words = self._extract_words(text)
            all_words.extend(words)
        
        if not all_words:
            raise ValueError("No valid words found in texts")
        
        # Count frequencies
        word_freq = Counter(all_words)
        
        # Get top N words
        top_words = word_freq.most_common(top_n)
        
        # Normalize frequencies
        total = sum(count for _, count in top_words)
        fingerprint = {
            word: count / total for word, count in top_words
        }
        
        return fingerprint
    
    def compare_fingerprints(
        self, fp1: Dict[str, float], fp2: Dict[str, float]
    ) -> float:
        """
        Compare two vocabulary fingerprints using cosine similarity.
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
        
        Returns:
            Similarity score (0-1)
        """
        if not fp1 or not fp2:
            return 0.0
        
        # Get all words
        all_words = set(fp1.keys()) | set(fp2.keys())
        
        # Calculate cosine similarity
        dot_product = sum(
            fp1.get(word, 0) * fp2.get(word, 0) for word in all_words
        )
        
        magnitude1 = sum(v ** 2 for v in fp1.values()) ** 0.5
        magnitude2 = sum(v ** 2 for v in fp2.values()) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def get_distinctive_words(
        self, target_fp: Dict[str, float], baseline_fp: Dict[str, float], top_n: int = 20
    ) -> List[tuple]:
        """
        Get words distinctive to target fingerprint.
        
        Args:
            target_fp: Target vocabulary fingerprint
            baseline_fp: Baseline vocabulary fingerprint
            top_n: Number of distinctive words to return
        
        Returns:
            List of (word, distinctiveness_score) tuples
        """
        distinctive: List[tuple] = []
        
        for word, target_freq in target_fp.items():
            baseline_freq = baseline_fp.get(word, 0)
            # Distinctiveness = target frequency - baseline frequency
            distinctiveness = target_freq - baseline_freq
            distinctive.append((word, distinctiveness))
        
        # Sort by distinctiveness
        distinctive.sort(key=lambda x: x[1], reverse=True)
        
        return distinctive[:top_n]
    
    def _extract_words(self, text: str) -> List[str]:
        """
        Extract content words from text.
        
        Args:
            text: Text to process
        
        Returns:
            List of content words
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter: alphabetic, not stop words, length > 2
        words = [
            t for t in tokens
            if t.isalpha() and t not in self.stop_words and len(t) > 2
        ]
        
        return words