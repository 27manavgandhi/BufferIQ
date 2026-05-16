"""
Punctuation and capitalization pattern analysis.

Analyzes writing patterns for voice fingerprinting.
"""

from typing import Dict, List
import re
from collections import Counter

from nltk import word_tokenize, sent_tokenize


class PatternAnalyzer:
    """
    Analyze writing patterns (punctuation, capitalization, etc.).
    
    Identifies characteristic patterns for voice profiling.
    
    Example:
```python
        analyzer = PatternAnalyzer()
        patterns = analyzer.analyze(texts)
        print(f"Patterns: {patterns}")
```
    """
    
    def __init__(self):
        """Initialize pattern analyzer."""
        pass
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze writing patterns.
        
        Args:
            text: Text to analyze
        
        Returns:
            Pattern dictionary
        
        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for pattern analysis")
        
        return {
            'punctuation_patterns': self._analyze_punctuation(text),
            'capitalization_patterns': self._analyze_capitalization(text),
            'formatting_patterns': self._analyze_formatting(text),
            'sentence_starters': self._analyze_sentence_starters(text),
        }
    
    def _analyze_punctuation(self, text: str) -> Dict[str, any]:
        """Analyze punctuation usage patterns."""
        # Count punctuation marks
        punct_counts = {
            'period': text.count('.'),
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'exclamation': text.count('!'),
            'question': text.count('?'),
            'dash': text.count('--') + text.count('—'),
            'parentheses': text.count('('),
        }
        
        # Analyze patterns
        sentences = sent_tokenize(text)
        
        # Sentence-ending punctuation
        ending_punct = Counter()
        for sent in sentences:
            if sent.endswith('!'):
                ending_punct['exclamation'] += 1
            elif sent.endswith('?'):
                ending_punct['question'] += 1
            else:
                ending_punct['period'] += 1
        
        return {
            'counts': punct_counts,
            'sentence_endings': dict(ending_punct),
            'comma_per_sentence': punct_counts['comma'] / len(sentences) if sentences else 0,
        }
    
    def _analyze_capitalization(self, text: str) -> Dict[str, any]:
        """Analyze capitalization patterns."""
        words = word_tokenize(text)
        
        if not words:
            return {}
        
        # Count different capitalization types
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
        title_case = sum(1 for w in words if w.istitle())
        lower_case = sum(1 for w in words if w.islower())
        
        return {
            'all_caps_ratio': all_caps / len(words),
            'title_case_ratio': title_case / len(words),
            'lower_case_ratio': lower_case / len(words),
        }
    
    def _analyze_formatting(self, text: str) -> Dict[str, any]:
        """Analyze text formatting patterns."""
        return {
            'has_bullets': bool(re.search(r'^\s*[-•*]\s', text, re.MULTILINE)),
            'has_numbers': bool(re.search(r'^\s*\d+\.\s', text, re.MULTILINE)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'line_breaks': text.count('\n'),
        }
    
    def _analyze_sentence_starters(self, text: str) -> Dict[str, int]:
        """Analyze how sentences typically start."""
        sentences = sent_tokenize(text)
        
        starters = Counter()
        for sent in sentences:
            words = word_tokenize(sent)
            if words:
                first_word = words[0].lower()
                # Group by category
                if first_word in ('the', 'a', 'an'):
                    starters['article'] += 1
                elif first_word in ('i', 'we', 'you', 'they', 'he', 'she', 'it'):
                    starters['pronoun'] += 1
                elif first_word in ('but', 'and', 'or', 'so'):
                    starters['conjunction'] += 1
                else:
                    starters['other'] += 1
        
        return dict(starters)