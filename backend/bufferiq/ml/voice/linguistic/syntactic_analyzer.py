"""
Syntactic complexity and structure analysis.

Measures sentence structure, grammatical patterns,
and syntactic variety for voice characterization.
"""

from dataclasses import dataclass
from typing import Dict, List
import re

import nltk
from nltk import pos_tag, word_tokenize, sent_tokenize
from nltk.corpus import stopwords


@dataclass
class SyntacticMetrics:
    """Syntactic complexity metrics."""
    
    average_sentence_length: float
    sentence_complexity: float  # Dependent clauses ratio
    pos_distribution: Dict[str, float]  # Part-of-speech distribution
    dependency_depth: float
    clause_density: float
    syntactic_variety: float  # Structure variation


class SyntacticAnalyzer:
    """
    Analyze syntactic complexity and structure.
    
    Measures sentence structure, grammatical patterns,
    and syntactic variety for voice characterization.
    
    Example:
```python
        analyzer = SyntacticAnalyzer()
        metrics = analyzer.analyze(text)
        print(f"Avg sentence length: {metrics.average_sentence_length:.1f}")
        print(f"Complexity: {metrics.sentence_complexity:.2f}")
```
    """
    
    def __init__(self):
        """Initialize syntactic analyzer."""
        pass
    
    def analyze(self, text: str) -> SyntacticMetrics:
        """
        Analyze syntactic characteristics.
        
        Args:
            text: Text to analyze
        
        Returns:
            Syntactic metrics
        
        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Text too short for syntactic analysis")
        
        # Tokenize sentences and words
        sentences = sent_tokenize(text)
        if not sentences:
            raise ValueError("No sentences found in text")
        
        words = word_tokenize(text)
        
        # Calculate metrics
        avg_sent_length = len(words) / len(sentences)
        sent_complexity = self._calculate_sentence_complexity(sentences)
        pos_dist = self._calculate_pos_distribution(words)
        dep_depth = self._estimate_dependency_depth(sentences)
        clause_density = self._calculate_clause_density(sentences)
        syntactic_variety = self._calculate_syntactic_variety(sentences)
        
        return SyntacticMetrics(
            average_sentence_length=avg_sent_length,
            sentence_complexity=sent_complexity,
            pos_distribution=pos_dist,
            dependency_depth=dep_depth,
            clause_density=clause_density,
            syntactic_variety=syntactic_variety,
        )
    
    def _calculate_sentence_complexity(self, sentences: List[str]) -> float:
        """
        Calculate sentence complexity based on conjunctions and clauses.
        
        Args:
            sentences: List of sentences
        
        Returns:
            Complexity score (0-1)
        """
        if not sentences:
            return 0.0
        
        # Count subordinating conjunctions and relative pronouns
        complexity_markers = [
            'although', 'because', 'since', 'unless', 'while', 'whereas',
            'that', 'which', 'who', 'whom', 'whose', 'when', 'where'
        ]
        
        total_markers = 0
        for sent in sentences:
            words = sent.lower().split()
            total_markers += sum(1 for word in words if word in complexity_markers)
        
        # Normalize by sentence count
        return min(total_markers / len(sentences), 1.0)
    
    def _calculate_pos_distribution(self, words: List[str]) -> Dict[str, float]:
        """
        Calculate part-of-speech distribution.
        
        Args:
            words: Word tokens
        
        Returns:
            POS distribution dictionary
        """
        if not words:
            return {}
        
        pos_tags = pos_tag(words)
        
        # Group into major categories
        pos_counts: Dict[str, int] = {}
        for _, tag in pos_tags:
            # Simplify tag to major category
            if tag.startswith('NN'):
                category = 'NOUN'
            elif tag.startswith('VB'):
                category = 'VERB'
            elif tag.startswith('JJ'):
                category = 'ADJ'
            elif tag.startswith('RB'):
                category = 'ADV'
            elif tag in ('DT', 'PDT', 'WDT'):
                category = 'DET'
            elif tag in ('IN', 'TO'):
                category = 'PREP'
            elif tag in ('CC',):
                category = 'CONJ'
            elif tag.startswith('PR'):
                category = 'PRON'
            else:
                category = 'OTHER'
            
            pos_counts[category] = pos_counts.get(category, 0) + 1
        
        # Convert to percentages
        total = len(pos_tags)
        return {cat: count / total for cat, count in pos_counts.items()}
    
    def _estimate_dependency_depth(self, sentences: List[str]) -> float:
        """
        Estimate average dependency depth (simplified).
        
        Args:
            sentences: List of sentences
        
        Returns:
            Estimated depth score
        """
        if not sentences:
            return 0.0
        
        # Simple heuristic: count nested structures
        total_depth = 0
        for sent in sentences:
            # Count commas and conjunctions as indicators of depth
            depth = sent.count(',') + sent.count(' and ') + sent.count(' or ')
            total_depth += depth
        
        return total_depth / len(sentences)
    
    def _calculate_clause_density(self, sentences: List[str]) -> float:
        """
        Calculate clause density (clauses per sentence).
        
        Args:
            sentences: List of sentences
        
        Returns:
            Clause density
        """
        if not sentences:
            return 0.0
        
        total_clauses = 0
        for sent in sentences:
            # Count clause indicators
            clauses = 1  # Main clause
            clauses += sent.count(',')
            clauses += sent.count(';')
            clauses += sent.lower().count(' and ')
            clauses += sent.lower().count(' but ')
            clauses += sent.lower().count(' or ')
            total_clauses += clauses
        
        return total_clauses / len(sentences)
    
    def _calculate_syntactic_variety(self, sentences: List[str]) -> float:
        """
        Calculate syntactic variety (structure variation).
        
        Args:
            sentences: List of sentences
        
        Returns:
            Variety score (0-1)
        """
        if len(sentences) < 2:
            return 0.0
        
        # Simplified: measure variation in sentence length
        lengths = [len(sent.split()) for sent in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        # Calculate coefficient of variation
        if avg_length == 0:
            return 0.0
        
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        std_dev = variance ** 0.5
        cv = std_dev / avg_length
        
        # Normalize to 0-1 range
        return min(cv, 1.0)