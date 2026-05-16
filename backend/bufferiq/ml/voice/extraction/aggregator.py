"""
Voice feature aggregation across posts.

Aggregates voice features from multiple posts
to create representative profiles.
"""

from typing import List, Dict
import statistics

from bufferiq.ml.voice.linguistic.lexical_analyzer import LexicalMetrics
from bufferiq.ml.voice.linguistic.syntactic_analyzer import SyntacticMetrics
from bufferiq.ml.voice.stylistic.style_detector import StylisticFeatures, WritingStyle


class VoiceAggregator:
    """
    Aggregate voice features from multiple analyses.
    
    Creates representative profiles by combining features
    from multiple content samples.
    
    Example:
```python
        aggregator = VoiceAggregator()
        avg_features = aggregator.aggregate_lexical([metrics1, metrics2])
```
    """
    
    def __init__(self):
        """Initialize voice aggregator."""
        pass
    
    def aggregate_lexical(self, metrics_list: List[LexicalMetrics]) -> LexicalMetrics:
        """
        Aggregate lexical metrics.
        
        Args:
            metrics_list: List of lexical metrics
        
        Returns:
            Aggregated lexical metrics
        
        Raises:
            ValueError: If list is empty
        """
        if not metrics_list:
            raise ValueError("Cannot aggregate empty metrics list")
        
        # Average numerical metrics
        avg_ttr = statistics.mean(m.type_token_ratio for m in metrics_list)
        avg_hapax = statistics.mean(m.hapax_legomena_ratio for m in metrics_list)
        avg_word_len = statistics.mean(m.average_word_length for m in metrics_list)
        avg_vocab_size = int(statistics.mean(m.vocabulary_size for m in metrics_list))
        avg_unique = int(statistics.mean(m.unique_words for m in metrics_list))
        avg_density = statistics.mean(m.lexical_density for m in metrics_list)
        avg_complexity = statistics.mean(m.complexity_score for m in metrics_list)
        
        # Combine word frequency distributions
        combined_freq: Dict[str, int] = {}
        for metrics in metrics_list:
            for word, count in metrics.word_frequency_dist.items():
                combined_freq[word] = combined_freq.get(word, 0) + count
        
        # Get top 50 most common
        sorted_freq = sorted(combined_freq.items(), key=lambda x: x[1], reverse=True)
        top_freq = dict(sorted_freq[:50])
        
        return LexicalMetrics(
            type_token_ratio=avg_ttr,
            hapax_legomena_ratio=avg_hapax,
            average_word_length=avg_word_len,
            vocabulary_size=avg_vocab_size,
            unique_words=avg_unique,
            word_frequency_dist=top_freq,
            lexical_density=avg_density,
            complexity_score=avg_complexity,
        )
    
    def aggregate_syntactic(
        self, metrics_list: List[SyntacticMetrics]
    ) -> SyntacticMetrics:
        """
        Aggregate syntactic metrics.
        
        Args:
            metrics_list: List of syntactic metrics
        
        Returns:
            Aggregated syntactic metrics
        
        Raises:
            ValueError: If list is empty
        """
        if not metrics_list:
            raise ValueError("Cannot aggregate empty metrics list")
        
        # Average numerical metrics
        avg_sent_len = statistics.mean(
            m.average_sentence_length for m in metrics_list
        )
        avg_complexity = statistics.mean(m.sentence_complexity for m in metrics_list)
        avg_dep_depth = statistics.mean(m.dependency_depth for m in metrics_list)
        avg_clause_density = statistics.mean(m.clause_density for m in metrics_list)
        avg_variety = statistics.mean(m.syntactic_variety for m in metrics_list)
        
        # Aggregate POS distributions
        combined_pos: Dict[str, List[float]] = {}
        for metrics in metrics_list:
            for pos, ratio in metrics.pos_distribution.items():
                if pos not in combined_pos:
                    combined_pos[pos] = []
                combined_pos[pos].append(ratio)
        
        avg_pos = {pos: statistics.mean(ratios) for pos, ratios in combined_pos.items()}
        
        return SyntacticMetrics(
            average_sentence_length=avg_sent_len,
            sentence_complexity=avg_complexity,
            pos_distribution=avg_pos,
            dependency_depth=avg_dep_depth,
            clause_density=avg_clause_density,
            syntactic_variety=avg_variety,
        )
    
    def aggregate_stylistic(
        self, features_list: List[StylisticFeatures]
    ) -> StylisticFeatures:
        """
        Aggregate stylistic features.
        
        Args:
            features_list: List of stylistic features
        
        Returns:
            Aggregated stylistic features
        
        Raises:
            ValueError: If list is empty
        """
        if not features_list:
            raise ValueError("Cannot aggregate empty features list")
        
        # Find most common style
        from collections import Counter
        style_counts = Counter(f.style for f in features_list)
        most_common_style = style_counts.most_common(1)[0][0]
        style_confidence = style_counts[most_common_style] / len(features_list)
        
        # Average numerical metrics
        avg_formality = statistics.mean(f.formality_score for f in features_list)
        avg_emoji_density = statistics.mean(f.emoji_density for f in features_list)
        avg_contraction = statistics.mean(f.contraction_ratio for f in features_list)
        avg_question = statistics.mean(f.question_ratio for f in features_list)
        avg_exclamation = statistics.mean(f.exclamation_ratio for f in features_list)
        avg_para_len = statistics.mean(f.average_paragraph_length for f in features_list)
        
        # Aggregate punctuation density
        combined_punct: Dict[str, List[float]] = {}
        for features in features_list:
            for punct, density in features.punctuation_density.items():
                if punct not in combined_punct:
                    combined_punct[punct] = []
                combined_punct[punct].append(density)
        
        avg_punct = {
            punct: statistics.mean(densities)
            for punct, densities in combined_punct.items()
        }
        
        # Most common capitalization pattern
        cap_counts = Counter(f.capitalization_pattern for f in features_list)
        most_common_cap = cap_counts.most_common(1)[0][0]
        
        return StylisticFeatures(
            style=most_common_style,
            style_confidence=style_confidence,
            formality_score=avg_formality,
            punctuation_density=avg_punct,
            emoji_density=avg_emoji_density,
            capitalization_pattern=most_common_cap,
            contraction_ratio=avg_contraction,
            question_ratio=avg_question,
            exclamation_ratio=avg_exclamation,
            average_paragraph_length=avg_para_len,
        )