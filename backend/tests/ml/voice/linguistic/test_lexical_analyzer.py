"""Tests for lexical analyzer."""

import pytest
from bufferiq.ml.voice.linguistic.lexical_analyzer import LexicalAnalyzer, LexicalMetrics


class TestLexicalAnalyzer:
    """Test lexical analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return LexicalAnalyzer()
    
    def test_analyze_basic_text(self, analyzer):
        """Test basic text analysis."""
        text = "The quick brown fox jumps over the lazy dog"
        metrics = analyzer.analyze(text)
        
        assert isinstance(metrics, LexicalMetrics)
        assert 0 < metrics.type_token_ratio <= 1.0
        assert metrics.vocabulary_size > 0
        assert metrics.average_word_length > 0
    
    def test_analyze_empty_text_raises_error(self, analyzer):
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("")
    
    def test_analyze_short_text_raises_error(self, analyzer):
        """Test short text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("hi")
    
    def test_calculate_ttr_perfect_diversity(self, analyzer):
        """Test TTR with perfect diversity."""
        tokens = ["the", "quick", "brown", "fox"]
        ttr = analyzer.calculate_ttr(tokens)
        assert ttr == 1.0
    
    def test_calculate_ttr_no_diversity(self, analyzer):
        """Test TTR with no diversity."""
        tokens = ["the", "the", "the", "the"]
        ttr = analyzer.calculate_ttr(tokens)
        assert ttr == 0.25
    
    def test_calculate_ttr_empty_tokens(self, analyzer):
        """Test TTR with empty tokens."""
        ttr = analyzer.calculate_ttr([])
        assert ttr == 0.0
    
    def test_calculate_lexical_density(self, analyzer):
        """Test lexical density calculation."""
        from nltk import pos_tag
        tokens = ["the", "quick", "brown", "fox", "jumps"]
        pos_tags = pos_tag(tokens)
        
        density = analyzer.calculate_lexical_density(tokens, pos_tags)
        assert 0 <= density <= 1.0
    
    def test_analyze_complex_text(self, analyzer):
        """Test analysis of complex text."""
        text = """
        Furthermore, we would like to acknowledge the significant contributions
        made by our research team in developing this innovative methodology.
        """
        metrics = analyzer.analyze(text)
        
        assert metrics.complexity_score > 50  # Complex text
        assert metrics.average_word_length > 5
    
    def test_analyze_simple_text(self, analyzer):
        """Test analysis of simple text."""
        text = "The cat sat on the mat. It was a big cat."
        metrics = analyzer.analyze(text)
        
        assert metrics.complexity_score < 60  # Simple text
    
    def test_word_frequency_distribution(self, analyzer):
        """Test word frequency distribution."""
        text = "the cat sat on the mat the cat was fat"
        metrics = analyzer.analyze(text)
        
        assert "cat" in metrics.word_frequency_dist
        assert "sat" in metrics.word_frequency_dist
        assert len(metrics.word_frequency_dist) > 0
    
    def test_hapax_legomena_ratio(self, analyzer):
        """Test hapax legomena calculation."""
        text = "the cat sat on the mat. the cat was fat and the mat was red"
        metrics = analyzer.analyze(text)
        
        # Some words appear only once
        assert 0 < metrics.hapax_legomena_ratio < 1.0
    
    def test_vocabulary_size_tracking(self, analyzer):
        """Test vocabulary size is tracked correctly."""
        text = "apple banana cherry apple banana cherry"
        metrics = analyzer.analyze(text)
        
        assert metrics.vocabulary_size == 3
        assert metrics.unique_words == 3
    
    def test_lexical_density_content_heavy(self, analyzer):
        """Test lexical density for content-heavy text."""
        text = "Technology innovation drives business transformation rapidly"
        metrics = analyzer.analyze(text)
        
        # Content words dominate
        assert metrics.lexical_density > 0.7
    
    def test_lexical_density_function_heavy(self, analyzer):
        """Test lexical density for function word heavy text."""
        text = "it is what it is and it will be what it will be"
        metrics = analyzer.analyze(text)
        
        # Function words dominate
        assert metrics.lexical_density < 0.5
    
    def test_complexity_score_range(self, analyzer):
        """Test complexity score is in valid range."""
        text = "The quick brown fox jumps over the lazy dog repeatedly"
        metrics = analyzer.analyze(text)
        
        assert 0 <= metrics.complexity_score <= 100
    
    def test_analyze_with_punctuation(self, analyzer):
        """Test analysis handles punctuation correctly."""
        text = "Hello! How are you? I'm doing great!!!"
        metrics = analyzer.analyze(text)
        
        # Should still work despite punctuation
        assert metrics.vocabulary_size > 0
        assert metrics.type_token_ratio > 0
    
    def test_analyze_mixed_case(self, analyzer):
        """Test analysis handles mixed case."""
        text = "The QUICK brown Fox JUMPS over the LAZY dog"
        metrics = analyzer.analyze(text)
        
        # Should normalize case
        assert metrics.type_token_ratio > 0.8  # Mostly unique words
    
    def test_metrics_consistency(self, analyzer):
        """Test metrics are consistent across calls."""
        text = "consistent text for testing purposes only"
        
        metrics1 = analyzer.analyze(text)
        metrics2 = analyzer.analyze(text)
        
        assert metrics1.type_token_ratio == metrics2.type_token_ratio
        assert metrics1.complexity_score == metrics2.complexity_score
    
    def test_long_text_analysis(self, analyzer):
        """Test analysis of longer text."""
        text = " ".join(["word" + str(i) for i in range(100)])
        metrics = analyzer.analyze(text)
        
        assert metrics.vocabulary_size == 100
        assert metrics.type_token_ratio == 1.0  # All unique
    
    def test_repeated_words_analysis(self, analyzer):
        """Test analysis with many repeated words."""
        text = " ".join(["the"] * 50 + ["cat"] * 30 + ["sat"] * 20)
        metrics = analyzer.analyze(text)
        
        assert metrics.vocabulary_size == 3
        assert metrics.type_token_ratio == 3/100
    
    def test_average_word_length_calculation(self, analyzer):
        """Test average word length is calculated correctly."""
        text = "a an the and but or"
        metrics = analyzer.analyze(text)
        
        # Short words
        assert metrics.average_word_length < 3
        
        text2 = "extraordinary phenomenal magnificent"
        metrics2 = analyzer.analyze(text2)
        
        # Long words
        assert metrics2.average_word_length > 10
    
    def test_tokenization_accuracy(self, analyzer):
        """Test tokenization is accurate."""
        text = "don't can't won't it's"
        metrics = analyzer.analyze(text)
        
        # Should handle contractions
        assert metrics.vocabulary_size > 0
    
    def test_special_characters_handling(self, analyzer):
        """Test handling of special characters."""
        text = "hello@world.com test#hashtag $price €euro"
        metrics = analyzer.analyze(text)
        
        # Should filter out non-alphabetic tokens
        assert all(word.isalpha() for word in metrics.word_frequency_dist.keys())
    
    def test_minimum_token_length(self, analyzer):
        """Test tokens must be minimum length."""
        text = "a I to be or not to be that is the question"
        metrics = analyzer.analyze(text)
        
        # Single letter words should be filtered
        assert all(len(word) > 1 for word in metrics.word_frequency_dist.keys())