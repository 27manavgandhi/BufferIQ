"""Tests for voice aggregator."""

import pytest
from bufferiq.ml.voice.extraction.aggregator import VoiceAggregator
from bufferiq.ml.voice.linguistic.lexical_analyzer import LexicalMetrics
from bufferiq.ml.voice.linguistic.syntactic_analyzer import SyntacticMetrics
from bufferiq.ml.voice.stylistic.style_detector import StylisticFeatures, WritingStyle


class TestVoiceAggregator:
    """Test voice aggregator."""
    
    @pytest.fixture
    def aggregator(self):
        """Create aggregator instance."""
        return VoiceAggregator()
    
    @pytest.fixture
    def sample_lexical_metrics(self):
        """Create sample lexical metrics."""
        return [
            LexicalMetrics(
                type_token_ratio=0.8,
                hapax_legomena_ratio=0.3,
                average_word_length=5.0,
                vocabulary_size=100,
                unique_words=80,
                word_frequency_dist={"word1": 10, "word2": 8},
                lexical_density=0.6,
                complexity_score=70.0,
            ),
            LexicalMetrics(
                type_token_ratio=0.9,
                hapax_legomena_ratio=0.4,
                average_word_length=6.0,
                vocabulary_size=120,
                unique_words=100,
                word_frequency_dist={"word1": 12, "word3": 10},
                lexical_density=0.7,
                complexity_score=80.0,
            ),
        ]
    
    def test_aggregate_lexical_basic(self, aggregator, sample_lexical_metrics):
        """Test basic lexical aggregation."""
        result = aggregator.aggregate_lexical(sample_lexical_metrics)
        
        assert isinstance(result, LexicalMetrics)
        assert 0.8 <= result.type_token_ratio <= 0.9
        assert result.average_word_length == pytest.approx(5.5)
    
    def test_aggregate_lexical_empty_raises_error(self, aggregator):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            aggregator.aggregate_lexical([])
    
    def test_aggregate_lexical_averages_numerical_values(self, aggregator, sample_lexical_metrics):
        """Test numerical values are averaged."""
        result = aggregator.aggregate_lexical(sample_lexical_metrics)
        
        # Average TTR should be (0.8 + 0.9) / 2 = 0.85
        assert result.type_token_ratio == pytest.approx(0.85)
        
        # Average complexity should be (70 + 80) / 2 = 75
        assert result.complexity_score == pytest.approx(75.0)
    
    def test_aggregate_lexical_combines_word_frequencies(self, aggregator, sample_lexical_metrics):
        """Test word frequencies are combined."""
        result = aggregator.aggregate_lexical(sample_lexical_metrics)
        
        # Should combine frequencies from both
        assert "word1" in result.word_frequency_dist
        assert result.word_frequency_dist["word1"] == 22  # 10 + 12
    
    def test_aggregate_lexical_limits_to_top_n(self, aggregator):
        """Test aggregation limits to top N words."""
        metrics = [
            LexicalMetrics(
                type_token_ratio=0.8,
                hapax_legomena_ratio=0.3,
                average_word_length=5.0,
                vocabulary_size=100,
                unique_words=80,
                word_frequency_dist={f"word{i}": i for i in range(100)},
                lexical_density=0.6,
                complexity_score=70.0,
            )
        ]
        
        result = aggregator.aggregate_lexical(metrics)
        
        # Should limit to top 50
        assert len(result.word_frequency_dist) <= 50
    
    def test_aggregate_syntactic_basic(self, aggregator):
        """Test basic syntactic aggregation."""
        metrics = [
            SyntacticMetrics(
                average_sentence_length=15.0,
                sentence_complexity=0.5,
                pos_distribution={"NOUN": 0.3, "VERB": 0.2},
                dependency_depth=2.0,
                clause_density=1.5,
                syntactic_variety=0.6,
            ),
            SyntacticMetrics(
                average_sentence_length=20.0,
                sentence_complexity=0.6,
                pos_distribution={"NOUN": 0.4, "VERB": 0.3},
                dependency_depth=2.5,
                clause_density=2.0,
                syntactic_variety=0.7,
            ),
        ]
        
        result = aggregator.aggregate_syntactic(metrics)
        
        assert isinstance(result, SyntacticMetrics)
        assert result.average_sentence_length == pytest.approx(17.5)
        assert result.sentence_complexity == pytest.approx(0.55)
    
    def test_aggregate_syntactic_empty_raises_error(self, aggregator):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            aggregator.aggregate_syntactic([])
    
    def test_aggregate_syntactic_combines_pos_distribution(self, aggregator):
        """Test POS distributions are averaged."""
        metrics = [
            SyntacticMetrics(
                average_sentence_length=15.0,
                sentence_complexity=0.5,
                pos_distribution={"NOUN": 0.3, "VERB": 0.2},
                dependency_depth=2.0,
                clause_density=1.5,
                syntactic_variety=0.6,
            ),
            SyntacticMetrics(
                average_sentence_length=20.0,
                sentence_complexity=0.6,
                pos_distribution={"NOUN": 0.5, "VERB": 0.4},
                dependency_depth=2.5,
                clause_density=2.0,
                syntactic_variety=0.7,
            ),
        ]
        
        result = aggregator.aggregate_syntactic(metrics)
        
        # NOUN average: (0.3 + 0.5) / 2 = 0.4
        assert result.pos_distribution["NOUN"] == pytest.approx(0.4)
        # VERB average: (0.2 + 0.4) / 2 = 0.3
        assert result.pos_distribution["VERB"] == pytest.approx(0.3)
    
    def test_aggregate_stylistic_basic(self, aggregator):
        """Test basic stylistic aggregation."""
        features = [
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.8,
                formality_score=80.0,
                punctuation_density={"period": 5.0, "comma": 3.0},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.1,
                exclamation_ratio=0.0,
                average_paragraph_length=3.0,
            ),
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.9,
                formality_score=85.0,
                punctuation_density={"period": 6.0, "comma": 4.0},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.2,
                exclamation_ratio=0.0,
                average_paragraph_length=4.0,
            ),
        ]
        
        result = aggregator.aggregate_stylistic(features)
        
        assert isinstance(result, StylisticFeatures)
        assert result.style == WritingStyle.FORMAL
        assert result.formality_score == pytest.approx(82.5)
    
    def test_aggregate_stylistic_empty_raises_error(self, aggregator):
        """Test empty list raises error."""
        with pytest.raises(ValueError, match="empty"):
            aggregator.aggregate_stylistic([])
    
    def test_aggregate_stylistic_most_common_style(self, aggregator):
        """Test most common style is selected."""
        features = [
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.8,
                formality_score=80.0,
                punctuation_density={},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.0,
                exclamation_ratio=0.0,
                average_paragraph_length=3.0,
            ),
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.9,
                formality_score=85.0,
                punctuation_density={},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.0,
                exclamation_ratio=0.0,
                average_paragraph_length=4.0,
            ),
            StylisticFeatures(
                style=WritingStyle.CASUAL,
                style_confidence=0.7,
                formality_score=40.0,
                punctuation_density={},
                emoji_density=5.0,
                capitalization_pattern="standard",
                contraction_ratio=10.0,
                question_ratio=0.2,
                exclamation_ratio=0.3,
                average_paragraph_length=2.0,
            ),
        ]
        
        result = aggregator.aggregate_stylistic(features)
        
        # FORMAL appears twice, CASUAL once
        assert result.style == WritingStyle.FORMAL
    
    def test_aggregate_stylistic_averages_punctuation(self, aggregator):
        """Test punctuation density is averaged."""
        features = [
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.8,
                formality_score=80.0,
                punctuation_density={"period": 5.0, "comma": 3.0},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.0,
                exclamation_ratio=0.0,
                average_paragraph_length=3.0,
            ),
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.9,
                formality_score=85.0,
                punctuation_density={"period": 7.0, "comma": 5.0},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.0,
                exclamation_ratio=0.0,
                average_paragraph_length=4.0,
            ),
        ]
        
        result = aggregator.aggregate_stylistic(features)
        
        # Period average: (5 + 7) / 2 = 6
        assert result.punctuation_density["period"] == pytest.approx(6.0)
        # Comma average: (3 + 5) / 2 = 4
        assert result.punctuation_density["comma"] == pytest.approx(4.0)
    
    def test_aggregate_stylistic_most_common_capitalization(self, aggregator):
        """Test most common capitalization pattern is selected."""
        features = [
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.8,
                formality_score=80.0,
                punctuation_density={},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.0,
                exclamation_ratio=0.0,
                average_paragraph_length=3.0,
            ),
            StylisticFeatures(
                style=WritingStyle.FORMAL,
                style_confidence=0.9,
                formality_score=85.0,
                punctuation_density={},
                emoji_density=0.0,
                capitalization_pattern="standard",
                contraction_ratio=0.0,
                question_ratio=0.0,
                exclamation_ratio=0.0,
                average_paragraph_length=4.0,
            ),
            StylisticFeatures(
                style=WritingStyle.CASUAL,
                style_confidence=0.7,
                formality_score=40.0,
                punctuation_density={},
                emoji_density=5.0,
                capitalization_pattern="lowercase",
                contraction_ratio=10.0,
                question_ratio=0.2,
                exclamation_ratio=0.3,
                average_paragraph_length=2.0,
            ),
        ]
        
        result = aggregator.aggregate_stylistic(features)
        
        # "standard" appears twice
        assert result.capitalization_pattern == "standard"