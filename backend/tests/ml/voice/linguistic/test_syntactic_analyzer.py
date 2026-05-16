"""Tests for syntactic analyzer."""

import pytest
from bufferiq.ml.voice.linguistic.syntactic_analyzer import SyntacticAnalyzer, SyntacticMetrics


class TestSyntacticAnalyzer:
    """Test syntactic analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SyntacticAnalyzer()
    
    def test_analyze_basic_text(self, analyzer):
        """Test basic text analysis."""
        text = "The cat sat on the mat. The dog ran in the park."
        metrics = analyzer.analyze(text)
        
        assert isinstance(metrics, SyntacticMetrics)
        assert metrics.average_sentence_length > 0
        assert len(metrics.pos_distribution) > 0
    
    def test_analyze_empty_text_raises_error(self, analyzer):
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("")
    
    def test_analyze_short_text_raises_error(self, analyzer):
        """Test short text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("Hi")
    
    def test_simple_sentences(self, analyzer):
        """Test analysis of simple sentences."""
        text = "I like cats. Cats are cute. They sleep a lot."
        metrics = analyzer.analyze(text)
        
        # Simple sentences have low complexity
        assert metrics.sentence_complexity < 0.3
        assert metrics.average_sentence_length < 10
    
    def test_complex_sentences(self, analyzer):
        """Test analysis of complex sentences."""
        text = """
        Although it was raining, we decided to go hiking because
        we had been planning this trip for months and we didn't
        want to cancel it.
        """
        metrics = analyzer.analyze(text)
        
        # Complex sentences have higher complexity
        assert metrics.sentence_complexity > 0.3
    
    def test_pos_distribution(self, analyzer):
        """Test POS distribution calculation."""
        text = "The quick brown fox jumps over the lazy dog."
        metrics = analyzer.analyze(text)
        
        # Should have various POS tags
        assert 'NOUN' in metrics.pos_distribution
        assert 'DET' in metrics.pos_distribution
        assert sum(metrics.pos_distribution.values()) == pytest.approx(1.0)
    
    def test_average_sentence_length(self, analyzer):
        """Test average sentence length calculation."""
        text = "Short. A bit longer now. This is even longer than before."
        metrics = analyzer.analyze(text)
        
        assert metrics.average_sentence_length > 0
        assert metrics.average_sentence_length < 20
    
    def test_clause_density(self, analyzer):
        """Test clause density calculation."""
        text = "I went to the store, bought some milk, and came home."
        metrics = analyzer.analyze(text)
        
        # Multiple clauses
        assert metrics.clause_density > 1.0
    
    def test_syntactic_variety(self, analyzer):
        """Test syntactic variety calculation."""
        text = """
        Short sentence. This is a bit longer. 
        Now we have a much longer sentence with more words.
        """
        metrics = analyzer.analyze(text)
        
        # Varied sentence lengths
        assert metrics.syntactic_variety > 0
    
    def test_no_syntactic_variety(self, analyzer):
        """Test no syntactic variety."""
        text = "I like cats. I like dogs. I like fish."
        metrics = analyzer.analyze(text)
        
        # Similar sentence lengths = low variety
        assert metrics.syntactic_variety < 0.3
    
    def test_dependency_depth_estimation(self, analyzer):
        """Test dependency depth estimation."""
        text = "The cat, which was black, sat on the mat, which was red."
        metrics = analyzer.analyze(text)
        
        # Nested structures increase depth
        assert metrics.dependency_depth > 1.0
    
    def test_single_sentence(self, analyzer):
        """Test analysis of single sentence."""
        text = "This is just one sentence."
        metrics = analyzer.analyze(text)
        
        assert metrics.average_sentence_length > 0
        assert metrics.syntactic_variety == 0  # Only one sentence
    
    def test_multiple_sentences_same_length(self, analyzer):
        """Test multiple sentences of same length."""
        text = "I like cats. I like dogs. I like fish."
        metrics = analyzer.analyze(text)
        
        # All sentences same length
        assert metrics.syntactic_variety < 0.2
    
    def test_pos_distribution_nouns(self, analyzer):
        """Test POS distribution for noun-heavy text."""
        text = "Technology innovation business transformation leadership management"
        metrics = analyzer.analyze(text)
        
        assert 'NOUN' in metrics.pos_distribution
        assert metrics.pos_distribution['NOUN'] > 0.5
    
    def test_pos_distribution_verbs(self, analyzer):
        """Test POS distribution for verb-heavy text."""
        text = "Running jumping swimming dancing singing playing"
        metrics = analyzer.analyze(text)
        
        assert 'VERB' in metrics.pos_distribution
        assert metrics.pos_distribution['VERB'] > 0.5
    
    def test_complexity_with_conjunctions(self, analyzer):
        """Test complexity increases with conjunctions."""
        simple = "I went to the store. I bought milk."
        complex_text = "I went to the store and bought milk because we were out."
        
        simple_metrics = analyzer.analyze(simple)
        complex_metrics = analyzer.analyze(complex_text)
        
        assert complex_metrics.sentence_complexity > simple_metrics.sentence_complexity
    
    def test_clause_density_simple(self, analyzer):
        """Test clause density for simple sentences."""
        text = "The cat sat. The dog ran."
        metrics = analyzer.analyze(text)
        
        # Simple sentences = low clause density
        assert metrics.clause_density < 2.0
    
    def test_clause_density_complex(self, analyzer):
        """Test clause density for complex sentences."""
        text = "I went to the store, bought milk, picked up bread, and came home."
        metrics = analyzer.analyze(text)
        
        # Multiple clauses = high density
        assert metrics.clause_density > 3.0
    
    def test_metrics_consistency(self, analyzer):
        """Test metrics are consistent."""
        text = "The quick brown fox jumps over the lazy dog."
        
        metrics1 = analyzer.analyze(text)
        metrics2 = analyzer.analyze(text)
        
        assert metrics1.average_sentence_length == metrics2.average_sentence_length
        assert metrics1.sentence_complexity == metrics2.sentence_complexity
    
    def test_very_long_sentence(self, analyzer):
        """Test very long sentence."""
        text = "This is a very long sentence with many words " * 10 + "."
        metrics = analyzer.analyze(text)
        
        assert metrics.average_sentence_length > 50
    
    def test_many_short_sentences(self, analyzer):
        """Test many short sentences."""
        text = ". ".join(["Short"] * 20) + "."
        metrics = analyzer.analyze(text)
        
        assert metrics.average_sentence_length < 3
        assert metrics.syntactic_variety < 0.1