"""Tests for pattern analyzer."""

import pytest
from bufferiq.ml.voice.stylistic.pattern_analyzer import PatternAnalyzer


class TestPatternAnalyzer:
    """Test pattern analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return PatternAnalyzer()
    
    def test_analyze_basic_text(self, analyzer):
        """Test basic pattern analysis."""
        text = "Hello world! How are you?"
        patterns = analyzer.analyze(text)
        
        assert isinstance(patterns, dict)
        assert 'punctuation_patterns' in patterns
        assert 'capitalization_patterns' in patterns
    
    def test_analyze_empty_text_raises_error(self, analyzer):
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("")
    
    def test_punctuation_pattern_counts(self, analyzer):
        """Test punctuation pattern counting."""
        text = "Hello! How are you? I'm fine, thanks; great."
        patterns = analyzer.analyze(text)
        
        punct = patterns['punctuation_patterns']['counts']
        assert punct['exclamation'] == 1
        assert punct['question'] == 1
        assert punct['comma'] == 1
        assert punct['semicolon'] == 1
    
    def test_sentence_ending_patterns(self, analyzer):
        """Test sentence ending pattern detection."""
        text = "Hello! How are you? I'm fine."
        patterns = analyzer.analyze(text)
        
        endings = patterns['punctuation_patterns']['sentence_endings']
        assert endings['exclamation'] == 1
        assert endings['question'] == 1
        assert endings['period'] == 1
    
    def test_capitalization_all_caps_ratio(self, analyzer):
        """Test all caps ratio calculation."""
        text = "THIS IS ALL CAPS TEXT"
        patterns = analyzer.analyze(text)
        
        cap = patterns['capitalization_patterns']
        assert cap['all_caps_ratio'] > 0.8
    
    def test_capitalization_title_case_ratio(self, analyzer):
        """Test title case ratio calculation."""
        text = "This Is Title Case Text"
        patterns = analyzer.analyze(text)
        
        cap = patterns['capitalization_patterns']
        assert cap['title_case_ratio'] > 0.7
    
    def test_capitalization_lowercase_ratio(self, analyzer):
        """Test lowercase ratio calculation."""
        text = "this is mostly lowercase text"
        patterns = analyzer.analyze(text)
        
        cap = patterns['capitalization_patterns']
        assert cap['lower_case_ratio'] > 0.7
    
    def test_formatting_bullets_detection(self, analyzer):
        """Test bullet point detection."""
        text = """
        - First item
        - Second item
        - Third item
        """
        patterns = analyzer.analyze(text)
        
        assert patterns['formatting_patterns']['has_bullets'] is True
    
    def test_formatting_numbered_list_detection(self, analyzer):
        """Test numbered list detection."""
        text = """
        1. First item
        2. Second item
        3. Third item
        """
        patterns = analyzer.analyze(text)
        
        assert patterns['formatting_patterns']['has_numbers'] is True
    
    def test_formatting_paragraph_count(self, analyzer):
        """Test paragraph counting."""
        text = """
        First paragraph.
        
        Second paragraph.
        
        Third paragraph.
        """
        patterns = analyzer.analyze(text)
        
        assert patterns['formatting_patterns']['paragraph_count'] == 3
    
    def test_formatting_line_breaks(self, analyzer):
        """Test line break counting."""
        text = "Line 1\nLine 2\nLine 3"
        patterns = analyzer.analyze(text)
        
        assert patterns['formatting_patterns']['line_breaks'] == 2
    
    def test_sentence_starters_article(self, analyzer):
        """Test article sentence starters."""
        text = "The cat sat. A dog ran. An apple fell."
        patterns = analyzer.analyze(text)
        
        starters = patterns['sentence_starters']
        assert starters.get('article', 0) == 3
    
    def test_sentence_starters_pronoun(self, analyzer):
        """Test pronoun sentence starters."""
        text = "I went home. We stayed late. They left early."
        patterns = analyzer.analyze(text)
        
        starters = patterns['sentence_starters']
        assert starters.get('pronoun', 0) == 3
    
    def test_sentence_starters_conjunction(self, analyzer):
        """Test conjunction sentence starters."""
        text = "But wait! And there's more! So what now?"
        patterns = analyzer.analyze(text)
        
        starters = patterns['sentence_starters']
        assert starters.get('conjunction', 0) == 3
    
    def test_comma_per_sentence(self, analyzer):
        """Test comma per sentence calculation."""
        text = "First, second, third. Fourth, fifth."
        patterns = analyzer.analyze(text)
        
        comma_per_sent = patterns['punctuation_patterns']['comma_per_sentence']
        assert comma_per_sent > 1.0
    
    def test_no_special_formatting(self, analyzer):
        """Test text without special formatting."""
        text = "This is plain text without any special formatting."
        patterns = analyzer.analyze(text)
        
        assert patterns['formatting_patterns']['has_bullets'] is False
        assert patterns['formatting_patterns']['has_numbers'] is False
    
    def test_mixed_capitalization(self, analyzer):
        """Test mixed capitalization patterns."""
        text = "Some WORDS are UPPERCASE and some are lowercase"
        patterns = analyzer.analyze(text)
        
        cap = patterns['capitalization_patterns']
        # Should have mix of different cases
        assert cap['all_caps_ratio'] > 0
        assert cap['lower_case_ratio'] > 0
    
    def test_heavy_punctuation(self, analyzer):
        """Test heavy punctuation usage."""
        text = "Really!!! Amazing??? Wow... Great, awesome; fantastic: wonderful."
        patterns = analyzer.analyze(text)
        
        counts = patterns['punctuation_patterns']['counts']
        assert counts['exclamation'] > 2
        assert counts['question'] > 2
    
    def test_minimal_punctuation(self, analyzer):
        """Test minimal punctuation usage."""
        text = "Simple sentence without much punctuation"
        patterns = analyzer.analyze(text)
        
        counts = patterns['punctuation_patterns']['counts']
        assert counts['exclamation'] == 0
        assert counts['question'] == 0
        assert counts['semicolon'] == 0
    
    def test_paragraph_detection_single_paragraph(self, analyzer):
        """Test single paragraph detection."""
        text = "This is one continuous paragraph without breaks."
        patterns = analyzer.analyze(text)
        
        assert patterns['formatting_patterns']['paragraph_count'] == 1