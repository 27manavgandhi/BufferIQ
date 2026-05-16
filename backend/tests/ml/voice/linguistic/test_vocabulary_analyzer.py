"""Tests for vocabulary analyzer."""

import pytest
from bufferiq.ml.voice.linguistic.vocabulary_analyzer import VocabularyAnalyzer


class TestVocabularyAnalyzer:
    """Test vocabulary analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return VocabularyAnalyzer()
    
    def test_create_fingerprint_basic(self, analyzer):
        """Test basic fingerprint creation."""
        texts = [
            "The quick brown fox jumps",
            "The lazy dog sleeps",
            "The quick cat runs"
        ]
        
        fingerprint = analyzer.create_fingerprint(texts, top_n=10)
        
        assert isinstance(fingerprint, dict)
        assert len(fingerprint) > 0
        assert "quick" in fingerprint
        assert all(0 <= v <= 1.0 for v in fingerprint.values())
    
    def test_create_fingerprint_empty_texts_raises_error(self, analyzer):
        """Test empty texts raises error."""
        with pytest.raises(ValueError, match="empty"):
            analyzer.create_fingerprint([])
    
    def test_create_fingerprint_frequencies_sum_to_one(self, analyzer):
        """Test fingerprint frequencies sum to 1.0."""
        texts = ["word1 word2 word3", "word4 word5 word6"]
        fingerprint = analyzer.create_fingerprint(texts, top_n=10)
        
        total = sum(fingerprint.values())
        assert total == pytest.approx(1.0, rel=0.01)
    
    def test_compare_fingerprints_identical(self, analyzer):
        """Test comparing identical fingerprints."""
        texts = ["cat dog bird", "cat dog bird"]
        fp1 = analyzer.create_fingerprint(texts, top_n=5)
        fp2 = analyzer.create_fingerprint(texts, top_n=5)
        
        similarity = analyzer.compare_fingerprints(fp1, fp2)
        assert similarity == pytest.approx(1.0, rel=0.01)
    
    def test_compare_fingerprints_different(self, analyzer):
        """Test comparing different fingerprints."""
        texts1 = ["technology innovation digital"]
        texts2 = ["nature wildlife forest"]
        
        fp1 = analyzer.create_fingerprint(texts1, top_n=5)
        fp2 = analyzer.create_fingerprint(texts2, top_n=5)
        
        similarity = analyzer.compare_fingerprints(fp1, fp2)
        assert similarity < 0.3
    
    def test_compare_fingerprints_similar(self, analyzer):
        """Test comparing similar fingerprints."""
        texts1 = ["technology innovation digital software"]
        texts2 = ["technology digital innovation coding"]
        
        fp1 = analyzer.create_fingerprint(texts1, top_n=10)
        fp2 = analyzer.create_fingerprint(texts2, top_n=10)
        
        similarity = analyzer.compare_fingerprints(fp1, fp2)
        assert similarity > 0.5
    
    def test_compare_fingerprints_empty_returns_zero(self, analyzer):
        """Test comparing with empty fingerprint."""
        fp1 = {"word1": 0.5, "word2": 0.5}
        fp2 = {}
        
        similarity = analyzer.compare_fingerprints(fp1, fp2)
        assert similarity == 0.0
    
    def test_get_distinctive_words(self, analyzer):
        """Test getting distinctive words."""
        texts_target = ["unique special rare exclusive"]
        texts_baseline = ["common usual typical regular"]
        
        fp_target = analyzer.create_fingerprint(texts_target, top_n=10)
        fp_baseline = analyzer.create_fingerprint(texts_baseline, top_n=10)
        
        distinctive = analyzer.get_distinctive_words(fp_target, fp_baseline, top_n=5)
        
        assert isinstance(distinctive, list)
        assert len(distinctive) > 0
        assert all(isinstance(item, tuple) for item in distinctive)
    
    def test_fingerprint_top_n_limit(self, analyzer):
        """Test top_n parameter limits fingerprint size."""
        texts = [" ".join([f"word{i}" for i in range(100)])]
        
        fingerprint = analyzer.create_fingerprint(texts, top_n=10)
        assert len(fingerprint) <= 10
    
    def test_fingerprint_filters_stop_words(self, analyzer):
        """Test fingerprint filters stop words."""
        texts = ["the and or but if then when"]
        fingerprint = analyzer.create_fingerprint(texts, top_n=10)
        
        # Stop words should be filtered
        assert len(fingerprint) == 0 or all(
            word not in ["the", "and", "or", "but", "if", "then", "when"]
            for word in fingerprint.keys()
        )
    
    def test_fingerprint_filters_short_words(self, analyzer):
        """Test fingerprint filters short words."""
        texts = ["a I at to be or if cat dog"]
        fingerprint = analyzer.create_fingerprint(texts, top_n=10)
        
        # All words should be longer than 2 characters
        assert all(len(word) > 2 for word in fingerprint.keys())
    
    def test_distinctive_words_order(self, analyzer):
        """Test distinctive words are ordered by distinctiveness."""
        texts_target = ["apple banana cherry"] * 5 + ["apple"] * 10
        texts_baseline = ["apple banana cherry"]
        
        fp_target = analyzer.create_fingerprint(texts_target, top_n=10)
        fp_baseline = analyzer.create_fingerprint(texts_baseline, top_n=10)
        
        distinctive = analyzer.get_distinctive_words(fp_target, fp_baseline, top_n=5)
        
        # Should be ordered by descending distinctiveness
        if len(distinctive) > 1:
            for i in range(len(distinctive) - 1):
                assert distinctive[i][1] >= distinctive[i + 1][1]
    
    def test_compare_fingerprints_partial_overlap(self, analyzer):
        """Test comparing fingerprints with partial overlap."""
        texts1 = ["shared word1 word2"]
        texts2 = ["shared word3 word4"]
        
        fp1 = analyzer.create_fingerprint(texts1, top_n=10)
        fp2 = analyzer.create_fingerprint(texts2, top_n=10)
        
        similarity = analyzer.compare_fingerprints(fp1, fp2)
        assert 0 < similarity < 1.0
    
    def test_fingerprint_normalized_frequencies(self, analyzer):
        """Test fingerprint frequencies are normalized."""
        texts = ["word1"] * 10 + ["word2"] * 5 + ["word3"] * 2
        fingerprint = analyzer.create_fingerprint([" ".join(texts)], top_n=10)
        
        # Frequencies should reflect relative proportions
        assert fingerprint.get("word1", 0) > fingerprint.get("word2", 0)
        assert fingerprint.get("word2", 0) > fingerprint.get("word3", 0)
    
    def test_create_fingerprint_handles_duplicates(self, analyzer):
        """Test fingerprint handles duplicate texts correctly."""
        texts = [
            "cat dog bird",
            "cat dog bird",
            "cat dog bird"
        ]
        fingerprint = analyzer.create_fingerprint(texts, top_n=10)
        
        # Should still create valid fingerprint
        assert len(fingerprint) > 0
        assert sum(fingerprint.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_fingerprint_case_insensitive(self, analyzer):
        """Test fingerprint is case insensitive."""
        texts1 = ["Technology Innovation"]
        texts2 = ["technology innovation"]
        
        fp1 = analyzer.create_fingerprint(texts1, top_n=10)
        fp2 = analyzer.create_fingerprint(texts2, top_n=10)
        
        similarity = analyzer.compare_fingerprints(fp1, fp2)
        assert similarity == pytest.approx(1.0, rel=0.01)
    
    def test_distinctive_words_negative_distinctiveness(self, analyzer):
        """Test distinctive words can have negative distinctiveness."""
        texts_target = ["word1"] * 2
        texts_baseline = ["word1"] * 10 + ["word2"] * 10
        
        fp_target = analyzer.create_fingerprint([" ".join(texts_target)], top_n=10)
        fp_baseline = analyzer.create_fingerprint([" ".join(texts_baseline)], top_n=10)
        
        distinctive = analyzer.get_distinctive_words(fp_target, fp_baseline, top_n=10)
        
        # Some words may be less distinctive (negative values)
        assert any(score < 0 for _, score in distinctive)