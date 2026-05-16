"""Tests for style detector."""

import pytest
from bufferiq.ml.voice.stylistic.style_detector import (
    StyleDetector,
    StylisticFeatures,
    WritingStyle,
)


class TestStyleDetector:
    """Test style detector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance."""
        return StyleDetector()
    
    def test_detect_basic_text(self, detector):
        """Test basic style detection."""
        text = "Check out our new product! 🚀"
        features = detector.detect(text)
        
        assert isinstance(features, StylisticFeatures)
        assert isinstance(features.style, WritingStyle)
        assert 0 <= features.formality_score <= 100
    
    def test_detect_empty_text_raises_error(self, detector):
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="too short"):
            detector.detect("")
    
    def test_detect_formal_style(self, detector):
        """Test formal style detection."""
        text = """
        Furthermore, we would like to acknowledge the significant
        contributions made by our research team. Therefore, we shall
        proceed with the implementation accordingly.
        """
        features = detector.detect(text)
        
        assert features.style == WritingStyle.FORMAL
        assert features.formality_score > 70
    
    def test_detect_casual_style(self, detector):
        """Test casual style detection."""
        text = "Hey! Check this out! It's awesome! 😊"
        features = detector.detect(text)
        
        assert features.style == WritingStyle.CASUAL
        assert features.formality_score < 50
    
    def test_detect_technical_style(self, detector):
        """Test technical style detection."""
        text = """
        The algorithm implements a binary search methodology with
        O(log n) complexity. The optimization framework leverages
        dynamic programming techniques.
        """
        features = detector.detect(text)
        
        assert features.style == WritingStyle.TECHNICAL
    
    def test_formality_score_with_formal_indicators(self, detector):
        """Test formality score increases with formal indicators."""
        text = "Furthermore, moreover, consequently, we shall proceed."
        features = detector.detect(text)
        
        assert features.formality_score > 60
    
    def test_formality_score_with_casual_indicators(self, detector):
        """Test formality score decreases with casual indicators."""
        text = "Yeah, gonna check it out, kinda cool!"
        features = detector.detect(text)
        
        assert features.formality_score < 50
    
    def test_emoji_density_calculation(self, detector):
        """Test emoji density calculation."""
        text = "Great! 🎉 Awesome! 🚀 Amazing! 💯"
        features = detector.detect(text)
        
        assert features.emoji_density > 0
    
    def test_no_emoji_density(self, detector):
        """Test no emoji density."""
        text = "This is a serious business document without any emojis."
        features = detector.detect(text)
        
        assert features.emoji_density == 0
    
    def test_punctuation_density(self, detector):
        """Test punctuation density calculation."""
        text = "Hello! How are you? I'm doing great!!!"
        features = detector.detect(text)
        
        assert features.punctuation_density['exclamation'] > 0
        assert features.punctuation_density['question'] > 0
    
    def test_capitalization_pattern_all_caps(self, detector):
        """Test all caps capitalization pattern."""
        text = "THIS IS ALL CAPS TEXT FOR EMPHASIS"
        features = detector.detect(text)
        
        assert features.capitalization_pattern == "all_caps"
    
    def test_capitalization_pattern_title_case(self, detector):
        """Test title case capitalization pattern."""
        text = "This Is Title Case Text For Headlines"
        features = detector.detect(text)
        
        assert features.capitalization_pattern == "title"
    
    def test_capitalization_pattern_standard(self, detector):
        """Test standard capitalization pattern."""
        text = "This is standard sentence case text."
        features = detector.detect(text)
        
        assert features.capitalization_pattern in ["standard", "lowercase"]
    
    def test_contraction_ratio_with_contractions(self, detector):
        """Test contraction ratio calculation."""
        text = "I can't believe it's already time. We're gonna make it!"
        features = detector.detect(text)
        
        assert features.contraction_ratio > 0
    
    def test_contraction_ratio_without_contractions(self, detector):
        """Test contraction ratio without contractions."""
        text = "I cannot believe it is already time. We are going to make it."
        features = detector.detect(text)
        
        assert features.contraction_ratio == 0
    
    def test_question_ratio(self, detector):
        """Test question ratio calculation."""
        text = "What is this? How does it work? Why is it important?"
        features = detector.detect(text)
        
        assert features.question_ratio > 0.5
    
    def test_exclamation_ratio(self, detector):
        """Test exclamation ratio calculation."""
        text = "Amazing! Fantastic! Incredible! Wow!"
        features = detector.detect(text)
        
        assert features.exclamation_ratio > 0.5
    
    def test_average_paragraph_length(self, detector):
        """Test average paragraph length calculation."""
        text = """
        First paragraph with several sentences. This is sentence two.
        
        Second paragraph also with multiple sentences. Another one here.
        """
        features = detector.detect(text)
        
        assert features.average_paragraph_length > 1
    
    def test_professional_style_detection(self, detector):
        """Test professional style detection."""
        text = """
        We are pleased to announce our quarterly results.
        The team has achieved significant milestones this period.
        """
        features = detector.detect(text)
        
        assert features.style in [WritingStyle.PROFESSIONAL, WritingStyle.FORMAL]
        assert features.formality_score > 55
    
    def test_conversational_style_detection(self, detector):
        """Test conversational style detection."""
        text = """
        Let me tell you about this cool feature.
        You're going to love it!
        """
        features = detector.detect(text)
        
        assert features.style in [WritingStyle.CONVERSATIONAL, WritingStyle.CASUAL]
    
    def test_formality_with_emojis_decreases(self, detector):
        """Test formality decreases with emojis."""
        formal = "This is a professional document."
        casual = "This is a professional document. 😊🎉"
        
        formal_features = detector.detect(formal)
        casual_features = detector.detect(casual)
        
        assert casual_features.formality_score < formal_features.formality_score
    
    def test_formality_with_exclamations_decreases(self, detector):
        """Test formality decreases with exclamations."""
        formal = "This is important."
        casual = "This is important!!!"
        
        formal_features = detector.detect(formal)
        casual_features = detector.detect(casual)
        
        assert casual_features.formality_score < formal_features.formality_score
    
    def test_style_confidence_calculation(self, detector):
        """Test style confidence is calculated."""
        text = "Furthermore, we acknowledge these contributions accordingly."
        features = detector.detect(text)
        
        assert 0 <= features.style_confidence <= 1.0
    
    def test_punctuation_density_per_100_words(self, detector):
        """Test punctuation density is normalized per 100 words."""
        text = "Word. " * 100
        features = detector.detect(text)
        
        # Should be approximately 1 period per word
        assert features.punctuation_density['period'] > 90
    
    def test_mixed_capitalization_pattern(self, detector):
        """Test mixed capitalization pattern."""
        text = "Some WORDS are CAPS and some are not"
        features = detector.detect(text)
        
        assert features.capitalization_pattern in ["mixed", "standard"]
    
    def test_no_punctuation_density(self, detector):
        """Test text without punctuation."""
        text = "This text has no special punctuation marks"
        features = detector.detect(text)
        
        assert features.punctuation_density['exclamation'] == 0
        assert features.punctuation_density['question'] == 0
    
    def test_high_emoji_density(self, detector):
        """Test high emoji density."""
        text = "🎉 🚀 💯 🔥 ✨ Great work team!"
        features = detector.detect(text)
        
        # 5 emojis, ~3 words = very high density
        assert features.emoji_density > 100
    
    def test_semicolon_usage(self, detector):
        """Test semicolon punctuation density."""
        text = "First clause; second clause; third clause; fourth clause."
        features = detector.detect(text)
        
        assert features.punctuation_density['semicolon'] > 0
    
    def test_colon_usage(self, detector):
        """Test colon punctuation density."""
        text = "List includes: apples, oranges, bananas, and grapes."
        features = detector.detect(text)
        
        assert features.punctuation_density['colon'] > 0
    
    def test_comma_usage(self, detector):
        """Test comma punctuation density."""
        text = "We have apples, oranges, bananas, grapes, and melons."
        features = detector.detect(text)
        
        assert features.punctuation_density['comma'] > 0