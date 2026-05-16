"""Tests for tone analyzer."""

import pytest
from bufferiq.ml.voice.stylistic.tone_analyzer import ToneAnalyzer


class TestToneAnalyzer:
    """Test tone analyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return ToneAnalyzer()
    
    def test_analyze_basic_text(self, analyzer):
        """Test basic tone analysis."""
        text = "Excited to share this news!"
        tone = analyzer.analyze(text)
        
        assert isinstance(tone, dict)
        assert 'primary_tone' in tone
        assert 'polarity' in tone
        assert 'subjectivity' in tone
    
    def test_analyze_empty_text_raises_error(self, analyzer):
        """Test empty text raises error."""
        with pytest.raises(ValueError, match="too short"):
            analyzer.analyze("")
    
    def test_analyze_positive_tone(self, analyzer):
        """Test positive tone detection."""
        text = "This is amazing! I'm so happy and excited!"
        tone = analyzer.analyze(text)
        
        assert tone['primary_tone'] == 'positive'
        assert tone['polarity'] > 0.3
    
    def test_analyze_negative_tone(self, analyzer):
        """Test negative tone detection."""
        text = "This is terrible. I'm very disappointed and frustrated."
        tone = analyzer.analyze(text)
        
        assert tone['primary_tone'] == 'negative'
        assert tone['polarity'] < -0.3
    
    def test_analyze_neutral_tone(self, analyzer):
        """Test neutral tone detection."""
        text = "The meeting is scheduled for tomorrow at 3pm."
        tone = analyzer.analyze(text)
        
        assert tone['primary_tone'] == 'neutral'
        assert -0.3 <= tone['polarity'] <= 0.3
    
    def test_analyze_urgent_tone(self, analyzer):
        """Test urgent tone detection."""
        text = "URGENT: Immediate action required! Critical deadline approaching!"
        tone = analyzer.analyze(text)
        
        assert tone['primary_tone'] == 'urgent'
        assert tone['urgency_indicators'] > 0
    
    def test_polarity_range(self, analyzer):
        """Test polarity is in valid range."""
        text = "This is a test message."
        tone = analyzer.analyze(text)
        
        assert -1.0 <= tone['polarity'] <= 1.0
    
    def test_subjectivity_range(self, analyzer):
        """Test subjectivity is in valid range."""
        text = "This is a test message."
        tone = analyzer.analyze(text)
        
        assert 0.0 <= tone['subjectivity'] <= 1.0
    
    def test_emotional_text_high_subjectivity(self, analyzer):
        """Test emotional text has high subjectivity."""
        text = "I absolutely love this! It's the best thing ever!"
        tone = analyzer.analyze(text)
        
        assert tone['subjectivity'] > 0.6
        assert tone['emotion_level'] == 'emotional'
    
    def test_factual_text_low_subjectivity(self, analyzer):
        """Test factual text has low subjectivity."""
        text = "The report shows a 15% increase in revenue for Q3."
        tone = analyzer.analyze(text)
        
        assert tone['subjectivity'] < 0.4
        assert tone['emotion_level'] == 'factual'
    
    def test_compare_tones_identical(self, analyzer):
        """Test comparing identical tones."""
        text = "This is a positive message."
        tone1 = analyzer.analyze(text)
        tone2 = analyzer.analyze(text)
        
        consistency = analyzer.compare_tones(tone1, tone2)
        assert consistency == pytest.approx(1.0, rel=0.01)
    
    def test_compare_tones_different_primary(self, analyzer):
        """Test comparing different primary tones."""
        tone1 = {
            'primary_tone': 'positive',
            'polarity': 0.8,
            'subjectivity': 0.7,
        }
        tone2 = {
            'primary_tone': 'negative',
            'polarity': -0.8,
            'subjectivity': 0.7,
        }
        
        consistency = analyzer.compare_tones(tone1, tone2)
        assert consistency < 0.5
    
    def test_compare_tones_similar_polarity(self, analyzer):
        """Test comparing similar polarity."""
        tone1 = {
            'primary_tone': 'positive',
            'polarity': 0.7,
            'subjectivity': 0.5,
        }
        tone2 = {
            'primary_tone': 'positive',
            'polarity': 0.8,
            'subjectivity': 0.5,
        }
        
        consistency = analyzer.compare_tones(tone1, tone2)
        assert consistency > 0.8
    
    def test_positive_indicators_counting(self, analyzer):
        """Test positive indicator counting."""
        text = "Excellent amazing fantastic wonderful great!"
        tone = analyzer.analyze(text)
        
        assert tone['positive_indicators'] > 3
    
    def test_negative_indicators_counting(self, analyzer):
        """Test negative indicator counting."""
        text = "Terrible awful horrible bad disappointing!"
        tone = analyzer.analyze(text)
        
        assert tone['negative_indicators'] > 3
    
    def test_urgency_indicators_counting(self, analyzer):
        """Test urgency indicator counting."""
        text = "Urgent! Immediately! ASAP! Critical deadline!"
        tone = analyzer.analyze(text)
        
        assert tone['urgency_indicators'] > 2
    
    def test_balanced_emotion_level(self, analyzer):
        """Test balanced emotion level."""
        text = "The project shows promise, though some challenges remain."
        tone = analyzer.analyze(text)
        
        assert tone['emotion_level'] in ['balanced', 'factual']
    
    def test_polarity_very_positive(self, analyzer):
        """Test very positive polarity."""
        text = "Absolutely incredible! The best experience ever! Highly recommend!"
        tone = analyzer.analyze(text)
        
        assert tone['polarity'] > 0.7
    
    def test_polarity_very_negative(self, analyzer):
        """Test very negative polarity."""
        text = "Absolutely terrible! The worst experience ever! Do not recommend!"
        tone = analyzer.analyze(text)
        
        assert tone['polarity'] < -0.7
    
    def test_mixed_tone(self, analyzer):
        """Test mixed tone detection."""
        text = "The product is great but the service was disappointing."
        tone = analyzer.analyze(text)
        
        # Should be close to neutral due to mixed sentiment
        assert -0.3 <= tone['polarity'] <= 0.3
    
    def test_compare_tones_consistency_range(self, analyzer):
        """Test tone consistency is in valid range."""
        tone1 = {'primary_tone': 'positive', 'polarity': 0.5, 'subjectivity': 0.5}
        tone2 = {'primary_tone': 'negative', 'polarity': -0.5, 'subjectivity': 0.5}
        
        consistency = analyzer.compare_tones(tone1, tone2)
        assert 0.0 <= consistency <= 1.0
    
    def test_urgency_overrides_sentiment(self, analyzer):
        """Test urgency tone overrides sentiment."""
        text = "URGENT: This is critical and needs immediate attention!"
        tone = analyzer.analyze(text)
        
        # Should be urgent even if sentiment is neutral/positive
        assert tone['primary_tone'] == 'urgent'
    
    def test_no_indicators(self, analyzer):
        """Test text without special indicators."""
        text = "The cat sat on the mat."
        tone = analyzer.analyze(text)
        
        assert tone['positive_indicators'] == 0
        assert tone['negative_indicators'] == 0
        assert tone['urgency_indicators'] == 0