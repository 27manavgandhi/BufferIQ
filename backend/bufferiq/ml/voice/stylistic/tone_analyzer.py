"""
Tone consistency measurement and analysis.

Analyzes emotional tone and consistency across content.
"""

from typing import Dict, List
from collections import Counter

from textblob import TextBlob
from nltk import word_tokenize


class ToneAnalyzer:
    """
    Analyze emotional tone of text.
    
    Measures tone characteristics for voice consistency.
    
    Example:
```python
        analyzer = ToneAnalyzer()
        tone_profile = analyzer.analyze("Excited to share this news!")
        print(f"Tone: {tone_profile['primary_tone']}")
```
    """
    
    def __init__(self):
        """Initialize tone analyzer."""
        self.positive_words = {
            'excellent', 'amazing', 'fantastic', 'wonderful', 'great',
            'awesome', 'brilliant', 'superb', 'outstanding', 'excited',
            'happy', 'delighted', 'thrilled', 'pleased', 'glad'
        }
        self.negative_words = {
            'terrible', 'awful', 'horrible', 'bad', 'poor',
            'disappointing', 'sad', 'angry', 'frustrated', 'upset',
            'concerned', 'worried', 'unfortunate', 'difficult'
        }
        self.urgent_words = {
            'urgent', 'immediately', 'asap', 'critical', 'important',
            'emergency', 'now', 'quickly', 'hurry', 'deadline'
        }
    
    def analyze(self, text: str) -> Dict[str, any]:
        """
        Analyze tone characteristics.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tone profile dictionary
        
        Raises:
            ValueError: If text is empty
        """
        if not text or len(text.strip()) < 3:
            raise ValueError("Text too short for tone analysis")
        
        # Use TextBlob for sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Analyze word choices
        words = set(word_tokenize(text.lower()))
        
        positive_count = len(words & self.positive_words)
        negative_count = len(words & self.negative_words)
        urgent_count = len(words & self.urgent_words)
        
        # Determine primary tone
        if urgent_count > 0:
            primary_tone = "urgent"
        elif polarity > 0.3:
            primary_tone = "positive"
        elif polarity < -0.3:
            primary_tone = "negative"
        else:
            primary_tone = "neutral"
        
        # Determine secondary characteristics
        if subjectivity > 0.6:
            emotion = "emotional"
        elif subjectivity > 0.3:
            emotion = "balanced"
        else:
            emotion = "factual"
        
        return {
            'primary_tone': primary_tone,
            'polarity': polarity,
            'subjectivity': subjectivity,
            'emotion_level': emotion,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'urgency_indicators': urgent_count,
        }
    
    def compare_tones(
        self, tone1: Dict[str, any], tone2: Dict[str, any]
    ) -> float:
        """
        Compare two tone profiles for consistency.
        
        Args:
            tone1: First tone profile
            tone2: Second tone profile
        
        Returns:
            Consistency score (0-1)
        """
        # Compare primary tone
        tone_match = 1.0 if tone1['primary_tone'] == tone2['primary_tone'] else 0.0
        
        # Compare polarity
        polarity_diff = abs(tone1['polarity'] - tone2['polarity'])
        polarity_sim = 1.0 - min(polarity_diff / 2.0, 1.0)
        
        # Compare subjectivity
        subj_diff = abs(tone1['subjectivity'] - tone2['subjectivity'])
        subj_sim = 1.0 - min(subj_diff, 1.0)
        
        # Weighted average
        consistency = (
            tone_match * 0.4 +
            polarity_sim * 0.3 +
            subj_sim * 0.3
        )
        
        return consistency