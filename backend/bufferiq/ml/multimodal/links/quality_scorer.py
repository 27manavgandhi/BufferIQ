"""Link preview quality scoring."""

from typing import Optional


class QualityScorer:
    """Score link preview quality."""
    
    def __init__(self):
        """Initialize quality scorer."""
        pass
    
    def score_title(self, title: Optional[str]) -> float:
        """
        Score title quality.
        
        Args:
            title: Title text
            
        Returns:
            Quality score (0-100)
        """
        if not title:
            return 0.0
        
        score = 50.0  # Base score for having a title
        
        # Length score (optimal 40-70 characters)
        length = len(title)
        if 40 <= length <= 70:
            score += 30.0
        elif 30 <= length < 40 or 70 < length <= 80:
            score += 15.0
        elif length < 30:
            score += length / 30 * 15.0
        
        # Contains numbers (often engaging)
        if any(char.isdigit() for char in title):
            score += 10.0
        
        # Title case or sentence case (professional)
        if title[0].isupper():
            score += 10.0
        
        return min(score, 100.0)
    
    def score_description(self, description: Optional[str]) -> float:
        """
        Score description quality.
        
        Args:
            description: Description text
            
        Returns:
            Quality score (0-100)
        """
        if not description:
            return 0.0
        
        score = 40.0  # Base score for having a description
        
        # Length score (optimal 100-200 characters)
        length = len(description)
        if 100 <= length <= 200:
            score += 40.0
        elif 50 <= length < 100 or 200 < length <= 250:
            score += 20.0
        elif length < 50:
            score += length / 50 * 20.0
        
        # Complete sentences (ends with punctuation)
        if description.rstrip()[-1] in '.!?':
            score += 10.0
        
        # Not too promotional (limited use of exclamation marks)
        exclamation_count = description.count('!')
        if exclamation_count <= 1:
            score += 10.0
        
        return min(score, 100.0)
    
    def score_image(self, image_url: Optional[str]) -> float:
        """
        Score image quality.
        
        Args:
            image_url: Image URL
            
        Returns:
            Quality score (0-100)
        """
        if not image_url:
            return 0.0
        
        score = 60.0  # Base score for having an image
        
        # Check if URL suggests high quality
        url_lower = image_url.lower()
        
        # High resolution indicators
        if any(res in url_lower for res in ['1200', '1920', '2048', 'large', 'hd']):
            score += 20.0
        
        # Image format (prefer modern formats)
        if url_lower.endswith(('.jpg', '.jpeg', '.png', '.webp')):
            score += 10.0
        
        # Avoid thumbnail indicators
        if any(thumb in url_lower for thumb in ['thumb', 'small', '150', '200']):
            score -= 20.0
        
        # Secure URL
        if image_url.startswith('https://'):
            score += 10.0
        
        return max(0.0, min(score, 100.0))
    
    def score_overall(
        self,
        title_score: float,
        description_score: float,
        image_score: float
    ) -> float:
        """
        Calculate overall quality score.
        
        Args:
            title_score: Title quality score
            description_score: Description quality score
            image_score: Image quality score
            
        Returns:
            Overall quality score (0-100)
        """
        # Weighted average
        overall = (
            title_score * 0.40 +
            description_score * 0.35 +
            image_score * 0.25
        )
        
        return overall