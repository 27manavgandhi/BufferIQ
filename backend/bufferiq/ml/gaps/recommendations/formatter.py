"""Content format recommendation."""

from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class FormatRecommender:
    """
    Recommend optimal content format.

    Suggests format, length, and structure based on topic and goals.
    """

    def recommend(self, topic: str, keywords: List[str]) -> Dict[str, Any]:
        """
        Recommend content format.

        Args:
            topic: Content topic
            keywords: Related keywords

        Returns:
            Format recommendation with details
        """
        # Analyze keywords for format hints
        keyword_text = " ".join(keywords).lower()
        
        # Determine format
        if any(word in keyword_text for word in ["how", "guide", "tutorial"]):
            content_format = "how-to"
            length = "long"
            structure = "step-by-step"
        elif any(word in keyword_text for word in ["best", "top", "list"]):
            content_format = "listicle"
            length = "medium"
            structure = "numbered-list"
        elif any(word in keyword_text for word in ["what", "why", "explained"]):
            content_format = "explainer"
            length = "medium"
            structure = "q-and-a"
        elif any(word in keyword_text for word in ["case", "study", "example"]):
            content_format = "case-study"
            length = "long"
            structure = "narrative"
        elif any(word in keyword_text for word in ["opinion", "think", "believe"]):
            content_format = "opinion"
            length = "short"
            structure = "argument"
        else:
            content_format = "article"
            length = "medium"
            structure = "standard"
        
        # Map length to word count
        word_counts = {
            "short": "500-800 words",
            "medium": "800-1500 words",
            "long": "1500-2500 words",
        }
        
        return {
            "format": content_format,
            "length": length,
            "word_count": word_counts[length],
            "structure": structure,
            "sections": self._recommend_sections(content_format),
        }

    def _recommend_sections(self, content_format: str) -> List[str]:
        """Recommend content sections."""
        section_templates = {
            "how-to": [
                "Introduction",
                "Prerequisites",
                "Step-by-step instructions",
                "Common pitfalls",
                "Conclusion",
            ],
            "listicle": [
                "Introduction",
                "Item 1-N with descriptions",
                "Summary",
            ],
            "explainer": [
                "What is it?",
                "Why does it matter?",
                "How does it work?",
                "Key takeaways",
            ],
            "case-study": [
                "Background",
                "Challenge",
                "Solution",
                "Results",
                "Lessons learned",
            ],
            "opinion": [
                "Opening statement",
                "Supporting arguments",
                "Counterarguments",
                "Conclusion",
            ],
            "article": [
                "Introduction",
                "Main points",
                "Analysis",
                "Conclusion",
            ],
        }
        
        return section_templates.get(content_format, section_templates["article"])

    def recommend_media(self, content_format: str) -> List[str]:
        """
        Recommend media elements.

        Args:
            content_format: Content format

        Returns:
            List of recommended media types
        """
        media_recommendations = {
            "how-to": ["screenshots", "diagrams", "video"],
            "listicle": ["images", "icons", "charts"],
            "explainer": ["diagrams", "infographics", "animations"],
            "case-study": ["charts", "before-after images", "quotes"],
            "opinion": ["header image", "pull quotes"],
            "article": ["header image", "supporting images"],
        }
        
        return media_recommendations.get(content_format, ["images"])