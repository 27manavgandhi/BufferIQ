"""Title and headline suggestion generation."""

from typing import List
import logging
import random

logger = logging.getLogger(__name__)


class TitleSuggester:
    """
    Generate compelling titles and headlines.

    Creates multiple title variations optimized for engagement.
    """

    def __init__(self):
        """Initialize title suggester."""
        self.title_templates = [
            # How-to templates
            "How to {action} {topic}",
            "The Complete Guide to {topic}",
            "A Beginner's Guide to {topic}",
            "{number} Ways to {action} {topic}",
            
            # List templates
            "{number} {topic} Best Practices You Need to Know",
            "Top {number} {topic} Strategies for {year}",
            "{number} Essential {topic} Tips",
            
            # Problem-solution
            "Why {topic} Matters in {year}",
            "The Ultimate {topic} Checklist",
            "What You Need to Know About {topic}",
            
            # Curiosity
            "Everything You Need to Know About {topic}",
            "The Truth About {topic}",
            "{topic}: A Deep Dive",
            
            # Authority
            "Master {topic} in {number} Steps",
            "The Definitive {topic} Guide",
            "{topic} Explained: From Basics to Advanced",
        ]

    def generate_titles(
        self, topic: str, keywords: List[str], count: int = 5
    ) -> List[str]:
        """
        Generate title suggestions.

        Args:
            topic: Main topic
            keywords: Related keywords
            count: Number of titles to generate

        Returns:
            List of title suggestions
        """
        titles = []
        
        # Extract action verbs from keywords
        actions = self._extract_actions(keywords)
        
        # Current year
        from datetime import datetime
        year = datetime.now().year
        
        # Generate titles from templates
        for template in random.sample(self.title_templates, min(count, len(self.title_templates))):
            title = template.format(
                topic=topic,
                action=random.choice(actions) if actions else "master",
                number=random.choice([5, 7, 10]),
                year=year,
            )
            titles.append(title)
        
        return titles[:count]

    def _extract_actions(self, keywords: List[str]) -> List[str]:
        """Extract action verbs from keywords."""
        common_actions = [
            "master", "learn", "understand", "implement",
            "optimize", "improve", "build", "create",
            "leverage", "use", "apply", "develop"
        ]
        
        # Check if keywords contain actions
        actions = []
        for keyword in keywords:
            if keyword.lower() in common_actions:
                actions.append(keyword.lower())
        
        # If no actions found, use defaults
        if not actions:
            actions = ["master", "learn", "implement"]
        
        return actions

    def optimize_title_for_platform(
        self, title: str, platform: str
    ) -> str:
        """
        Optimize title for specific platform.

        Args:
            title: Original title
            platform: Target platform

        Returns:
            Optimized title
        """
        if platform == "twitter":
            # Keep it short for Twitter
            if len(title) > 100:
                # Truncate intelligently
                words = title.split()
                truncated = []
                length = 0
                for word in words:
                    if length + len(word) + 1 > 95:
                        break
                    truncated.append(word)
                    length += len(word) + 1
                title = " ".join(truncated) + "..."
        
        elif platform == "linkedin":
            # LinkedIn can be longer and more professional
            if not any(word in title.lower() for word in ["guide", "how", "complete"]):
                # Make it more professional
                title = f"Professional Guide: {title}"
        
        return title

    def generate_hook(self, topic: str) -> str:
        """
        Generate opening hook for content.

        Args:
            topic: Content topic

        Returns:
            Hook sentence
        """
        hooks = [
            f"If you're looking to master {topic}, you're in the right place.",
            f"Understanding {topic} has never been more important.",
            f"Let's dive deep into {topic} and unlock its full potential.",
            f"Here's everything you need to know about {topic}.",
            f"Ready to transform your approach to {topic}? Let's begin.",
        ]
        
        return random.choice(hooks)