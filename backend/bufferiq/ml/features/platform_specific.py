"""Platform-specific feature extraction."""

import re
from typing import Any

import pandas as pd

from bufferiq.core.logging import get_logger
from bufferiq.ml.features.base import (
    SUPPORTED_PLATFORMS,
    BaseFeatureExtractor,
    validate_platform,
)

logger = get_logger(__name__)

# Platform character limits
PLATFORM_CHAR_LIMITS = {
    "linkedin": {"max": 3000, "optimal_min": 1300, "optimal_max": 1500},
    "twitter": {"max": 280, "optimal_min": 71, "optimal_max": 100},
    "bluesky": {"max": 300, "optimal_min": 71, "optimal_max": 100},
}

# LinkedIn professional keywords
LINKEDIN_CAREER_KEYWORDS = {
    "job",
    "hire",
    "hiring",
    "opportunity",
    "career",
    "position",
    "role",
    "team",
    "recruiting",
    "apply",
}

LINKEDIN_INDUSTRY_HASHTAGS = {
    "#leadership",
    "#technology",
    "#innovation",
    "#business",
    "#professional",
    "#career",
    "#hiring",
}

# Twitter keywords
TWITTER_LINGO = {"til", "imo", "imho", "icymi", "fyi", "tbh", "btw", "afaik"}

TWITTER_RT_KEYWORDS = {"rt", "via", "ht", "h/t"}

# Bluesky keywords
BLUESKY_TECH_KEYWORDS = {
    "protocol",
    "federation",
    "decentralization",
    "atproto",
    "bluesky",
}


class PlatformSpecificFeatureExtractor(BaseFeatureExtractor):
    """Extract platform-specific features."""

    @property
    def feature_names(self) -> list[str]:
        """Return list of platform-specific feature names."""
        return [
            # LinkedIn
            "is_professional_tone",
            "has_career_keywords",
            "has_industry_hashtags",
            "optimal_length_linkedin",
            "has_call_to_action",
            "document_structure_score",
            # Twitter
            "is_thread_starter",
            "has_retweet_keywords",
            "is_reply",
            "uses_twitter_lingo",
            "optimal_length_twitter",
            "hashtag_position",
            # Bluesky
            "is_decentralization_topic",
            "has_tech_keywords",
            "community_engagement_style",
            "optimal_length_bluesky",
        ]

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract platform-specific features from DataFrame.

        Args:
            df: DataFrame with 'content' and 'platform' columns

        Returns:
            DataFrame with platform-specific features
        """
        self.validate_input(df, ["content", "platform"])

        # Validate all platforms
        for platform in df["platform"].dropna().unique():
            validate_platform(platform)

        result = pd.DataFrame(index=df.index)

        # Extract for each platform
        for platform in SUPPORTED_PLATFORMS:
            platform_mask = df["platform"] == platform

            if platform == "linkedin":
                platform_features = df[platform_mask].apply(
                    lambda row: self._extract_linkedin_features(row["content"]),
                    axis=1,
                    result_type="expand",
                )
            elif platform == "twitter":
                platform_features = df[platform_mask].apply(
                    lambda row: self._extract_twitter_features(row["content"]),
                    axis=1,
                    result_type="expand",
                )
            elif platform == "bluesky":
                platform_features = df[platform_mask].apply(
                    lambda row: self._extract_bluesky_features(row["content"]),
                    axis=1,
                    result_type="expand",
                )
            else:
                continue

            # Assign features for this platform
            for col in platform_features.columns:
                if col not in result.columns:
                    result[col] = 0
                result.loc[platform_mask, col] = platform_features[col]

        # Fill missing features with 0
        for feature_name in self.feature_names:
            if feature_name not in result.columns:
                result[feature_name] = 0

        logger.info(f"Extracted {len(result.columns)} platform-specific features")

        return result

    def extract_single(self, post_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract platform-specific features from single post.

        Args:
            post_data: Dictionary with 'content' and 'platform' keys

        Returns:
            Dictionary with platform-specific features
        """
        content = post_data.get("content", "")
        platform = post_data.get("platform", "")

        # Validate platform
        if platform:
            validate_platform(platform)

        # Initialize all features to 0
        features = {name: 0 for name in self.feature_names}

        if not content or not platform:
            return features

        # Extract based on platform
        if platform == "linkedin":
            platform_features = self._extract_linkedin_features(content)
        elif platform == "twitter":
            platform_features = self._extract_twitter_features(content)
        elif platform == "bluesky":
            platform_features = self._extract_bluesky_features(content)
        else:
            return features

        # Update features
        features.update(platform_features)

        return features

    def _extract_linkedin_features(self, content: str) -> dict[str, Any]:
        """Extract LinkedIn-specific features."""
        content_lower = content.lower()
        text_length = len(content)

        # Professional tone (simple heuristic: formal words, no excessive punctuation)
        formal_indicators = sum(
            1
            for word in [
                "pleased",
                "excited",
                "honored",
                "delighted",
                "thrilled",
                "grateful",
            ]
            if word in content_lower
        )
        excessive_exclamation = content.count("!") > 3
        is_professional_tone = int(formal_indicators > 0 and not excessive_exclamation)

        # Career keywords
        has_career_keywords = int(
            any(keyword in content_lower for keyword in LINKEDIN_CAREER_KEYWORDS)
        )

        # Industry hashtags
        has_industry_hashtags = int(
            any(hashtag in content_lower for hashtag in LINKEDIN_INDUSTRY_HASHTAGS)
        )

        # Optimal length for LinkedIn
        optimal_length_linkedin = int(
            PLATFORM_CHAR_LIMITS["linkedin"]["optimal_min"]
            <= text_length
            <= PLATFORM_CHAR_LIMITS["linkedin"]["optimal_max"]
        )

        # Call to action
        cta_phrases = [
            "learn more",
            "read more",
            "join us",
            "apply now",
            "get in touch",
            "contact us",
            "register",
            "sign up",
        ]
        has_call_to_action = int(any(phrase in content_lower for phrase in cta_phrases))

        # Document structure score (paragraphs, bullet points)
        paragraph_count = content.count("\n\n") + 1
        has_bullet_points = bool(
            re.search(r"[•\-\*]\s", content) or re.search(r"^\d+\.", content, re.M)
        )
        document_structure_score = min(
            1.0, (paragraph_count / 3 + (0.5 if has_bullet_points else 0))
        )

        return {
            "is_professional_tone": is_professional_tone,
            "has_career_keywords": has_career_keywords,
            "has_industry_hashtags": has_industry_hashtags,
            "optimal_length_linkedin": optimal_length_linkedin,
            "has_call_to_action": has_call_to_action,
            "document_structure_score": document_structure_score,
            # Non-LinkedIn features set to 0
            "is_thread_starter": 0,
            "has_retweet_keywords": 0,
            "is_reply": 0,
            "uses_twitter_lingo": 0,
            "optimal_length_twitter": 0,
            "hashtag_position": 0,
            "is_decentralization_topic": 0,
            "has_tech_keywords": 0,
            "community_engagement_style": 0,
            "optimal_length_bluesky": 0,
        }

    def _extract_twitter_features(self, content: str) -> dict[str, Any]:
        """Extract Twitter-specific features."""
        content_lower = content.lower()
        text_length = len(content)

        # Thread starter
        is_thread_starter = int("🧵" in content or content_lower.startswith("thread:"))

        # Retweet keywords
        has_retweet_keywords = int(
            any(keyword in content_lower for keyword in TWITTER_RT_KEYWORDS)
        )

        # Reply (starts with @)
        is_reply = int(content.strip().startswith("@"))

        # Twitter lingo
        uses_twitter_lingo = int(
            any(word in content_lower.split() for word in TWITTER_LINGO)
        )

        # Optimal length
        optimal_length_twitter = int(
            PLATFORM_CHAR_LIMITS["twitter"]["optimal_min"]
            <= text_length
            <= PLATFORM_CHAR_LIMITS["twitter"]["optimal_max"]
        )

        # Hashtag position
        hashtags = re.findall(r"#\w+", content)
        if not hashtags:
            hashtag_position = 0  # none
        else:
            first_hashtag_pos = content.find(hashtags[0])
            if first_hashtag_pos < text_length * 0.33:
                hashtag_position = 1  # beginning
            elif first_hashtag_pos > text_length * 0.67:
                hashtag_position = 3  # end
            else:
                hashtag_position = 2  # middle

        return {
            "is_thread_starter": is_thread_starter,
            "has_retweet_keywords": has_retweet_keywords,
            "is_reply": is_reply,
            "uses_twitter_lingo": uses_twitter_lingo,
            "optimal_length_twitter": optimal_length_twitter,
            "hashtag_position": hashtag_position,
            # Non-Twitter features set to 0
            "is_professional_tone": 0,
            "has_career_keywords": 0,
            "has_industry_hashtags": 0,
            "optimal_length_linkedin": 0,
            "has_call_to_action": 0,
            "document_structure_score": 0,
            "is_decentralization_topic": 0,
            "has_tech_keywords": 0,
            "community_engagement_style": 0,
            "optimal_length_bluesky": 0,
        }

    def _extract_bluesky_features(self, content: str) -> dict[str, Any]:
        """Extract Bluesky-specific features."""
        content_lower = content.lower()
        text_length = len(content)

        # Decentralization topic
        is_decentralization_topic = int(
            any(keyword in content_lower for keyword in BLUESKY_TECH_KEYWORDS)
        )

        # Tech keywords (broader than just decentralization)
        tech_keywords = {
            "tech",
            "software",
            "developer",
            "coding",
            "programming",
            "api",
            "open source",
        }
        has_tech_keywords = int(
            any(keyword in content_lower for keyword in tech_keywords)
        )

        # Community engagement style (conversational vs broadcast)
        # Conversational: has questions, personal pronouns
        has_question = "?" in content
        has_personal_pronouns = any(
            word in content_lower.split() for word in ["i", "we", "my", "our"]
        )
        community_engagement_style = int(has_question or has_personal_pronouns)

        # Optimal length (similar to Twitter)
        optimal_length_bluesky = int(
            PLATFORM_CHAR_LIMITS["bluesky"]["optimal_min"]
            <= text_length
            <= PLATFORM_CHAR_LIMITS["bluesky"]["optimal_max"]
        )

        return {
            "is_decentralization_topic": is_decentralization_topic,
            "has_tech_keywords": has_tech_keywords,
            "community_engagement_style": community_engagement_style,
            "optimal_length_bluesky": optimal_length_bluesky,
            # Non-Bluesky features set to 0
            "is_professional_tone": 0,
            "has_career_keywords": 0,
            "has_industry_hashtags": 0,
            "optimal_length_linkedin": 0,
            "has_call_to_action": 0,
            "document_structure_score": 0,
            "is_thread_starter": 0,
            "has_retweet_keywords": 0,
            "is_reply": 0,
            "uses_twitter_lingo": 0,
            "optimal_length_twitter": 0,
            "hashtag_position": 0,
        }
