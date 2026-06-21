"""Constants for segmentation module."""

from typing import Dict, List

# Supported platforms
SUPPORTED_PLATFORMS: List[str] = ["linkedin", "twitter", "bluesky"]

# Clustering parameters
MIN_CLUSTERS: int = 2
MAX_CLUSTERS: int = 10
MIN_SAMPLES_FOR_CLUSTERING: int = 10

# Persona archetypes by platform
LINKEDIN_ARCHETYPES: List[str] = [
    "Professional",
    "Executive",
    "Entrepreneur",
    "Innovator",
    "Analyst",
    "Strategist",
    "Influencer",
    "Networker",
]

TWITTER_ARCHETYPES: List[str] = [
    "Commentator",
    "Curator",
    "Creator",
    "Amplifier",
    "Observer",
    "Debater",
    "Trendsetter",
    "Explorer",
]

BLUESKY_ARCHETYPES: List[str] = [
    "Pioneer",
    "Builder",
    "Advocate",
    "Connector",
    "Thinker",
    "Contributor",
    "Discoverer",
    "Collaborator",
]

ARCHETYPES_BY_PLATFORM: Dict[str, List[str]] = {
    "linkedin": LINKEDIN_ARCHETYPES,
    "twitter": TWITTER_ARCHETYPES,
    "bluesky": BLUESKY_ARCHETYPES,
}

# Adjectives by behavior
ADJECTIVES_BY_BEHAVIOR: Dict[str, List[str]] = {
    "high_engagement": ["Engaged", "Active", "Enthusiastic", "Passionate"],
    "low_engagement": ["Passive", "Occasional", "Selective", "Careful"],
    "content_creator": ["Creative", "Expressive", "Prolific", "Original"],
    "content_consumer": ["Curious", "Informed", "Attentive", "Discerning"],
    "professional": ["Driven", "Ambitious", "Focused", "Strategic"],
    "casual": ["Relaxed", "Open", "Exploratory", "Balanced"],
}

# Performance targets
SEGMENTATION_P95_MS: float = 300.0
CLUSTERING_P95_MS: float = 200.0
PERSONA_GENERATION_P95_MS: float = 500.0
RECOMMENDATION_P95_MS: float = 100.0

# Quality thresholds
MIN_SILHOUETTE_SCORE: float = 0.65
MIN_PERSONA_ACCURACY: float = 0.88
MIN_ENGAGEMENT_IMPROVEMENT: float = 0.20
MIN_STABILITY_SCORE: float = 0.80

# Feature extraction defaults
DEFAULT_FEATURE_SCALER: str = "standard"
DEFAULT_IMPUTATION_METHOD: str = "mean"
OUTLIER_THRESHOLD: float = 3.0  # Standard deviations

# Engagement levels
ENGAGEMENT_LEVELS: Dict[str, tuple[float, float]] = {
    "high": (0.75, 1.0),
    "medium": (0.4, 0.75),
    "low": (0.0, 0.4),
}