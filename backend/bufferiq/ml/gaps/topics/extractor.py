"""
Topic extraction from content corpus.

Extracts and clusters topics using TF-IDF vectorization and DBSCAN clustering.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session

from bufferiq.ml.gaps.topics.clusterer import TopicClusterer
from bufferiq.ml.gaps.topics.lifecycle_analyzer import LifecycleAnalyzer

logger = logging.getLogger(__name__)

SUPPORTED_PLATFORMS = ["linkedin", "twitter", "bluesky"]


@dataclass
class Topic:
    """Extracted topic with metadata."""

    topic_id: str
    name: str
    keywords: List[str]
    description: str
    cluster_id: int

    # Lifecycle
    lifecycle_stage: str  # "emerging", "growing", "mature", "declining"
    first_seen: datetime
    last_seen: datetime

    # Metrics
    post_count: int
    total_engagement: int
    avg_engagement: float
    growth_rate: float  # % change over last 30 days

    # Relevance
    relevance_score: float  # 0-1
    search_volume: Optional[int] = None
    trend_momentum: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic_id": self.topic_id,
            "name": self.name,
            "keywords": self.keywords,
            "description": self.description,
            "cluster_id": self.cluster_id,
            "lifecycle_stage": self.lifecycle_stage,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "post_count": self.post_count,
            "total_engagement": self.total_engagement,
            "avg_engagement": self.avg_engagement,
            "growth_rate": self.growth_rate,
            "relevance_score": self.relevance_score,
            "search_volume": self.search_volume,
            "trend_momentum": self.trend_momentum,
        }


@dataclass
class TopicCluster:
    """Group of related topics."""

    cluster_id: int
    name: str
    topics: List[Topic] = field(default_factory=list)
    centroid_keywords: List[str] = field(default_factory=list)
    total_posts: int = 0
    avg_engagement: float = 0.0
    coverage_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "topics": [t.to_dict() for t in self.topics],
            "centroid_keywords": self.centroid_keywords,
            "total_posts": self.total_posts,
            "avg_engagement": self.avg_engagement,
            "coverage_percentage": self.coverage_percentage,
        }


class TopicExtractor:
    """
    Extract and cluster topics from content corpus.

    Uses TF-IDF and clustering to identify distinct topics
    and their relationships in user's content.

    Example:
```python
        extractor = TopicExtractor(db_session)
        topics = await extractor.extract(
            user_id="user123",
            platform="linkedin",
            lookback_days=90
        )

        for topic in topics:
            print(f"{topic.name}: {topic.post_count} posts")
            print(f"  Stage: {topic.lifecycle_stage}")
            print(f"  Growth: {topic.growth_rate:.1f}%")
```
    """

    def __init__(
        self,
        db_session: Session,
        min_topic_posts: int = 3,
        max_topics: int = 50,
        similarity_threshold: float = 0.3,
    ):
        """
        Initialize topic extractor.

        Args:
            db_session: Database session
            min_topic_posts: Minimum posts to form a topic
            max_topics: Maximum topics to extract
            similarity_threshold: Similarity threshold for clustering
        """
        self.db = db_session
        self.min_posts = min_topic_posts
        self.max_topics = max_topics
        self.threshold = similarity_threshold

        self.vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 3), stop_words="english"
        )
        self.clusterer = TopicClusterer(similarity_threshold=similarity_threshold)
        self.lifecycle_analyzer = LifecycleAnalyzer()

    async def extract(
        self, user_id: str, platform: str, lookback_days: int = 90
    ) -> List[Topic]:
        """
        Extract topics from user's content.

        Args:
            user_id: User identifier
            platform: Platform to analyze
            lookback_days: Days of history

        Returns:
            List of extracted topics

        Raises:
            ValueError: If platform not supported or insufficient posts
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise ValueError(
                f"Platform '{platform}' not supported. "
                f"Supported: {SUPPORTED_PLATFORMS}"
            )

        # Fetch posts from database
        posts = await self._fetch_posts(user_id, platform, lookback_days)

        if len(posts) < self.min_posts:
            raise ValueError(
                f"Insufficient posts: {len(posts)}. Minimum required: {self.min_posts}"
            )

        logger.info(
            f"Extracting topics from {len(posts)} posts for user {user_id} on {platform}"
        )

        # Extract text content
        texts = [post["content"] for post in posts]

        # Vectorize
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        except ValueError as e:
            raise ValueError(f"Failed to vectorize content: {str(e)}")

        # Cluster topics
        clusters = self.clusterer.cluster(tfidf_matrix, posts)

        # Build topics from clusters
        topics = []
        for cluster_data in clusters:
            topic = self._build_topic(cluster_data, posts)
            if topic.post_count >= self.min_posts:
                topics.append(topic)

        # Sort by relevance
        topics.sort(key=lambda t: t.relevance_score, reverse=True)

        # Limit to max topics
        topics = topics[: self.max_topics]

        logger.info(f"Extracted {len(topics)} topics")

        return topics

    async def _fetch_posts(
        self, user_id: str, platform: str, lookback_days: int
    ) -> List[Dict[str, Any]]:
        """Fetch posts from database (mock implementation)."""
        # In production, this would query the actual database
        # For now, return mock data
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        mock_posts = []
        base_date = datetime.now() - timedelta(days=lookback_days)

        # Generate mock posts with realistic topics
        topics_data = [
            {
                "keywords": ["AI", "machine learning", "artificial intelligence"],
                "count": 15,
            },
            {"keywords": ["python", "programming", "development"], "count": 12},
            {"keywords": ["data science", "analytics", "insights"], "count": 10},
            {"keywords": ["leadership", "management", "team"], "count": 8},
            {"keywords": ["innovation", "technology", "future"], "count": 7},
        ]

        post_id = 1
        for topic_data in topics_data:
            for i in range(topic_data["count"]):
                keywords = " ".join(topic_data["keywords"])
                mock_posts.append(
                    {
                        "id": f"post_{post_id}",
                        "content": f"Post about {keywords} with insights and analysis",
                        "created_at": base_date + timedelta(days=i * 2),
                        "engagement": np.random.randint(50, 500),
                        "platform": platform,
                    }
                )
                post_id += 1

        return mock_posts

    def _build_topic(
        self, cluster_data: Dict[str, Any], all_posts: List[Dict[str, Any]]
    ) -> Topic:
        """Build topic from cluster data."""
        cluster_id = cluster_data["cluster_id"]
        post_indices = cluster_data["post_indices"]
        keywords = cluster_data["keywords"]

        # Get posts in this cluster
        cluster_posts = [all_posts[i] for i in post_indices]

        # Calculate metrics
        post_count = len(cluster_posts)
        total_engagement = sum(p["engagement"] for p in cluster_posts)
        avg_engagement = total_engagement / post_count if post_count > 0 else 0

        # Temporal data
        dates = [p["created_at"] for p in cluster_posts]
        first_seen = min(dates)
        last_seen = max(dates)

        # Calculate growth rate
        growth_rate = self._calculate_growth_rate(cluster_posts)

        # Determine lifecycle stage
        post_counts = self._get_temporal_distribution(cluster_posts)
        lifecycle_stage = self.lifecycle_analyzer.determine_stage(
            post_counts, list(post_counts.keys())
        )

        # Calculate relevance score
        relevance_score = self._calculate_relevance(
            post_count, avg_engagement, growth_rate
        )

        # Generate topic ID and name
        topic_id = self._generate_topic_id(keywords)
        topic_name = self._generate_topic_name(keywords)

        return Topic(
            topic_id=topic_id,
            name=topic_name,
            keywords=keywords,
            description=f"Topic covering {', '.join(keywords[:3])}",
            cluster_id=cluster_id,
            lifecycle_stage=lifecycle_stage,
            first_seen=first_seen,
            last_seen=last_seen,
            post_count=post_count,
            total_engagement=total_engagement,
            avg_engagement=avg_engagement,
            growth_rate=growth_rate,
            relevance_score=relevance_score,
        )

    def _calculate_growth_rate(self, posts: List[Dict[str, Any]]) -> float:
        """Calculate 30-day growth rate."""
        if len(posts) < 2:
            return 0.0

        now = datetime.now()
        last_30_days = [p for p in posts if (now - p["created_at"]).days <= 30]
        prev_30_days = [
            p for p in posts if 30 < (now - p["created_at"]).days <= 60
        ]

        current_count = len(last_30_days)
        previous_count = len(prev_30_days) if prev_30_days else 1

        growth_rate = ((current_count - previous_count) / previous_count) * 100
        return round(growth_rate, 2)

    def _get_temporal_distribution(
        self, posts: List[Dict[str, Any]]
    ) -> Dict[datetime, int]:
        """Get post count distribution over time."""
        distribution: Dict[datetime, int] = {}
        for post in posts:
            date = post["created_at"].date()
            distribution[date] = distribution.get(date, 0) + 1
        return distribution

    def _calculate_relevance(
        self, post_count: int, avg_engagement: float, growth_rate: float
    ) -> float:
        """Calculate topic relevance score (0-1)."""
        # Normalize components
        count_score = min(post_count / 20, 1.0)  # Max at 20 posts
        engagement_score = min(avg_engagement / 1000, 1.0)  # Max at 1000
        growth_score = min(max(growth_rate / 100, 0), 1.0)  # -100% to +100%

        # Weighted average
        relevance = (count_score * 0.4) + (engagement_score * 0.3) + (growth_score * 0.3)

        return round(relevance, 3)

    def _generate_topic_id(self, keywords: List[str]) -> str:
        """Generate unique topic ID."""
        keyword_str = "_".join(sorted(keywords[:3]))
        hash_obj = hashlib.sha256(keyword_str.encode())
        return f"topic_{hash_obj.hexdigest()[:12]}"

    def _generate_topic_name(self, keywords: List[str]) -> str:
        """Generate human-readable topic name."""
        if not keywords:
            return "Untitled Topic"

        # Use top 2-3 keywords
        top_keywords = keywords[:3]
        return " & ".join([kw.title() for kw in top_keywords])

    def cluster_topics(
        self, topic_vectors: np.ndarray, topic_data: List[Dict[str, Any]]
    ) -> List[TopicCluster]:
        """
        Cluster similar topics together.

        Args:
            topic_vectors: TF-IDF vectors
            topic_data: Topic metadata

        Returns:
            List of topic clusters
        """
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(topic_vectors)

        # Group by similarity threshold
        clusters: Dict[int, List[int]] = {}
        assigned = set()

        for i in range(len(topic_data)):
            if i in assigned:
                continue

            # Find similar topics
            similar = [
                j
                for j in range(len(topic_data))
                if j not in assigned and similarity_matrix[i][j] > self.threshold
            ]

            if similar:
                cluster_id = len(clusters)
                clusters[cluster_id] = similar
                assigned.update(similar)

        # Build TopicCluster objects
        topic_clusters = []
        for cluster_id, indices in clusters.items():
            cluster_topics = [topic_data[i] for i in indices]

            # Aggregate metrics
            total_posts = sum(t.get("post_count", 0) for t in cluster_topics)
            avg_engagement = np.mean([t.get("avg_engagement", 0) for t in cluster_topics])

            # Extract centroid keywords
            all_keywords = []
            for topic in cluster_topics:
                all_keywords.extend(topic.get("keywords", []))
            from collections import Counter

            keyword_counts = Counter(all_keywords)
            centroid_keywords = [kw for kw, _ in keyword_counts.most_common(5)]

            topic_cluster = TopicCluster(
                cluster_id=cluster_id,
                name=self._generate_topic_name(centroid_keywords),
                centroid_keywords=centroid_keywords,
                total_posts=total_posts,
                avg_engagement=float(avg_engagement),
            )

            topic_clusters.append(topic_cluster)

        return topic_clusters