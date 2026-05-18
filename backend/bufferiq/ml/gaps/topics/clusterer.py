"""Topic clustering using DBSCAN."""

from typing import Any, Dict, List
import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class TopicClusterer:
    """
    Cluster topics using DBSCAN algorithm.

    Groups similar content into topic clusters based on TF-IDF similarity.
    """

    def __init__(
        self, similarity_threshold: float = 0.3, min_samples: int = 2, eps: float = 0.5
    ):
        """
        Initialize clusterer.

        Args:
            similarity_threshold: Minimum similarity for clustering
            min_samples: Minimum samples for core point
            eps: DBSCAN epsilon parameter
        """
        self.similarity_threshold = similarity_threshold
        self.min_samples = min_samples
        self.eps = eps

    def cluster(
        self, tfidf_matrix: np.ndarray, posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Cluster posts into topics.

        Args:
            tfidf_matrix: TF-IDF matrix of posts
            posts: Post metadata

        Returns:
            List of cluster dictionaries with post indices and keywords
        """
        # Convert to dense array
        dense_matrix = tfidf_matrix.toarray()

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        cluster_labels = dbscan.fit_predict(dense_matrix)

        # Group posts by cluster
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # Noise
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        # Extract keywords for each cluster
        cluster_data = []
        for cluster_id, post_indices in clusters.items():
            # Get cluster centroid
            cluster_vectors = dense_matrix[post_indices]
            centroid = np.mean(cluster_vectors, axis=0)

            # Extract top keywords
            keywords = self._extract_keywords(centroid, tfidf_matrix)

            cluster_data.append(
                {
                    "cluster_id": cluster_id,
                    "post_indices": post_indices,
                    "keywords": keywords,
                }
            )

        logger.info(f"Found {len(cluster_data)} clusters from {len(posts)} posts")

        return cluster_data

    def _extract_keywords(
        self, centroid: np.ndarray, tfidf_matrix: np.ndarray, top_n: int = 10
    ) -> List[str]:
        """Extract top keywords from cluster centroid."""
        # Get feature names from vectorizer
        # Note: In production, pass vectorizer to get actual feature names
        # For now, generate placeholder keywords
        top_indices = np.argsort(centroid)[-top_n:][::-1]
        keywords = [f"keyword_{idx}" for idx in top_indices]
        return keywords

    def calculate_cluster_coherence(
        self, cluster_vectors: np.ndarray
    ) -> float:
        """
        Calculate cluster coherence score.

        Args:
            cluster_vectors: Vectors in cluster

        Returns:
            Coherence score (0-1)
        """
        if len(cluster_vectors) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = cosine_similarity(cluster_vectors)

        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(similarities, k=1)
        pairwise_sims = similarities[triu_indices]

        # Average similarity
        coherence = float(np.mean(pairwise_sims))

        return coherence