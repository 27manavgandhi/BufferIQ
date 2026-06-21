"""Main intelligence service orchestrating all components."""

from typing import Any, Dict, List, Optional

import numpy as np
import asyncio
from datetime import datetime

from bufferiq.ml.segmentation.types import (
    AudienceDataPoint,
    SUPPORTED_PLATFORMS,
    SegmentSnapshot,
)
from bufferiq.ml.segmentation.exceptions import (
    UnsupportedPlatformError,
    InsufficientDataError,
)
from bufferiq.ml.segmentation.preprocessing.preprocessor import (
    AudienceDataPreprocessor,
)
from bufferiq.ml.segmentation.clustering.optimizer import ClusteringOptimizer
from bufferiq.ml.segmentation.clustering.kmeans import KMeansClusterer
from bufferiq.ml.segmentation.clustering.dbscan import DBSCANClusterer
from bufferiq.ml.segmentation.clustering.hierarchical import HierarchicalClusterer
from bufferiq.ml.segmentation.clustering.gmm import GMMClusterer
from bufferiq.ml.segmentation.personas.persona_builder import PersonaBuilder
from bufferiq.ml.segmentation.personas.demographic_inferrer import DemographicInferrer
from bufferiq.ml.segmentation.personas.behavioral_profiler import BehavioralProfiler
from bufferiq.ml.segmentation.personas.interest_mapper import InterestMapper
from bufferiq.ml.segmentation.personas.content_preference_modeler import (
    ContentPreferenceModeler,
)
from bufferiq.ml.segmentation.personas.namer import PersonaNamer
from bufferiq.ml.segmentation.tracking.evolution_tracker import SegmentEvolutionTracker
from bufferiq.ml.segmentation.recommendations.engine import RecommendationEngine
from bufferiq.ml.segmentation.prediction.predictor import SegmentEngagementPredictor


class SegmentationIntelligenceService:
    """
    Unified service for audience segmentation intelligence.

    Orchestrates:
    - Data preprocessing
    - Clustering
    - Persona generation
    - Evolution tracking
    - Recommendations
    - Engagement prediction
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize intelligence service."""
        self.config = config or {}

        # Initialize components
        self.preprocessor = AudienceDataPreprocessor(
            self.config.get("preprocessing", {})
        )
        self.clustering_optimizer = ClusteringOptimizer(
            self.config.get("clustering_optimizer", {})
        )
        self.kmeans_clusterer = KMeansClusterer(self.config.get("kmeans", {}))
        self.dbscan_clusterer = DBSCANClusterer(self.config.get("dbscan", {}))
        self.hierarchical_clusterer = HierarchicalClusterer(
            self.config.get("hierarchical", {})
        )
        self.gmm_clusterer = GMMClusterer(self.config.get("gmm", {}))

        # Persona components
        self.demographic_inferrer = DemographicInferrer(
            self.config.get("demographic", {})
        )
        self.behavioral_profiler = BehavioralProfiler(
            self.config.get("behavioral", {})
        )
        self.interest_mapper = InterestMapper(self.config.get("interest", {}))
        self.content_preference_modeler = ContentPreferenceModeler(
            self.config.get("content_preference", {})
        )
        self.namer = PersonaNamer(self.config.get("namer", {}))
        self.persona_builder = PersonaBuilder(self.config.get("persona", {}))

        # Tracking and recommendation
        self.evolution_tracker = SegmentEvolutionTracker(
            self.config.get("evolution", {})
        )
        self.recommendation_engine = RecommendationEngine(
            self.config.get("recommendation", {})
        )
        self.predictor = SegmentEngagementPredictor(self.config.get("prediction", {}))

    async def segment_audience(
        self,
        audience_data: List[AudienceDataPoint],
        platform: str,
        historical_snapshots: Optional[Dict[str, List[SegmentSnapshot]]] = None,
    ) -> Dict[str, Any]:
        """
        Full audience segmentation pipeline.

        Args:
            audience_data: List of audience data points
            platform: Platform type
            historical_snapshots: Past snapshots keyed by segment_id

        Returns:
            Complete segmentation result

        Raises:
            UnsupportedPlatformError: If platform not supported
            InsufficientDataError: If not enough data
        """
        if platform not in SUPPORTED_PLATFORMS:
            raise UnsupportedPlatformError(platform, SUPPORTED_PLATFORMS)

        if len(audience_data) < 10:
            raise InsufficientDataError(
                "Need at least 10 audience members for segmentation"
            )

        start_time = datetime.utcnow()

        # Step 1: Preprocess data
        processed_features = self.preprocessor.process(audience_data, platform)
        feature_matrix = np.array([pf.feature_vector for pf in processed_features])

        # Step 2: Find optimal clustering
        optimal_config = self.clustering_optimizer.find_optimal(
            feature_matrix, platform
        )

        # Step 3: Apply clustering
        clustering_result = await self._apply_clustering(
            feature_matrix, optimal_config, platform
        )

        # Step 4: Build personas for each cluster
        personas = []
        recommendations = []
        evolutions = []

        for cluster_id in range(clustering_result.n_clusters):
            cluster_mask = clustering_result.labels == cluster_id
            cluster_members = [
                audience_data[i]
                for i, mask in enumerate(cluster_mask)
                if mask
            ]
            cluster_features = feature_matrix[cluster_mask]

            if not cluster_members:
                continue

            # Build persona
            persona = self.persona_builder.build(
                cluster_id=cluster_id,
                cluster_members=cluster_members,
                feature_matrix=cluster_features,
                platform=platform,
            )
            persona.size_percentage = (
                len(cluster_members) / len(audience_data) * 100
            )
            personas.append(persona)

            # Generate recommendations
            rec = self.recommendation_engine.generate(persona, platform)
            recommendations.append(rec)

            # Track evolution if history available
            if historical_snapshots and persona.segment_id in historical_snapshots:
                current_snapshot = self._create_snapshot(
                    persona, cluster_features, cluster_members
                )
                evolution = self.evolution_tracker.track(
                    current_snapshot=current_snapshot,
                    historical_snapshots=historical_snapshots[persona.segment_id],
                    platform=platform,
                )
                evolutions.append(evolution)

        processing_time = (
            datetime.utcnow() - start_time
        ).total_seconds() * 1000

        return {
            "platform": platform,
            "total_audience_size": len(audience_data),
            "n_segments": clustering_result.n_clusters,
            "clustering_algorithm": optimal_config.algorithm,
            "clustering_quality": {
                "silhouette_score": float(clustering_result.silhouette_score),
                "calinski_harabasz_score": float(
                    clustering_result.calinski_harabasz_score
                ),
                "davies_bouldin_score": float(clustering_result.davies_bouldin_score),
                "stability_score": float(clustering_result.stability_score),
            },
            "personas": [p.to_dict() for p in personas],
            "recommendations": [r.to_dict() for r in recommendations],
            "evolutions": [e.to_dict() for e in evolutions],
            "processing_time_ms": processing_time,
            "segmented_at": start_time.isoformat(),
        }

    async def _apply_clustering(
        self, feature_matrix: np.ndarray, optimal_config: Any, platform: str
    ) -> Any:
        """Apply selected clustering algorithm."""
        if optimal_config.algorithm == "kmeans":
            return self.kmeans_clusterer.fit(
                feature_matrix, optimal_config.n_clusters, platform
            )
        elif optimal_config.algorithm == "dbscan":
            return self.dbscan_clusterer.fit(feature_matrix, platform)
        elif optimal_config.algorithm == "hierarchical":
            return self.hierarchical_clusterer.fit(
                feature_matrix, optimal_config.n_clusters, platform
            )
        elif optimal_config.algorithm == "gmm":
            return self.gmm_clusterer.fit(
                feature_matrix, optimal_config.n_clusters, platform
            )
        else:
            return self.kmeans_clusterer.fit(
                feature_matrix, optimal_config.n_clusters, platform
            )

    def _create_snapshot(
        self,
        persona: Any,
        cluster_features: np.ndarray,
        cluster_members: List[AudienceDataPoint],
    ) -> SegmentSnapshot:
        """Create segment snapshot."""
        centroid = cluster_features.mean(axis=0) if len(cluster_features) > 0 else None

        return SegmentSnapshot(
            segment_id=persona.segment_id,
            platform=persona.platform,
            timestamp=datetime.utcnow(),
            size=persona.size,
            avg_engagement_rate=persona.avg_engagement_rate,
            member_ids=[m.user_id for m in cluster_members],
            centroid=centroid,
            health_score=persona.engagement_potential_score,
        )