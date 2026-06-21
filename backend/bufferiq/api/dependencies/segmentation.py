"""Dependencies for segmentation API."""

from typing import Any, Dict

from sqlalchemy.orm import Session

from bufferiq.database import get_db
from bufferiq.ml.segmentation.intelligence.service import SegmentationIntelligenceService
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
from bufferiq.api.services.segmentation_service import SegmentationAPIService


def get_intelligence_service() -> SegmentationIntelligenceService:
    """Get intelligence service."""
    config: Dict[str, Any] = {}

    service = SegmentationIntelligenceService(config=config)
    return service


def get_segmentation_service(
    db: Session = None,
) -> SegmentationAPIService:
    """Get segmentation API service."""
    if db is None:
        db_session = next(get_db())
    else:
        db_session = db

    intelligence_service = get_intelligence_service()
    service = SegmentationAPIService(
        intelligence_service=intelligence_service,
        db=db_session,
    )

    return service