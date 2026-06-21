"""Engagement prediction for audience segments."""

from bufferiq.ml.segmentation.prediction.segment_model import SegmentEngagementModel
from bufferiq.ml.segmentation.prediction.cross_segment import CrossSegmentAnalyzer
from bufferiq.ml.segmentation.prediction.calibrator import ModelCalibrator
from bufferiq.ml.segmentation.prediction.predictor import SegmentEngagementPredictor

__all__ = [
    "SegmentEngagementModel",
    "CrossSegmentAnalyzer",
    "ModelCalibrator",
    "SegmentEngagementPredictor",
]