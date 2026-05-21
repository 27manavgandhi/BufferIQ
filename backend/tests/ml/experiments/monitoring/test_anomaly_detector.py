"""Tests for anomaly detector."""

from bufferiq.ml.experiments.monitoring.anomaly_detector import AnomalyDetector


class TestAnomalyDetector:
    """Test AnomalyDetector."""

    def setup_method(self):
        """Setup test."""
        self.detector = AnomalyDetector(z_threshold=3.0)

    def test_detect_anomalies(self):
        """Test anomaly detection."""
        # Data with outlier
        time_series = [100, 102, 98, 101, 200, 99, 100]

        result = self.detector.detect_anomalies(time_series)

        assert result["num_anomalies"] > 0
        assert 4 in result["anomaly_indices"]

    def test_no_anomalies(self):
        """Test no anomalies."""
        time_series = [100, 102, 98, 101, 99, 100]

        result = self.detector.detect_anomalies(time_series)

        assert result["num_anomalies"] == 0

    def test_detect_anomalies_mad(self):
        """Test MAD-based anomaly detection."""
        time_series = [100, 102, 98, 101, 200, 99, 100]

        result = self.detector.detect_anomalies_mad(time_series)

        assert result["num_anomalies"] > 0
