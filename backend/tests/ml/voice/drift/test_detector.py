"""Tests for voice drift detector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

from bufferiq.ml.voice.drift.detector import VoiceDriftDetector, DriftAlert


class TestVoiceDriftDetector:
    """Test voice drift detector."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database session."""
        return Mock()
    
    @pytest.fixture
    def detector(self, mock_db):
        """Create detector instance."""
        return VoiceDriftDetector(mock_db)
    
    @pytest.mark.asyncio
    async def test_detect_basic(self, detector):
        """Test basic drift detection."""
        alert = await detector.detect(
            brand_id="brand123",
            platform="linkedin",
            window_days=30,
        )
        
        assert isinstance(alert, DriftAlert)
        assert isinstance(alert.drift_detected, bool)
    
    @pytest.mark.asyncio
    async def test_detect_invalid_platform_raises_error(self, detector):
        """Test invalid platform raises error."""
        with pytest.raises(ValueError, match="not supported"):
            await detector.detect(
                brand_id="brand123",
                platform="facebook",
            )
    
    @pytest.mark.asyncio
    async def test_detect_insufficient_baseline_raises_error(self, detector):
        """Test insufficient baseline data raises error."""
        with patch.object(detector, '_fetch_baseline_data', return_value=[]):
            with pytest.raises(ValueError, match="Insufficient baseline"):
                await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
    
    @pytest.mark.asyncio
    async def test_detect_insufficient_recent_raises_error(self, detector):
        """Test insufficient recent data raises error."""
        baseline = [{"formality": 70.0}] * 30
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=[]):
                with pytest.raises(ValueError, match="Insufficient recent"):
                    await detector.detect(
                        brand_id="brand123",
                        platform="linkedin",
                    )
    
    @pytest.mark.asyncio
    async def test_detect_no_drift(self, detector):
        """Test detection with no drift."""
        # Mock stable data
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        # Should detect no significant drift
        assert alert.drift_score < 20
    
    @pytest.mark.asyncio
    async def test_detect_with_drift(self, detector):
        """Test detection with drift."""
        # Mock data with drift
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 85.0, "complexity": 50.0, "emoji_density": 1.0, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        # Should detect drift
        assert alert.drift_detected is True
        assert alert.drift_score > 0
    
    @pytest.mark.asyncio
    async def test_drift_type_classification(self, detector):
        """Test drift type is classified."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 85.0, "complexity": 50.0, "emoji_density": 1.0, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        assert alert.drift_type in ["gradual", "sudden", "stable"]
    
    @pytest.mark.asyncio
    async def test_affected_dimensions_identified(self, detector):
        """Test affected dimensions are identified."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 85.0, "complexity": 50.0, "emoji_density": 1.0, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        assert isinstance(alert.affected_dimensions, list)
    
    @pytest.mark.asyncio
    async def test_severity_classification(self, detector):
        """Test severity is classified."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        assert alert.severity in ["low", "medium", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_statistical_tests_performed(self, detector):
        """Test statistical tests are performed."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 75.0, "complexity": 50.0, "emoji_density": 1.5, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        assert isinstance(alert.t_statistic, float)
        assert isinstance(alert.p_value, float)
        assert 0 <= alert.confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_likely_causes_generated(self, detector):
        """Test likely causes are generated."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 85.0, "complexity": 50.0, "emoji_density": 1.0, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        assert isinstance(alert.likely_causes, list)
        assert len(alert.likely_causes) > 0
    
    @pytest.mark.asyncio
    async def test_drift_timeline_created(self, detector):
        """Test drift timeline is created."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0, "timestamp": datetime.utcnow()}] * 30
        recent = [{"formality": 75.0, "complexity": 50.0, "emoji_density": 1.5, "timestamp": datetime.utcnow()}] * 15
        
        with patch.object(detector, '_fetch_baseline_data', return_value=baseline):
            with patch.object(detector, '_fetch_recent_data', return_value=recent):
                alert = await detector.detect(
                    brand_id="brand123",
                    platform="linkedin",
                )
        
        assert isinstance(alert.drift_timeline, pd.DataFrame)
    
    def test_perform_t_test(self, detector):
        """Test t-test performance."""
        baseline = [{"formality": 70.0}] * 30
        recent = [{"formality": 75.0}] * 15
        
        drift_detected, t_stat, p_value = detector._perform_t_test(baseline, recent)
        
        assert isinstance(drift_detected, bool)
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
    
    def test_calculate_drift_score(self, detector):
        """Test drift score calculation."""
        baseline = [{"formality": 70.0, "complexity": 50.0, "emoji_density": 2.0}] * 30
        recent = [{"formality": 80.0, "complexity": 55.0, "emoji_density": 1.0}] * 15
        
        score = detector._calculate_drift_score(baseline, recent)
        
        assert 0 <= score <= 100
    
    def test_identify_drift_type(self, detector):
        """Test drift type identification."""
        baseline = [{"formality": 70.0, "timestamp": datetime.utcnow() - timedelta(days=i)} for i in range(30)]
        recent = [{"formality": 80.0, "timestamp": datetime.utcnow() - timedelta(days=i)} for i in range(15)]
        
        all_data = baseline + recent
        drift_type = detector._identify_drift_type(all_data, [])
        
        assert drift_type in ["gradual", "sudden", "stable", "unknown"]
    
    def test_determine_severity_low(self, detector):
        """Test low severity determination."""
        severity = detector._determine_severity(10.0)
        
        assert severity == "low"
    
    def test_determine_severity_medium(self, detector):
        """Test medium severity determination."""
        severity = detector._determine_severity(20.0)
        
        assert severity == "medium"
    
    def test_determine_severity_high(self, detector):
        """Test high severity determination."""
        severity = detector._determine_severity(35.0)
        
        assert severity == "high"
    
    def test_determine_severity_critical(self, detector):
        """Test critical severity determination."""
        severity = detector._determine_severity(60.0)
        
        assert severity == "critical"