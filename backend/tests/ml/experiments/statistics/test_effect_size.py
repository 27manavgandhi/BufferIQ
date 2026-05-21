"""Tests for effect size calculator."""

import numpy as np
from bufferiq.ml.experiments.statistics.effect_size import EffectSizeCalculator


class TestEffectSizeCalculator:
    """Test EffectSizeCalculator."""

    def setup_method(self):
        """Setup test."""
        self.calculator = EffectSizeCalculator()

    def test_cohens_d(self):
        """Test Cohen's d calculation."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 1000)
        treatment = np.random.normal(110, 15, 1000)

        d = self.calculator.cohens_d(control, treatment)

        assert d > 0
        # Should be medium to large effect
        assert 0.5 < abs(d) < 1.0

    def test_hedges_g(self):
        """Test Hedge's g calculation."""
        np.random.seed(42)
        control = np.random.normal(100, 15, 100)
        treatment = np.random.normal(110, 15, 100)

        g = self.calculator.hedges_g(control, treatment)

        # Hedge's g should be slightly smaller than Cohen's d
        d = self.calculator.cohens_d(control, treatment)
        assert abs(g) < abs(d)

    def test_cliffs_delta(self):
        """Test Cliff's delta calculation."""
        control = np.array([1, 2, 3, 4, 5])
        treatment = np.array([6, 7, 8, 9, 10])

        delta = self.calculator.cliffs_delta(control, treatment)

        # All treatment > control
        assert delta == 1.0

    def test_cohens_h(self):
        """Test Cohen's h for proportions."""
        h = self.calculator.cohens_h(0.5, 0.6)

        assert h > 0
