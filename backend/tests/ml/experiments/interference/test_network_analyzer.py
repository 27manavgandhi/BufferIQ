"""Tests for network analyzer."""

from bufferiq.ml.experiments.interference.network_analyzer import NetworkAnalyzer


class TestNetworkAnalyzer:
    """Test NetworkAnalyzer."""

    def setup_method(self):
        """Setup test."""
        self.analyzer = NetworkAnalyzer()

    def test_find_clusters(self):
        """Test cluster finding."""
        user_ids = ["u1", "u2", "u3", "u4", "u5"]
        edges = [("u1", "u2"), ("u3", "u4")]

        clusters = self.analyzer.find_clusters(user_ids, edges)

        assert len(clusters) == 3  # {u1, u2}, {u3, u4}, {u5}

    def test_calculate_cluster_sizes(self):
        """Test cluster size calculation."""
        clusters = [{"u1", "u2"}, {"u3", "u4", "u5"}]

        stats = self.analyzer.calculate_cluster_sizes(clusters)

        assert stats["num_clusters"] == 2
        assert stats["mean_size"] == 2.5
        assert stats["max_size"] == 3
        assert stats["min_size"] == 2

    def test_recommend_cluster_randomization(self):
        """Test cluster randomization recommendation."""
        # Many large clusters
        clusters = [{f"u{i}" for i in range(j * 10, (j + 1) * 10)} for j in range(20)]

        rec = self.analyzer.recommend_cluster_randomization(clusters)

        assert rec["recommend_cluster_randomization"] is True
