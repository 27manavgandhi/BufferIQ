"""
Network analyzer.

Analyzes network structure for experiment design.

Example:
```python
    analyzer = NetworkAnalyzer()
    
    clusters = analyzer.find_clusters(
        user_ids=users,
        edges=network_edges
    )
```
"""

from typing import Dict, List, Set, Tuple

import numpy as np


class NetworkAnalyzer:
    """
    Analyze network structure.

    Example:
```python
        analyzer = NetworkAnalyzer()

        clusters = analyzer.find_clusters(
            user_ids=["u1", "u2", "u3", "u4"],
            edges=[("u1", "u2"), ("u3", "u4")]
        )

        print(f"Found {len(clusters)} clusters")
```
    """

    def find_clusters(
        self, user_ids: List[str], edges: List[Tuple[str, str]]
    ) -> List[Set[str]]:
        """
        Find connected components (clusters) in network.

        Args:
            user_ids: List of user IDs
            edges: Network edges

        Returns:
            List of clusters (sets of user IDs)
        """
        # Build adjacency list
        adjacency: Dict[str, Set[str]] = {uid: set() for uid in user_ids}

        for u1, u2 in edges:
            if u1 in adjacency and u2 in adjacency:
                adjacency[u1].add(u2)
                adjacency[u2].add(u1)

        # Find connected components using DFS
        visited = set()
        clusters = []

        def dfs(node: str, cluster: Set[str]) -> None:
            visited.add(node)
            cluster.add(node)

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for uid in user_ids:
            if uid not in visited:
                cluster: Set[str] = set()
                dfs(uid, cluster)
                clusters.append(cluster)

        return clusters

    def calculate_cluster_sizes(
        self, clusters: List[Set[str]]
    ) -> Dict[str, any]:
        """
        Calculate cluster size statistics.

        Args:
            clusters: List of clusters

        Returns:
            Statistics dictionary
        """
        sizes = [len(c) for c in clusters]

        if not sizes:
            return {
                "num_clusters": 0,
                "mean_size": 0.0,
                "median_size": 0.0,
                "max_size": 0,
                "min_size": 0,
            }

        return {
            "num_clusters": len(clusters),
            "mean_size": float(np.mean(sizes)),
            "median_size": float(np.median(sizes)),
            "max_size": int(np.max(sizes)),
            "min_size": int(np.min(sizes)),
        }

    def recommend_cluster_randomization(
        self, clusters: List[Set[str]], min_cluster_size: int = 10
    ) -> Dict[str, any]:
        """
        Recommend whether to use cluster randomization.

        Args:
            clusters: Network clusters
            min_cluster_size: Minimum viable cluster size

        Returns:
            Recommendation
        """
        cluster_stats = self.calculate_cluster_sizes(clusters)

        # Count viable clusters
        viable_clusters = [c for c in clusters if len(c) >= min_cluster_size]

        use_cluster_randomization = len(viable_clusters) >= 10

        return {
            "recommend_cluster_randomization": use_cluster_randomization,
            "total_clusters": cluster_stats["num_clusters"],
            "viable_clusters": len(viable_clusters),
            "mean_cluster_size": cluster_stats["mean_size"],
            "reason": (
                "Sufficient large clusters"
                if use_cluster_randomization
                else "Too few viable clusters"
            ),
        }