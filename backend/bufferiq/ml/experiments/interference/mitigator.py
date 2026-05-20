"""
Interference mitigator.

Provides strategies to mitigate interference effects.

Example:
```python
    mitigator = InterferenceMitigator()
    
    strategy = mitigator.recommend_strategy(
        network_structure=network,
        interference_detected=True
    )
```
"""

from typing import Dict, List, Set, Tuple

from bufferiq.ml.experiments.interference.network_analyzer import NetworkAnalyzer


class InterferenceMitigator:
    """
    Mitigate interference effects.

    Example:
```python
        mitigator = InterferenceMitigator()

        strategy = mitigator.recommend_mitigation(
            has_interference=True,
            num_clusters=20,
            mean_cluster_size=50
        )

        print(f"Strategy: {strategy['strategy']}")
        print(f"Reason: {strategy['reason']}")
```
    """

    def __init__(self) -> None:
        """Initialize mitigator."""
        self.network_analyzer = NetworkAnalyzer()

    def recommend_mitigation(
        self,
        has_interference: bool,
        num_clusters: int,
        mean_cluster_size: float,
    ) -> Dict[str, str]:
        """
        Recommend mitigation strategy.

        Args:
            has_interference: Whether interference detected
            num_clusters: Number of network clusters
            mean_cluster_size: Mean cluster size

        Returns:
            Mitigation strategy
        """
        if not has_interference:
            return {
                "strategy": "individual_randomization",
                "reason": "No interference detected",
            }

        # Check if cluster randomization is viable
        if num_clusters >= 10 and mean_cluster_size >= 10:
            return {
                "strategy": "cluster_randomization",
                "reason": "Sufficient clusters for cluster-level randomization",
            }

        # Check if buffer zones can help
        if mean_cluster_size > 50:
            return {
                "strategy": "buffer_zones",
                "reason": "Create buffer zones between treatment and control",
            }

        # Last resort
        return {
            "strategy": "accept_bias",
            "reason": "Interference unavoidable - document and analyze",
        }

    def create_buffer_zones(
        self,
        treatment_user_ids: List[str],
        control_user_ids: List[str],
        edges: List[Tuple[str, str]],
        buffer_distance: int = 1,
    ) -> Dict[str, List[str]]:
        """
        Create buffer zones around treatment.

        Args:
            treatment_user_ids: Treatment users
            control_user_ids: Control users
            edges: Network edges
            buffer_distance: Distance for buffer

        Returns:
            Dictionary with buffered assignments
        """
        # Build adjacency
        adjacency: Dict[str, Set[str]] = {}
        all_users = set(treatment_user_ids) | set(control_user_ids)

        for uid in all_users:
            adjacency[uid] = set()

        for u1, u2 in edges:
            if u1 in adjacency and u2 in adjacency:
                adjacency[u1].add(u2)
                adjacency[u2].add(u1)

        # Find control users within buffer distance of treatment
        buffer_users = set()

        def find_neighbors(uid: str, distance: int, visited: Set[str]) -> None:
            if distance == 0:
                return

            visited.add(uid)

            for neighbor in adjacency[uid]:
                if neighbor not in visited:
                    if neighbor in control_user_ids:
                        buffer_users.add(neighbor)
                    find_neighbors(neighbor, distance - 1, visited)

        for t_uid in treatment_user_ids:
            find_neighbors(t_uid, buffer_distance, set())

        # Create final assignments
        final_control = [
            uid for uid in control_user_ids if uid not in buffer_users
        ]

        return {
            "treatment": treatment_user_ids,
            "control": final_control,
            "buffer": list(buffer_users),
        }

    def assign_clusters(
        self,
        clusters: List[Set[str]],
        treatment_fraction: float = 0.5,
    ) -> Dict[str, List[str]]:
        """
        Assign clusters to treatment/control.

        Args:
            clusters: Network clusters
            treatment_fraction: Fraction for treatment

        Returns:
            Cluster assignments
        """
        import random

        # Sort clusters by size for balanced assignment
        sorted_clusters = sorted(clusters, key=len, reverse=True)

        treatment_users = []
        control_users = []

        # Assign clusters
        for i, cluster in enumerate(sorted_clusters):
            if i % 2 == 0:
                # Alternate assignment for balance
                if len(treatment_users) < len(control_users):
                    treatment_users.extend(cluster)
                else:
                    control_users.extend(cluster)
            else:
                if len(control_users) < len(treatment_users):
                    control_users.extend(cluster)
                else:
                    treatment_users.extend(cluster)

        return {
            "treatment": treatment_users,
            "control": control_users,
            "num_treatment_clusters": sum(
                1 for c in sorted_clusters if any(u in treatment_users for u in c)
            ),
            "num_control_clusters": sum(
                1 for c in sorted_clusters if any(u in control_users for u in c)
            ),
        }