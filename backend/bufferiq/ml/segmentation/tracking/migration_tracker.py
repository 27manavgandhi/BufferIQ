"""Track member migrations between segments."""

from typing import Any, Dict, List, Set

from bufferiq.ml.segmentation.types import SegmentSnapshot


class MigrationTracker:
    """Track how members move between segments."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize migration tracker."""
        self.config = config or {}

    def track_migrations(
        self,
        previous_snapshots: Dict[str, SegmentSnapshot],
        current_snapshots: Dict[str, SegmentSnapshot],
    ) -> Dict[str, Any]:
        """
        Track member migrations between snapshots.

        Args:
            previous_snapshots: Previous snapshots by segment ID
            current_snapshots: Current snapshots by segment ID

        Returns:
            Migration analysis
        """
        migrations: Dict[str, List[str]] = {}

        for segment_id, current_snapshot in current_snapshots.items():
            if segment_id not in previous_snapshots:
                migrations[segment_id] = {"new_members": current_snapshot.member_ids}
                continue

            previous_snapshot = previous_snapshots[segment_id]
            previous_members = set(previous_snapshot.member_ids)
            current_members = set(current_snapshot.member_ids)

            new_members = list(current_members - previous_members)
            lost_members = list(previous_members - current_members)

            migrations[segment_id] = {
                "new_members": new_members,
                "lost_members": lost_members,
                "retention_rate": len(previous_members & current_members)
                / len(previous_members)
                if previous_members
                else 0.0,
            }

        return migrations

    def analyze_member_movement(
        self,
        member_id: str,
        segment_histories: Dict[str, List[SegmentSnapshot]],
    ) -> Dict[str, Any]:
        """
        Analyze movement pattern of a specific member.

        Args:
            member_id: Member ID to analyze
            segment_histories: History of segment snapshots

        Returns:
            Member movement analysis
        """
        segments_visited = []

        for segment_id, snapshots in segment_histories.items():
            for snapshot in snapshots:
                if member_id in snapshot.member_ids:
                    segments_visited.append(
                        {
                            "segment_id": segment_id,
                            "timestamp": snapshot.timestamp,
                        }
                    )

        # Sort by timestamp
        segments_visited.sort(key=lambda x: x["timestamp"])

        return {
            "member_id": member_id,
            "segments_visited": [s["segment_id"] for s in segments_visited],
            "movement_count": len(segments_visited) - 1,
            "current_segment": segments_visited[-1]["segment_id"]
            if segments_visited
            else None,
        }