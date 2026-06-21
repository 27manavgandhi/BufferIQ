"""Add segment evolution tracking.

Revision ID: 030
Revises: 029
Create Date: 2025-05-30 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


revision = "030"
down_revision = "029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create segment_evolution table."""
    op.create_table(
        "segment_evolution",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("segment_id", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("snapshot_at", sa.DateTime(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("avg_engagement_rate", sa.Float(), nullable=True),
        sa.Column("health_score", sa.Float(), nullable=True),
        sa.Column("centroid", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["segment_id"], ["audience_segments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_segment_evolution_segment_id", "segment_evolution", ["segment_id"]
    )
    op.create_index(
        "ix_segment_evolution_platform", "segment_evolution", ["platform"]
    )
    op.create_index(
        "ix_segment_evolution_snapshot_at", "segment_evolution", ["snapshot_at"]
    )
    op.create_index(
        "ix_segment_evolution_created_at", "segment_evolution", ["created_at"]
    )


def downgrade() -> None:
    """Drop segment_evolution table."""
    op.drop_table("segment_evolution")