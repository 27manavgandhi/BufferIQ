"""Add audience segments table.

Revision ID: 028
Revises: 027
Create Date: 2025-05-30 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


revision = "028"
down_revision = "027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create audience_segments table."""
    op.create_table(
        "audience_segments",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("n_members", sa.Integer(), nullable=False),
        sa.Column("size_percentage", sa.Float(), nullable=False),
        sa.Column("centroid", sa.JSON(), nullable=True),
        sa.Column("clustering_algorithm", sa.String(), nullable=False),
        sa.Column("silhouette_score", sa.Float(), nullable=True),
        sa.Column("stability_score", sa.Float(), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_audience_segments_platform", "audience_segments", ["platform"]
    )
    op.create_index(
        "ix_audience_segments_is_active", "audience_segments", ["is_active"]
    )
    op.create_index(
        "ix_audience_segments_created_at", "audience_segments", ["created_at"]
    )


def downgrade() -> None:
    """Drop audience_segments table."""
    op.drop_table("audience_segments")