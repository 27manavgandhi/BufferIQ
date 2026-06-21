"""Add personas table.

Revision ID: 029
Revises: 028
Create Date: 2025-05-30 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa


revision = "029"
down_revision = "028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create personas table."""
    op.create_table(
        "personas",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("segment_id", sa.String(), nullable=False),
        sa.Column("platform", sa.String(), nullable=False),
        sa.Column("persona_name", sa.String(), nullable=False),
        sa.Column("persona_description", sa.Text(), nullable=True),
        sa.Column("estimated_age_min", sa.Integer(), nullable=True),
        sa.Column("estimated_age_max", sa.Integer(), nullable=True),
        sa.Column("estimated_location", sa.String(), nullable=True),
        sa.Column("verified_ratio", sa.Float(), nullable=True),
        sa.Column("avg_engagement_rate", sa.Float(), nullable=True),
        sa.Column("primary_interaction_type", sa.String(), nullable=True),
        sa.Column("content_preferences", sa.JSON(), nullable=True),
        sa.Column("peak_activity_hours", sa.JSON(), nullable=True),
        sa.Column("peak_activity_days", sa.JSON(), nullable=True),
        sa.Column("primary_topics", sa.JSON(), nullable=True),
        sa.Column("secondary_topics", sa.JSON(), nullable=True),
        sa.Column("avoided_topics", sa.JSON(), nullable=True),
        sa.Column("engagement_potential_score", sa.Float(), nullable=True),
        sa.Column("growth_potential_score", sa.Float(), nullable=True),
        sa.Column("retention_risk_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["segment_id"], ["audience_segments.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index("ix_personas_segment_id", "personas", ["segment_id"])
    op.create_index("ix_personas_platform", "personas", ["platform"])
    op.create_index("ix_personas_created_at", "personas", ["created_at"])


def downgrade() -> None:
    """Drop personas table."""
    op.drop_table("personas")