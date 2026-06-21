"""Add media analysis table.

Revision ID: 025_add_media_analysis
Revises: 024_add_experiments
Create Date: 2024-01-21 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '025_add_media_analysis'
down_revision = '024_add_experiments'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        'media_analyses',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('post_id', sa.String(), nullable=False),
        sa.Column('media_type', sa.String(), nullable=False),
        sa.Column('media_url', sa.Text(), nullable=False),
        sa.Column('platform', sa.String(), nullable=False),
        sa.Column('analysis_data', sa.JSON(), nullable=False),
        sa.Column('predicted_engagement', sa.Float(), nullable=True),
        sa.Column('actual_engagement', sa.Float(), nullable=True),
        sa.Column('prediction_error', sa.Float(), nullable=True),
        sa.Column('processing_time_ms', sa.Float(), nullable=True),
        sa.Column('analyzed_at', sa.DateTime(), nullable=False),
        sa.Column('analyzer_version', sa.String(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_media_analyses_post_id', 'media_analyses', ['post_id'])
    op.create_index('ix_media_analyses_platform', 'media_analyses', ['platform'])
    op.create_index('ix_media_analyses_media_type', 'media_analyses', ['media_type'])
    op.create_index('ix_media_analyses_analyzed_at', 'media_analyses', ['analyzed_at'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_media_analyses_analyzed_at', 'media_analyses')
    op.drop_index('ix_media_analyses_media_type', 'media_analyses')
    op.drop_index('ix_media_analyses_platform', 'media_analyses')
    op.drop_index('ix_media_analyses_post_id', 'media_analyses')
    op.drop_table('media_analyses')