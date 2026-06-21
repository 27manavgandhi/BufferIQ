"""Add visual features table.

Revision ID: 026_add_visual_features
Revises: 025_add_media_analysis
Create Date: 2024-01-21 10:05:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '026_add_visual_features'
down_revision = '025_add_media_analysis'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        'visual_features',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('media_analysis_id', sa.String(), nullable=False),
        sa.Column('objects_detected', sa.Integer(), nullable=True),
        sa.Column('detected_objects', sa.JSON(), nullable=True),
        sa.Column('text_extracted', sa.JSON(), nullable=True),
        sa.Column('faces_detected', sa.Integer(), nullable=True),
        sa.Column('face_details', sa.JSON(), nullable=True),
        sa.Column('dominant_colors', sa.JSON(), nullable=True),
        sa.Column('aesthetic_score', sa.Float(), nullable=True),
        sa.Column('composition_scores', sa.JSON(), nullable=True),
        sa.Column('brand_elements', sa.JSON(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('video_resolution', sa.JSON(), nullable=True),
        sa.Column('fps', sa.Float(), nullable=True),
        sa.Column('keyframe_count', sa.Integer(), nullable=True),
        sa.Column('keyframes', sa.JSON(), nullable=True),
        sa.Column('scene_count', sa.Integer(), nullable=True),
        sa.Column('scenes', sa.JSON(), nullable=True),
        sa.Column('has_audio', sa.Boolean(), nullable=True),
        sa.Column('audio_features', sa.JSON(), nullable=True),
        sa.Column('embedding_vector', sa.JSON(), nullable=True),
        sa.Column('embedding_dimension', sa.Integer(), nullable=True),
        sa.Column('technical_quality', sa.Float(), nullable=True),
        sa.Column('content_quality', sa.Float(), nullable=True),
        sa.Column('engagement_potential', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(
            ['media_analysis_id'],
            ['media_analyses.id'],
            ondelete='CASCADE'
        )
    )
    
    # Create index
    op.create_index('ix_visual_features_media_analysis_id', 'visual_features', ['media_analysis_id'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_visual_features_media_analysis_id', 'visual_features')
    op.drop_table('visual_features')