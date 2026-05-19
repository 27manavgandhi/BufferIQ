"""Add hashtag trends tables

Revision ID: 021_add_hashtag_trends
Revises: 020_add_hashtag_performance
Create Date: 2024-01-15 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '021_add_hashtag_trends'
down_revision = '020_add_hashtag_performance'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create hashtag_trends table
    op.create_table(
        'hashtag_trends',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('hashtag', sa.String(length=100), nullable=False),
        sa.Column('platform', sa.String(length=20), nullable=False),
        sa.Column('stage', sa.String(length=20), nullable=False),
        sa.Column('momentum_score', sa.Float(), nullable=True, default=0.0),
        sa.Column('velocity', sa.Float(), nullable=True, default=0.0),
        sa.Column('current_volume', sa.Integer(), nullable=True, default=0),
        sa.Column('volume_change', sa.Float(), nullable=True, default=0.0),
        sa.Column('peak_volume', sa.Integer(), nullable=True, default=0),
        sa.Column('trending_since', sa.DateTime(), nullable=True),
        sa.Column('time_to_peak', sa.Integer(), nullable=True),
        sa.Column('related_topics', sa.JSON(), nullable=True),
        sa.Column('top_influencers', sa.JSON(), nullable=True),
        sa.Column('geographic_hotspots', sa.JSON(), nullable=True),
        sa.Column('opportunity_score', sa.Float(), nullable=True, default=0.0),
        sa.Column('competition_level', sa.String(length=20), nullable=True, default='medium'),
        sa.Column('recommendation', sa.String(length=20), nullable=True, default='monitor'),
        sa.Column('detected_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_trend_hashtag_platform', 'hashtag_trends', ['hashtag', 'platform'])
    op.create_index('idx_trend_stage', 'hashtag_trends', ['stage'])
    op.create_index('idx_trend_momentum', 'hashtag_trends', ['momentum_score'])
    op.create_index('idx_trend_detected', 'hashtag_trends', ['detected_at'])
    op.create_index(op.f('ix_hashtag_trends_hashtag'), 'hashtag_trends', ['hashtag'])
    op.create_index(op.f('ix_hashtag_trends_platform'), 'hashtag_trends', ['platform'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index(op.f('ix_hashtag_trends_platform'), table_name='hashtag_trends')
    op.drop_index(op.f('ix_hashtag_trends_hashtag'), table_name='hashtag_trends')
    op.drop_index('idx_trend_detected', table_name='hashtag_trends')
    op.drop_index('idx_trend_momentum', table_name='hashtag_trends')
    op.drop_index('idx_trend_stage', table_name='hashtag_trends')
    op.drop_index('idx_trend_hashtag_platform', table_name='hashtag_trends')
    op.drop_table('hashtag_trends')