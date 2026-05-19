"""Add hashtag performance tables

Revision ID: 020_add_hashtag_performance
Revises: 019_add_competitor_data
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '020_add_hashtag_performance'
down_revision = '019_add_competitor_data'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create hashtag_performance table
    op.create_table(
        'hashtag_performance',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('hashtag', sa.String(length=100), nullable=False),
        sa.Column('platform', sa.String(length=20), nullable=False),
        sa.Column('user_id', sa.String(length=100), nullable=True),
        sa.Column('total_uses', sa.Integer(), nullable=True, default=0),
        sa.Column('unique_posts', sa.Integer(), nullable=True, default=0),
        sa.Column('first_used', sa.DateTime(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('avg_engagement', sa.Float(), nullable=True, default=0.0),
        sa.Column('median_engagement', sa.Float(), nullable=True, default=0.0),
        sa.Column('total_engagement', sa.Integer(), nullable=True, default=0),
        sa.Column('engagement_rate', sa.Float(), nullable=True, default=0.0),
        sa.Column('engagement_lift', sa.Float(), nullable=True, default=0.0),
        sa.Column('reach_amplification', sa.Float(), nullable=True, default=0.0),
        sa.Column('engagement_std', sa.Float(), nullable=True, default=0.0),
        sa.Column('trend_direction', sa.String(length=20), nullable=True, default='stable'),
        sa.Column('momentum', sa.Float(), nullable=True, default=0.0),
        sa.Column('estimated_roi', sa.Float(), nullable=True, default=0.0),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_hashtag_platform', 'hashtag_performance', ['hashtag', 'platform'])
    op.create_index('idx_hashtag_user', 'hashtag_performance', ['hashtag', 'user_id'])
    op.create_index('idx_platform_updated', 'hashtag_performance', ['platform', 'updated_at'])
    op.create_index(op.f('ix_hashtag_performance_hashtag'), 'hashtag_performance', ['hashtag'])
    op.create_index(op.f('ix_hashtag_performance_platform'), 'hashtag_performance', ['platform'])
    op.create_index(op.f('ix_hashtag_performance_user_id'), 'hashtag_performance', ['user_id'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index(op.f('ix_hashtag_performance_user_id'), table_name='hashtag_performance')
    op.drop_index(op.f('ix_hashtag_performance_platform'), table_name='hashtag_performance')
    op.drop_index(op.f('ix_hashtag_performance_hashtag'), table_name='hashtag_performance')
    op.drop_index('idx_platform_updated', table_name='hashtag_performance')
    op.drop_index('idx_hashtag_user', table_name='hashtag_performance')
    op.drop_index('idx_hashtag_platform', table_name='hashtag_performance')
    op.drop_table('hashtag_performance')