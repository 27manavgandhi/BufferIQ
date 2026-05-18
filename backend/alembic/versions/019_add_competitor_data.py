"""Add competitor analysis table.

Revision ID: 019
Revises: 018
Create Date: 2024-05-13

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '019'
down_revision = '018'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create competitor_analyses table
    op.create_table(
        'competitor_analyses',
        sa.Column('id', sa.String(50), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('platform', sa.String(20), nullable=False),
        
        # Analysis details
        sa.Column('competitor_ids', sa.JSON(), nullable=False),
        sa.Column('analysis_period_days', sa.Integer(), nullable=False),
        
        # Metrics
        sa.Column('user_rank', sa.Integer(), nullable=False),
        sa.Column('share_of_voice', sa.Float(), nullable=False),
        sa.Column('engagement_vs_avg', sa.Float(), nullable=False),
        
        # Insights
        sa.Column('unique_topics', sa.JSON(), nullable=True),
        sa.Column('missed_topics', sa.JSON(), nullable=True),
        sa.Column('common_topics', sa.JSON(), nullable=True),
        sa.Column('competitor_gaps', sa.JSON(), nullable=True),
        sa.Column('differentiation_opportunities', sa.JSON(), nullable=True),
        
        # Competitor profiles
        sa.Column('competitor_profiles', sa.JSON(), nullable=True),
        
        # Metadata
        sa.Column('analyzed_at', sa.DateTime(), nullable=False),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(
        'ix_competitor_analyses_user_id',
        'competitor_analyses',
        ['user_id']
    )
    op.create_index(
        'ix_competitor_analyses_platform',
        'competitor_analyses',
        ['platform']
    )
    op.create_index(
        'ix_competitor_analyses_user_platform',
        'competitor_analyses',
        ['user_id', 'platform']
    )
    op.create_index(
        'ix_competitor_analyses_analyzed_at',
        'competitor_analyses',
        ['analyzed_at']
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_competitor_analyses_analyzed_at', table_name='competitor_analyses')
    op.drop_index('ix_competitor_analyses_user_platform', table_name='competitor_analyses')
    op.drop_index('ix_competitor_analyses_platform', table_name='competitor_analyses')
    op.drop_index('ix_competitor_analyses_user_id', table_name='competitor_analyses')
    op.drop_table('competitor_analyses')