"""Add content gaps table.

Revision ID: 018
Revises: 017
Create Date: 2024-05-13

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '018'
down_revision = '017'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Create content_gaps table
    op.create_table(
        'content_gaps',
        sa.Column('id', sa.String(50), nullable=False),
        sa.Column('user_id', sa.String(100), nullable=False),
        sa.Column('platform', sa.String(20), nullable=False),
        
        # Gap details
        sa.Column('topic', sa.String(200), nullable=False),
        sa.Column('keywords', sa.JSON(), nullable=False),
        sa.Column('description', sa.String(500), nullable=True),
        
        # Scores
        sa.Column('severity', sa.String(20), nullable=False),
        sa.Column('priority_score', sa.Float(), nullable=False),
        sa.Column('opportunity_score', sa.Float(), nullable=False),
        
        # Context
        sa.Column('competitor_coverage', sa.Integer(), default=0),
        sa.Column('search_volume', sa.Integer(), nullable=True),
        sa.Column('trend_direction', sa.String(20), default='stable'),
        
        # Recommendations
        sa.Column('recommended_content_types', sa.JSON(), nullable=True),
        sa.Column('suggested_angles', sa.JSON(), nullable=True),
        sa.Column('estimated_engagement', sa.Float(), nullable=True),
        
        # Metadata
        sa.Column('detected_at', sa.DateTime(), nullable=False),
        sa.Column('confidence', sa.Float(), default=0.8),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index(
        'ix_content_gaps_user_id',
        'content_gaps',
        ['user_id']
    )
    op.create_index(
        'ix_content_gaps_platform',
        'content_gaps',
        ['platform']
    )
    op.create_index(
        'ix_content_gaps_severity',
        'content_gaps',
        ['severity']
    )
    op.create_index(
        'ix_content_gaps_user_platform',
        'content_gaps',
        ['user_id', 'platform']
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_content_gaps_user_platform', table_name='content_gaps')
    op.drop_index('ix_content_gaps_severity', table_name='content_gaps')
    op.drop_index('ix_content_gaps_platform', table_name='content_gaps')
    op.drop_index('ix_content_gaps_user_id', table_name='content_gaps')
    op.drop_table('content_gaps')