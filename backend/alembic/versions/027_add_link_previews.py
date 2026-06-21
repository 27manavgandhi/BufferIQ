"""Add link previews table.

Revision ID: 027_add_link_previews
Revises: 026_add_visual_features
Create Date: 2024-01-21 10:10:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '027_add_link_previews'
down_revision = '026_add_visual_features'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        'link_previews',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('media_analysis_id', sa.String(), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('image_url', sa.Text(), nullable=True),
        sa.Column('site_name', sa.String(), nullable=True),
        sa.Column('og_tags', sa.JSON(), nullable=True),
        sa.Column('twitter_tags', sa.JSON(), nullable=True),
        sa.Column('title_quality', sa.Float(), nullable=True),
        sa.Column('description_quality', sa.Float(), nullable=True),
        sa.Column('image_quality', sa.Float(), nullable=True),
        sa.Column('overall_quality', sa.Float(), nullable=True),
        sa.Column('predicted_ctr', sa.Float(), nullable=True),
        sa.Column('actual_ctr', sa.Float(), nullable=True),
        sa.Column('ctr_error', sa.Float(), nullable=True),
        sa.Column('optimization_suggestions', sa.JSON(), nullable=True),
        sa.Column('optimized_title', sa.Text(), nullable=True),
        sa.Column('optimized_description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(
            ['media_analysis_id'],
            ['media_analyses.id'],
            ondelete='CASCADE'
        )
    )
    
    # Create indexes
    op.create_index('ix_link_previews_media_analysis_id', 'link_previews', ['media_analysis_id'], unique=True)
    op.create_index('ix_link_previews_url', 'link_previews', ['url'])


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index('ix_link_previews_url', 'link_previews')
    op.drop_index('ix_link_previews_media_analysis_id', 'link_previews')
    op.drop_table('link_previews')