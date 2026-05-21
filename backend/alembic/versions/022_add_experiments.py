"""add experiments table

Revision ID: 022
Revises: 021
Create Date: 2024-01-20
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '022'
down_revision = '021'
branch_labels = None
depends_on = None


def upgrade():
    """Create experiments table."""
    op.create_table(
        'experiments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('primary_metric', sa.String(length=100), nullable=False),
        sa.Column('variants', sa.JSON(), nullable=False),
        sa.Column('alpha', sa.Float(), nullable=False),
        sa.Column('power', sa.Float(), nullable=False),
        sa.Column('mde', sa.Float(), nullable=False),
        sa.Column('required_sample_size', sa.Integer(), nullable=False),
        sa.Column('estimated_duration_days', sa.Integer(), nullable=True),
        sa.Column('start_date', sa.DateTime(), nullable=True),
        sa.Column('end_date', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('enable_sequential_testing', sa.Boolean(), nullable=True),
        sa.Column('enable_early_stopping', sa.Boolean(), nullable=True),
        sa.Column('stratification_key', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiments_experiment_id'), 'experiments', ['experiment_id'], unique=True)


def downgrade():
    """Drop experiments table."""
    op.drop_index(op.f('ix_experiments_experiment_id'), table_name='experiments')
    op.drop_table('experiments')