"""add experiment results table

Revision ID: 023
Revises: 022
Create Date: 2024-01-20
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '023'
down_revision = '022'
branch_labels = None
depends_on = None


def upgrade():
    """Create experiment_results table."""
    op.create_table(
        'experiment_results',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.String(length=255), nullable=False),
        sa.Column('analyzed_at', sa.DateTime(), nullable=False),
        sa.Column('analysis_type', sa.String(length=50), nullable=False),
        sa.Column('has_winner', sa.Boolean(), nullable=False),
        sa.Column('winner_variant', sa.String(length=100), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('should_launch', sa.Boolean(), nullable=True),
        sa.Column('test_type', sa.String(length=50), nullable=False),
        sa.Column('p_value', sa.Float(), nullable=False),
        sa.Column('is_significant', sa.Boolean(), nullable=False),
        sa.Column('effect_size', sa.Float(), nullable=False),
        sa.Column('effect_size_type', sa.String(length=50), nullable=False),
        sa.Column('absolute_diff', sa.Float(), nullable=False),
        sa.Column('relative_diff', sa.Float(), nullable=False),
        sa.Column('ci_lower', sa.Float(), nullable=False),
        sa.Column('ci_upper', sa.Float(), nullable=False),
        sa.Column('confidence_level', sa.Float(), nullable=False),
        sa.Column('n_control', sa.Integer(), nullable=False),
        sa.Column('n_treatment', sa.Integer(), nullable=False),
        sa.Column('control_mean', sa.Float(), nullable=False),
        sa.Column('treatment_mean', sa.Float(), nullable=False),
        sa.Column('control_metrics', sa.JSON(), nullable=True),
        sa.Column('treatment_metrics', sa.JSON(), nullable=True),
        sa.Column('recommendation', sa.Text(), nullable=True),
        sa.Column('segments', sa.JSON(), nullable=True),
        sa.Column('time_series', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.experiment_id'], )
    )
    op.create_index(op.f('ix_experiment_results_experiment_id'), 'experiment_results', ['experiment_id'], unique=False)


def downgrade():
    """Drop experiment_results table."""
    op.drop_index(op.f('ix_experiment_results_experiment_id'), table_name='experiment_results')
    op.drop_table('experiment_results')