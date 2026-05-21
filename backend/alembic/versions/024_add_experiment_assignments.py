"""add experiment assignments table

Revision ID: 024
Revises: 023
Create Date: 2024-01-20
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '024'
down_revision = '023'
branch_labels = None
depends_on = None


def upgrade():
    """Create experiment_assignments table."""
    op.create_table(
        'experiment_assignments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('experiment_id', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('variant_id', sa.String(length=100), nullable=False),
        sa.Column('variant_name', sa.String(length=500), nullable=False),
        sa.Column('assigned_at', sa.DateTime(), nullable=False),
        sa.Column('assignment_hash', sa.String(length=255), nullable=False),
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('platform', sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['experiment_id'], ['experiments.experiment_id'], )
    )
    op.create_index(op.f('ix_experiment_assignments_experiment_id'), 'experiment_assignments', ['experiment_id'], unique=False)
    op.create_index(op.f('ix_experiment_assignments_user_id'), 'experiment_assignments', ['user_id'], unique=False)
    op.create_index('idx_experiment_user', 'experiment_assignments', ['experiment_id', 'user_id'], unique=False)


def downgrade():
    """Drop experiment_assignments table."""
    op.drop_index('idx_experiment_user', table_name='experiment_assignments')
    op.drop_index(op.f('ix_experiment_assignments_user_id'), table_name='experiment_assignments')
    op.drop_index(op.f('ix_experiment_assignments_experiment_id'), table_name='experiment_assignments')
    op.drop_table('experiment_assignments')