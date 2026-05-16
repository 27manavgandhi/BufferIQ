"""Add voice profile tables

Revision ID: 017_add_voice_profiles
Revises: 016_add_diversity_metrics
Create Date: 2024-05-08 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '017_add_voice_profiles'
down_revision = '016_add_diversity_metrics'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Upgrade database schema."""
    
    # Create voice_profiles table
    op.create_table(
        'voice_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('profile_id', sa.String(length=255), nullable=False),
        sa.Column('brand_id', sa.String(length=255), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('lexical_fingerprint', sa.JSON(), nullable=False),
        sa.Column('syntactic_fingerprint', sa.JSON(), nullable=False),
        sa.Column('stylistic_fingerprint', sa.JSON(), nullable=False),
        sa.Column('signature', sa.String(length=64), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('platform_profiles', sa.JSON(), nullable=True),
        sa.Column('previous_version_id', sa.String(length=255), nullable=True),
        sa.Column('drift_from_previous', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for voice_profiles
    op.create_index('ix_voice_profiles_profile_id', 'voice_profiles', ['profile_id'], unique=True)
    op.create_index('ix_voice_profiles_brand_id', 'voice_profiles', ['brand_id'])
    op.create_index('ix_voice_profiles_signature', 'voice_profiles', ['signature'])
    op.create_index('ix_voice_profiles_brand_platform', 'voice_profiles', ['brand_id', 'platform'])
    
    # Create voice_analysis_logs table
    op.create_table(
        'voice_analysis_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('brand_id', sa.String(length=255), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('profile_id', sa.String(length=255), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('content_length', sa.Integer(), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('lexical_score', sa.Float(), nullable=False),
        sa.Column('syntactic_score', sa.Float(), nullable=False),
        sa.Column('stylistic_score', sa.Float(), nullable=False),
        sa.Column('cosine_similarity', sa.Float(), nullable=True),
        sa.Column('kl_divergence', sa.Float(), nullable=True),
        sa.Column('is_consistent', sa.Integer(), nullable=False),
        sa.Column('severity', sa.String(length=50), nullable=False),
        sa.Column('analyzed_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for voice_analysis_logs
    op.create_index('ix_voice_analysis_logs_brand_id', 'voice_analysis_logs', ['brand_id'])
    op.create_index('ix_voice_analysis_logs_content_hash', 'voice_analysis_logs', ['content_hash'])
    op.create_index('ix_voice_analysis_logs_analyzed_at', 'voice_analysis_logs', ['analyzed_at'])
    
    # Create voice_drift_logs table
    op.create_table(
        'voice_drift_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('brand_id', sa.String(length=255), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('drift_detected', sa.Integer(), nullable=False),
        sa.Column('drift_score', sa.Float(), nullable=False),
        sa.Column('drift_type', sa.String(length=50), nullable=False),
        sa.Column('severity', sa.String(length=50), nullable=False),
        sa.Column('affected_dimensions', sa.JSON(), nullable=True),
        sa.Column('t_statistic', sa.Float(), nullable=True),
        sa.Column('p_value', sa.Float(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('checked_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for voice_drift_logs
    op.create_index('ix_voice_drift_logs_brand_id', 'voice_drift_logs', ['brand_id'])
    op.create_index('ix_voice_drift_logs_checked_at', 'voice_drift_logs', ['checked_at'])


def downgrade() -> None:
    """Downgrade database schema."""
    
    # Drop indices
    op.drop_index('ix_voice_drift_logs_checked_at', table_name='voice_drift_logs')
    op.drop_index('ix_voice_drift_logs_brand_id', table_name='voice_drift_logs')
    op.drop_index('ix_voice_analysis_logs_analyzed_at', table_name='voice_analysis_logs')
    op.drop_index('ix_voice_analysis_logs_content_hash', table_name='voice_analysis_logs')
    op.drop_index('ix_voice_analysis_logs_brand_id', table_name='voice_analysis_logs')
    op.drop_index('ix_voice_profiles_brand_platform', table_name='voice_profiles')
    op.drop_index('ix_voice_profiles_signature', table_name='voice_profiles')
    op.drop_index('ix_voice_profiles_brand_id', table_name='voice_profiles')
    op.drop_index('ix_voice_profiles_profile_id', table_name='voice_profiles')
    
    # Drop tables
    op.drop_table('voice_drift_logs')
    op.drop_table('voice_analysis_logs')
    op.drop_table('voice_profiles')