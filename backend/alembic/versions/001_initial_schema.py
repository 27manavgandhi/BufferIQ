"""initial schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-04-05 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('buffer_org_id', sa.String(length=255), nullable=False),
        sa.Column('buffer_access_token', sa.Text(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('last_sync_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_buffer_org_id'), 'users', ['buffer_org_id'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # Create organizations table
    op.create_table(
        'organizations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('buffer_org_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_organizations_buffer_org_id'), 'organizations', ['buffer_org_id'], unique=True)
    op.create_index(op.f('ix_organizations_user_id'), 'organizations', ['user_id'], unique=False)
    op.create_index('idx_org_user_buffer', 'organizations', ['user_id', 'buffer_org_id'], unique=False)

    # Create channels table
    op.create_table(
        'channels',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('buffer_channel_id', sa.String(length=255), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('handle', sa.String(length=255), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint("platform IN ('linkedin', 'twitter', 'facebook', 'instagram')", name='check_platform_type'),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_channels_buffer_channel_id'), 'channels', ['buffer_channel_id'], unique=True)
    op.create_index(op.f('ix_channels_organization_id'), 'channels', ['organization_id'], unique=False)
    op.create_index(op.f('ix_channels_platform'), 'channels', ['platform'], unique=False)
    op.create_index('idx_channel_platform_active', 'channels', ['platform', 'is_active'], unique=False)

    # Create posts table
    op.create_table(
        'posts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('buffer_post_id', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('scheduled_at', sa.DateTime(), nullable=True),
        sa.Column('published_at', sa.DateTime(), nullable=True),
        sa.Column('likes', sa.Integer(), nullable=True),
        sa.Column('comments', sa.Integer(), nullable=True),
        sa.Column('shares', sa.Integer(), nullable=True),
        sa.Column('clicks', sa.Integer(), nullable=True),
        sa.Column('impressions', sa.Integer(), nullable=True),
        sa.Column('engagement_rate', sa.Float(), nullable=True),
        sa.Column('last_synced_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint("status IN ('draft', 'scheduled', 'sent', 'failed')", name='check_post_status'),
        sa.CheckConstraint('likes >= 0', name='check_likes_positive'),
        sa.CheckConstraint('comments >= 0', name='check_comments_positive'),
        sa.CheckConstraint('shares >= 0', name='check_shares_positive'),
        sa.CheckConstraint('clicks >= 0', name='check_clicks_positive'),
        sa.CheckConstraint('impressions >= 0', name='check_impressions_positive'),
        sa.CheckConstraint('engagement_rate >= 0 AND engagement_rate <= 1', name='check_engagement_rate_range'),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_posts_buffer_post_id'), 'posts', ['buffer_post_id'], unique=True)
    op.create_index(op.f('ix_posts_channel_id'), 'posts', ['channel_id'], unique=False)
    op.create_index(op.f('ix_posts_content_hash'), 'posts', ['content_hash'], unique=False)
    op.create_index(op.f('ix_posts_published_at'), 'posts', ['published_at'], unique=False)
    op.create_index(op.f('ix_posts_scheduled_at'), 'posts', ['scheduled_at'], unique=False)
    op.create_index(op.f('ix_posts_status'), 'posts', ['status'], unique=False)
    op.create_index('idx_post_channel_status', 'posts', ['channel_id', 'status'], unique=False)
    op.create_index('idx_post_channel_published', 'posts', ['channel_id', 'published_at'], unique=False)
    op.create_index('idx_post_scheduled', 'posts', ['scheduled_at', 'status'], unique=False)

    # Create model_versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('algorithm', sa.String(length=50), nullable=False),
        sa.Column('training_r2', sa.Float(), nullable=False),
        sa.Column('training_mae', sa.Float(), nullable=False),
        sa.Column('training_rmse', sa.Float(), nullable=False),
        sa.Column('validation_r2', sa.Float(), nullable=False),
        sa.Column('validation_mae', sa.Float(), nullable=False),
        sa.Column('validation_rmse', sa.Float(), nullable=False),
        sa.Column('model_path', sa.String(length=500), nullable=False),
        sa.Column('feature_names', sa.Text(), nullable=False),
        sa.Column('hyperparameters', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('training_data_size', sa.Integer(), nullable=False),
        sa.Column('training_data_date_range', sa.String(length=100), nullable=False),
        sa.Column('trained_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('deployed_at', sa.DateTime(), nullable=True),
        sa.Column('retired_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint('training_data_size > 0', name='check_training_size_positive'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_versions_user_id'), 'model_versions', ['user_id'], unique=False)
    op.create_index(op.f('ix_model_versions_version'), 'model_versions', ['version'], unique=False)
    op.create_index('idx_model_user_version', 'model_versions', ['user_id', 'version'], unique=False)
    op.create_index('idx_model_active', 'model_versions', ['is_active', 'user_id'], unique=False)

    # Create predictions table
    op.create_table(
        'predictions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('post_id', sa.Integer(), nullable=False),
        sa.Column('model_version_id', sa.Integer(), nullable=False),
        sa.Column('predicted_engagement_score', sa.Float(), nullable=False),
        sa.Column('predicted_likes', sa.Integer(), nullable=True),
        sa.Column('predicted_comments', sa.Integer(), nullable=True),
        sa.Column('predicted_shares', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('actual_engagement_score', sa.Float(), nullable=True),
        sa.Column('actual_likes', sa.Integer(), nullable=True),
        sa.Column('actual_comments', sa.Integer(), nullable=True),
        sa.Column('actual_shares', sa.Integer(), nullable=True),
        sa.Column('prediction_error', sa.Float(), nullable=True),
        sa.Column('is_accurate', sa.Boolean(), nullable=True),
        sa.Column('features_used', sa.Text(), nullable=False),
        sa.Column('prediction_made_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('actual_recorded_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
        sa.CheckConstraint('predicted_likes >= 0', name='check_predicted_likes_positive'),
        sa.CheckConstraint('predicted_comments >= 0', name='check_predicted_comments_positive'),
        sa.CheckConstraint('predicted_shares >= 0', name='check_predicted_shares_positive'),
        sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], ),
        sa.ForeignKeyConstraint(['post_id'], ['posts.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_predictions_model_version_id'), 'predictions', ['model_version_id'], unique=False)
    op.create_index(op.f('ix_predictions_post_id'), 'predictions', ['post_id'], unique=False)
    op.create_index(op.f('ix_predictions_prediction_made_at'), 'predictions', ['prediction_made_at'], unique=False)
    op.create_index('idx_prediction_post_model', 'predictions', ['post_id', 'model_version_id'], unique=False)

    # Create voice_profiles table
    op.create_table(
        'voice_profiles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('avg_post_length', sa.Float(), nullable=False),
        sa.Column('avg_word_count', sa.Float(), nullable=False),
        sa.Column('avg_sentence_length', sa.Float(), nullable=False),
        sa.Column('vocabulary_size', sa.Integer(), nullable=False),
        sa.Column('common_words', sa.Text(), nullable=False),
        sa.Column('tone', sa.String(length=50), nullable=False),
        sa.Column('formality_score', sa.Float(), nullable=False),
        sa.Column('emoji_usage_rate', sa.Float(), nullable=False),
        sa.Column('hashtag_usage_rate', sa.Float(), nullable=False),
        sa.Column('question_usage_rate', sa.Float(), nullable=False),
        sa.Column('embedding_centroid', sa.Text(), nullable=False),
        sa.Column('posts_analyzed', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint('avg_post_length >= 0', name='check_avg_length_positive'),
        sa.CheckConstraint('avg_word_count >= 0', name='check_word_count_positive'),
        sa.CheckConstraint('vocabulary_size >= 0', name='check_vocab_size_positive'),
        sa.CheckConstraint('formality_score >= 0 AND formality_score <= 1', name='check_formality_range'),
        sa.CheckConstraint('emoji_usage_rate >= 0 AND emoji_usage_rate <= 1', name='check_emoji_rate_range'),
        sa.CheckConstraint('hashtag_usage_rate >= 0 AND hashtag_usage_rate <= 1', name='check_hashtag_rate_range'),
        sa.CheckConstraint('question_usage_rate >= 0 AND question_usage_rate <= 1', name='check_question_rate_range'),
        sa.CheckConstraint('posts_analyzed > 0', name='check_posts_analyzed_positive'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id')
    )
    op.create_index(op.f('ix_voice_profiles_user_id'), 'voice_profiles', ['user_id'], unique=True)

    # Create content_gaps table
    op.create_table(
        'content_gaps',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('topic', sa.String(length=255), nullable=False),
        sa.Column('topic_keywords', sa.Text(), nullable=False),
        sa.Column('gap_type', sa.String(length=50), nullable=False),
        sa.Column('performance_score', sa.Float(), nullable=False),
        sa.Column('frequency_score', sa.Float(), nullable=False),
        sa.Column('opportunity_score', sa.Float(), nullable=False),
        sa.Column('suggested_angles', sa.Text(), nullable=False),
        sa.Column('identified_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('is_addressed', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint("gap_type IN ('underused_high_performer', 'declining', 'emerging')", name='check_gap_type'),
        sa.CheckConstraint('performance_score >= 0 AND performance_score <= 1', name='check_performance_score_range'),
        sa.CheckConstraint('frequency_score >= 0 AND frequency_score <= 1', name='check_frequency_score_range'),
        sa.CheckConstraint('opportunity_score >= 0 AND opportunity_score <= 1', name='check_opportunity_score_range'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_content_gaps_topic'), 'content_gaps', ['topic'], unique=False)
    op.create_index(op.f('ix_content_gaps_user_id'), 'content_gaps', ['user_id'], unique=False)
    op.create_index('idx_gap_user_score', 'content_gaps', ['user_id', 'opportunity_score'], unique=False)

    # Create sync_jobs table
    op.create_table(
        'sync_jobs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('job_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('total_items', sa.Integer(), nullable=True),
        sa.Column('processed_items', sa.Integer(), nullable=False),
        sa.Column('failed_items', sa.Integer(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=False),
        sa.CheckConstraint("job_type IN ('initial', 'incremental')", name='check_job_type'),
        sa.CheckConstraint("status IN ('pending', 'running', 'completed', 'failed')", name='check_sync_status'),
        sa.CheckConstraint('processed_items >= 0', name='check_processed_positive'),
        sa.CheckConstraint('failed_items >= 0', name='check_failed_positive'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_sync_jobs_status'), 'sync_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_sync_jobs_user_id'), 'sync_jobs', ['user_id'], unique=False)
    op.create_index('idx_sync_user_status', 'sync_jobs', ['user_id', 'status'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_sync_user_status', table_name='sync_jobs')
    op.drop_index(op.f('ix_sync_jobs_user_id'), table_name='sync_jobs')
    op.drop_index(op.f('ix_sync_jobs_status'), table_name='sync_jobs')
    op.drop_table('sync_jobs')
    
    op.drop_index('idx_gap_user_score', table_name='content_gaps')
    op.drop_index(op.f('ix_content_gaps_user_id'), table_name='content_gaps')
    op.drop_index(op.f('ix_content_gaps_topic'), table_name='content_gaps')
    op.drop_table('content_gaps')
    
    op.drop_index(op.f('ix_voice_profiles_user_id'), table_name='voice_profiles')
    op.drop_table('voice_profiles')
    
    op.drop_index('idx_prediction_post_model', table_name='predictions')
    op.drop_index(op.f('ix_predictions_prediction_made_at'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_post_id'), table_name='predictions')
    op.drop_index(op.f('ix_predictions_model_version_id'), table_name='predictions')
    op.drop_table('predictions')
    
    op.drop_index('idx_model_active', table_name='model_versions')
    op.drop_index('idx_model_user_version', table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_version'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_user_id'), table_name='model_versions')
    op.drop_table('model_versions')
    
    op.drop_index('idx_post_scheduled', table_name='posts')
    op.drop_index('idx_post_channel_published', table_name='posts')
    op.drop_index('idx_post_channel_status', table_name='posts')
    op.drop_index(op.f('ix_posts_status'), table_name='posts')
    op.drop_index(op.f('ix_posts_scheduled_at'), table_name='posts')
    op.drop_index(op.f('ix_posts_published_at'), table_name='posts')
    op.drop_index(op.f('ix_posts_content_hash'), table_name='posts')
    op.drop_index(op.f('ix_posts_channel_id'), table_name='posts')
    op.drop_index(op.f('ix_posts_buffer_post_id'), table_name='posts')
    op.drop_table('posts')
    
    op.drop_index('idx_channel_platform_active', table_name='channels')
    op.drop_index(op.f('ix_channels_platform'), table_name='channels')
    op.drop_index(op.f('ix_channels_organization_id'), table_name='channels')
    op.drop_index(op.f('ix_channels_buffer_channel_id'), table_name='channels')
    op.drop_table('channels')
    
    op.drop_index('idx_org_user_buffer', table_name='organizations')
    op.drop_index(op.f('ix_organizations_user_id'), table_name='organizations')
    op.drop_index(op.f('ix_organizations_buffer_org_id'), table_name='organizations')
    op.drop_table('organizations')
    
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_index(op.f('ix_users_buffer_org_id'), table_name='users')
    op.drop_table('users')