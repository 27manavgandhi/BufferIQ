"""
Tests for database models.

Tests model creation, relationships, validators, and constraints.
"""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from bufferiq.domain.models import (
    Channel,
    ContentGap,
    ModelVersion,
    Organization,
    Post,
    Prediction,
    SyncJob,
    User,
    VoiceProfile,
)


class TestUserModel:
    """Test User model."""

    @pytest.mark.asyncio
    async def test_create_user(self, test_session: AsyncSession) -> None:
        """Test creating a user."""
        user = User(
            buffer_org_id="org_test",
            buffer_access_token="token_test",
            email="user@test.com",
        )
        test_session.add(user)
        await test_session.commit()

        assert user.id is not None
        assert user.buffer_org_id == "org_test"
        assert user.email == "user@test.com"
        assert user.is_active is True

    @pytest.mark.asyncio
    async def test_user_unique_email(self, test_session: AsyncSession) -> None:
        """Test user email uniqueness constraint."""
        user1 = User(
            buffer_org_id="org_1",
            buffer_access_token="token_1",
            email="duplicate@test.com",
        )
        test_session.add(user1)
        await test_session.commit()

        user2 = User(
            buffer_org_id="org_2",
            buffer_access_token="token_2",
            email="duplicate@test.com",
        )
        test_session.add(user2)

        with pytest.raises(IntegrityError):
            await test_session.commit()

    @pytest.mark.asyncio
    async def test_user_invalid_email(self, test_session: AsyncSession) -> None:
        """Test user email validation."""
        with pytest.raises(ValueError, match="Invalid email format"):
            User(
                buffer_org_id="org_test",
                buffer_access_token="token_test",
                email="invalid_email",
            )

    @pytest.mark.asyncio
    async def test_user_empty_token(self, test_session: AsyncSession) -> None:
        """Test user access token validation."""
        with pytest.raises(ValueError, match="Access token cannot be empty"):
            User(
                buffer_org_id="org_test", buffer_access_token="", email="test@test.com"
            )

    @pytest.mark.asyncio
    async def test_user_cascade_delete(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test cascade delete removes related organizations."""
        org = Organization(
            user_id=sample_user.id, buffer_org_id="org_cascade", name="Test Org"
        )
        test_session.add(org)
        await test_session.commit()

        await test_session.delete(sample_user)
        await test_session.commit()

        result = await test_session.execute(
            select(Organization).where(Organization.user_id == sample_user.id)
        )
        assert result.scalar_one_or_none() is None


class TestOrganizationModel:
    """Test Organization model."""

    @pytest.mark.asyncio
    async def test_create_organization(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test creating an organization."""
        org = Organization(
            user_id=sample_user.id, buffer_org_id="buffer_org_123", name="My Org"
        )
        test_session.add(org)
        await test_session.commit()

        assert org.id is not None
        assert org.user_id == sample_user.id
        assert org.name == "My Org"

    @pytest.mark.asyncio
    async def test_organization_user_relationship(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test organization-user relationship."""
        org = Organization(
            user_id=sample_user.id, buffer_org_id="org_rel", name="Test Org"
        )
        test_session.add(org)
        await test_session.commit()
        await test_session.refresh(org)

        assert org.user.id == sample_user.id
        assert org in sample_user.organizations


class TestChannelModel:
    """Test Channel model."""

    @pytest.mark.asyncio
    async def test_create_channel(
        self, test_session: AsyncSession, sample_organization: Organization
    ) -> None:
        """Test creating a channel."""
        channel = Channel(
            organization_id=sample_organization.id,
            buffer_channel_id="ch_123",
            platform="twitter",
            handle="@testuser",
        )
        test_session.add(channel)
        await test_session.commit()

        assert channel.id is not None
        assert channel.platform == "twitter"
        assert channel.handle == "@testuser"

    @pytest.mark.asyncio
    async def test_channel_invalid_platform(
        self, test_session: AsyncSession, sample_organization: Organization
    ) -> None:
        """Test channel platform validation."""
        with pytest.raises(ValueError, match="Invalid platform"):
            Channel(
                organization_id=sample_organization.id,
                buffer_channel_id="ch_bad",
                platform="invalid_platform",
                handle="test",
            )

    @pytest.mark.asyncio
    async def test_channel_empty_handle(
        self, test_session: AsyncSession, sample_organization: Organization
    ) -> None:
        """Test channel handle validation."""
        with pytest.raises(ValueError, match="Handle cannot be empty"):
            Channel(
                organization_id=sample_organization.id,
                buffer_channel_id="ch_empty",
                platform="linkedin",
                handle="   ",
            )

    @pytest.mark.asyncio
    async def test_channel_cascade_delete(
        self, test_session: AsyncSession, sample_channel: Channel
    ) -> None:
        """Test cascade delete removes related posts."""
        post = Post(
            channel_id=sample_channel.id,
            buffer_post_id="post_cascade",
            content="Test content",
            content_hash="hash123",
        )
        test_session.add(post)
        await test_session.commit()

        await test_session.delete(sample_channel)
        await test_session.commit()

        result = await test_session.execute(
            select(Post).where(Post.channel_id == sample_channel.id)
        )
        assert result.scalar_one_or_none() is None


class TestPostModel:
    """Test Post model."""

    @pytest.mark.asyncio
    async def test_create_post(
        self, test_session: AsyncSession, sample_channel: Channel
    ) -> None:
        """Test creating a post."""
        post = Post(
            channel_id=sample_channel.id,
            buffer_post_id="post_new",
            content="This is a new post",
            content_hash="newhash",
            status="draft",
        )
        test_session.add(post)
        await test_session.commit()

        assert post.id is not None
        assert post.status == "draft"
        assert post.content == "This is a new post"

    @pytest.mark.asyncio
    async def test_post_invalid_status(
        self, test_session: AsyncSession, sample_channel: Channel
    ) -> None:
        """Test post status validation."""
        with pytest.raises(ValueError, match="Invalid status"):
            Post(
                channel_id=sample_channel.id,
                buffer_post_id="post_invalid",
                content="Test",
                content_hash="hash",
                status="invalid_status",
            )

    @pytest.mark.asyncio
    async def test_post_empty_content(
        self, test_session: AsyncSession, sample_channel: Channel
    ) -> None:
        """Test post content validation."""
        with pytest.raises(ValueError, match="Content cannot be empty"):
            Post(
                channel_id=sample_channel.id,
                buffer_post_id="post_empty",
                content="   ",
                content_hash="hash",
            )

    @pytest.mark.asyncio
    async def test_post_invalid_engagement_rate(
        self, test_session: AsyncSession, sample_channel: Channel
    ) -> None:
        """Test post engagement rate validation."""
        with pytest.raises(ValueError, match="Engagement rate must be between 0 and 1"):
            Post(
                channel_id=sample_channel.id,
                buffer_post_id="post_rate",
                content="Test",
                content_hash="hash",
                engagement_rate=1.5,
            )

    @pytest.mark.asyncio
    async def test_post_total_engagement_property(
        self, test_session: AsyncSession, sample_post: Post
    ) -> None:
        """Test post total engagement computed property."""
        assert sample_post.total_engagement == 65  # 50 + 10 + 5


class TestPredictionModel:
    """Test Prediction model."""

    @pytest.mark.asyncio
    async def test_create_prediction(
        self,
        test_session: AsyncSession,
        sample_post: Post,
        sample_user: User,
    ) -> None:
        """Test creating a prediction."""
        model_version = ModelVersion(
            user_id=sample_user.id,
            version="1.0.0",
            model_type="engagement_predictor",
            algorithm="xgboost",
            training_r2=0.75,
            training_mae=1.5,
            training_rmse=2.0,
            validation_r2=0.70,
            validation_mae=1.8,
            validation_rmse=2.2,
            model_path="/models/v1.pkl",
            feature_names='["feature1", "feature2"]',
            hyperparameters='{"max_depth": 5}',
            training_data_size=1000,
            training_data_date_range="2024-01-01 to 2024-06-01",
        )
        test_session.add(model_version)
        await test_session.commit()
        await test_session.refresh(model_version)

        prediction = Prediction(
            post_id=sample_post.id,
            model_version_id=model_version.id,
            predicted_engagement_score=7.5,
            predicted_likes=60,
            predicted_comments=12,
            predicted_shares=6,
            confidence=0.85,
            features_used='{"text_length": 100}',
        )
        test_session.add(prediction)
        await test_session.commit()

        assert prediction.id is not None
        assert prediction.predicted_engagement_score == 7.5
        assert prediction.confidence == 0.85

    @pytest.mark.asyncio
    async def test_prediction_invalid_confidence(
        self,
        test_session: AsyncSession,
        sample_post: Post,
        sample_user: User,
    ) -> None:
        """Test prediction confidence validation."""
        model_version = ModelVersion(
            user_id=sample_user.id,
            version="1.0.0",
            model_type="engagement_predictor",
            algorithm="xgboost",
            training_r2=0.75,
            training_mae=1.5,
            training_rmse=2.0,
            validation_r2=0.70,
            validation_mae=1.8,
            validation_rmse=2.2,
            model_path="/models/v1.pkl",
            feature_names='["feature1"]',
            hyperparameters='{"max_depth": 5}',
            training_data_size=1000,
            training_data_date_range="2024-01-01 to 2024-06-01",
        )
        test_session.add(model_version)
        await test_session.commit()
        await test_session.refresh(model_version)

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction(
                post_id=sample_post.id,
                model_version_id=model_version.id,
                predicted_engagement_score=7.5,
                confidence=1.5,
                features_used="{}",
            )


class TestModelVersionModel:
    """Test ModelVersion model."""

    @pytest.mark.asyncio
    async def test_create_model_version(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test creating a model version."""
        model_version = ModelVersion(
            user_id=sample_user.id,
            version="2.1.0",
            model_type="timing_optimizer",
            algorithm="lightgbm",
            training_r2=0.80,
            training_mae=1.2,
            training_rmse=1.8,
            validation_r2=0.75,
            validation_mae=1.5,
            validation_rmse=2.0,
            model_path="/models/v2.1.pkl",
            feature_names='["hour", "day_of_week"]',
            hyperparameters='{"num_leaves": 31}',
            training_data_size=5000,
            training_data_date_range="2024-01-01 to 2024-12-01",
        )
        test_session.add(model_version)
        await test_session.commit()

        assert model_version.id is not None
        assert model_version.version == "2.1.0"
        assert model_version.validation_r2 == 0.75

    @pytest.mark.asyncio
    async def test_model_version_invalid_version_format(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test model version format validation."""
        with pytest.raises(ValueError, match="Invalid version format"):
            ModelVersion(
                user_id=sample_user.id,
                version="1.0",  # Invalid format
                model_type="test",
                algorithm="test",
                training_r2=0.7,
                training_mae=1.0,
                training_rmse=1.5,
                validation_r2=0.65,
                validation_mae=1.2,
                validation_rmse=1.7,
                model_path="/models/test.pkl",
                feature_names="[]",
                hyperparameters="{}",
                training_data_size=100,
                training_data_date_range="2024-01-01 to 2024-06-01",
            )


class TestVoiceProfileModel:
    """Test VoiceProfile model."""

    @pytest.mark.asyncio
    async def test_create_voice_profile(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test creating a voice profile."""
        profile = VoiceProfile(
            user_id=sample_user.id,
            avg_post_length=150.5,
            avg_word_count=30.2,
            avg_sentence_length=15.1,
            vocabulary_size=500,
            common_words='["ai", "tech", "innovation"]',
            tone="professional",
            formality_score=0.75,
            emoji_usage_rate=0.10,
            hashtag_usage_rate=0.20,
            question_usage_rate=0.05,
            embedding_centroid="[0.1, 0.2, 0.3]",
            posts_analyzed=100,
        )
        test_session.add(profile)
        await test_session.commit()

        assert profile.id is not None
        assert profile.tone == "professional"
        assert profile.posts_analyzed == 100

    @pytest.mark.asyncio
    async def test_voice_profile_invalid_rate(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test voice profile rate validation."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            VoiceProfile(
                user_id=sample_user.id,
                avg_post_length=150.0,
                avg_word_count=30.0,
                avg_sentence_length=15.0,
                vocabulary_size=500,
                common_words="[]",
                tone="casual",
                formality_score=1.5,  # Invalid
                emoji_usage_rate=0.1,
                hashtag_usage_rate=0.2,
                question_usage_rate=0.05,
                embedding_centroid="[]",
                posts_analyzed=50,
            )


class TestContentGapModel:
    """Test ContentGap model."""

    @pytest.mark.asyncio
    async def test_create_content_gap(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test creating a content gap."""
        gap = ContentGap(
            user_id=sample_user.id,
            topic="Machine Learning",
            topic_keywords='["ml", "ai", "models"]',
            gap_type="underused_high_performer",
            performance_score=0.85,
            frequency_score=0.20,
            opportunity_score=0.90,
            suggested_angles='["tutorial", "case study"]',
        )
        test_session.add(gap)
        await test_session.commit()

        assert gap.id is not None
        assert gap.topic == "Machine Learning"
        assert gap.gap_type == "underused_high_performer"

    @pytest.mark.asyncio
    async def test_content_gap_invalid_type(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test content gap type validation."""
        with pytest.raises(ValueError, match="Invalid gap type"):
            ContentGap(
                user_id=sample_user.id,
                topic="Test",
                topic_keywords="[]",
                gap_type="invalid_type",
                performance_score=0.5,
                frequency_score=0.3,
                opportunity_score=0.7,
                suggested_angles="[]",
            )


class TestSyncJobModel:
    """Test SyncJob model."""

    @pytest.mark.asyncio
    async def test_create_sync_job(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test creating a sync job."""
        job = SyncJob(
            user_id=sample_user.id,
            job_type="initial",
            status="pending",
            total_items=1000,
            processed_items=0,
            failed_items=0,
        )
        test_session.add(job)
        await test_session.commit()

        assert job.id is not None
        assert job.job_type == "initial"
        assert job.status == "pending"

    @pytest.mark.asyncio
    async def test_sync_job_success_rate(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test sync job success rate property."""
        job = SyncJob(
            user_id=sample_user.id,
            job_type="incremental",
            status="completed",
            total_items=100,
            processed_items=95,
            failed_items=5,
        )
        test_session.add(job)
        await test_session.commit()

        assert job.success_rate == 0.9473684210526315  # (95-5)/95

    @pytest.mark.asyncio
    async def test_sync_job_invalid_status(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test sync job status validation."""
        with pytest.raises(ValueError, match="Invalid status"):
            SyncJob(
                user_id=sample_user.id,
                job_type="initial",
                status="invalid_status",
            )

    @pytest.mark.asyncio
    async def test_sync_job_invalid_type(
        self, test_session: AsyncSession, sample_user: User
    ) -> None:
        """Test sync job type validation."""
        with pytest.raises(ValueError, match="Invalid job type"):
            SyncJob(user_id=sample_user.id, job_type="invalid_type", status="pending")
