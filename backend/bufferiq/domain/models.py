"""
SQLAlchemy ORM models for BufferIQ.

Models:
- User: BufferIQ user with Buffer OAuth credentials
- Organization: Buffer organization
- Channel: Social media channel (LinkedIn, Twitter, etc.)
- Post: Social media post with engagement metrics
- Prediction: ML predictions for posts
- ModelVersion: ML model versioning and performance tracking
- VoiceProfile: User's writing style profile
- ContentGap: Identified content opportunities
- SyncJob: Data sync job tracking
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from bufferiq.domain.base import Base, TimestampMixin


class EnvironmentType(str, Enum):
    """Post status enumeration."""

    DRAFT = "draft"
    SCHEDULED = "scheduled"
    SENT = "sent"
    FAILED = "failed"


class PlatformType(str, Enum):
    """Social media platform types."""

    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    INSTAGRAM = "instagram"


class SyncJobStatus(str, Enum):
    """Sync job status enumeration."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentGapType(str, Enum):
    """Content gap type enumeration."""

    UNDERUSED_HIGH_PERFORMER = "underused_high_performer"
    DECLINING = "declining"
    EMERGING = "emerging"


class User(Base, TimestampMixin):
    """
    BufferIQ user.

    Represents a user who has connected their Buffer account.
    """

    __tablename__ = "users"

    buffer_org_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    buffer_access_token: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, index=True, nullable=True
    )
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    organizations: Mapped[list["Organization"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    model_versions: Mapped[list["ModelVersion"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    voice_profile: Mapped[Optional["VoiceProfile"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )
    content_gaps: Mapped[list["ContentGap"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    sync_jobs: Mapped[list["SyncJob"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    @validates("email")
    def validate_email(self, key: str, value: Optional[str]) -> Optional[str]:
        """Validate email format."""
        if value and "@" not in value:
            raise ValueError(f"Invalid email format: {value}")
        return value

    @validates("buffer_access_token")
    def validate_token(self, key: str, value: str) -> str:
        """Validate access token is not empty."""
        if not value or not value.strip():
            raise ValueError("Access token cannot be empty")
        return value

    def __repr__(self) -> str:
        return f"<User(id={self.id}, buffer_org_id='{self.buffer_org_id}', email='{self.email}')>"


class Organization(Base, TimestampMixin):
    """
    Buffer organization.

    Represents a Buffer organization that contains channels.
    """

    __tablename__ = "organizations"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    buffer_org_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)

    user: Mapped["User"] = relationship(back_populates="organizations")
    channels: Mapped[list["Channel"]] = relationship(
        back_populates="organization", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("idx_org_user_buffer", "user_id", "buffer_org_id"),)

    def __repr__(self) -> str:
        return f"<Organization(id={self.id}, name='{self.name}')>"


class Channel(Base, TimestampMixin):
    """
    Social media channel.

    Represents a social media account connected to Buffer.
    """

    __tablename__ = "channels"

    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True, nullable=False
    )
    buffer_channel_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    platform: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    handle: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)

    organization: Mapped["Organization"] = relationship(back_populates="channels")
    posts: Mapped[list["Post"]] = relationship(
        back_populates="channel", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_channel_platform_active", "platform", "is_active"),
        CheckConstraint(
            "platform IN ('linkedin', 'twitter', 'facebook', 'instagram')",
            name="check_platform_type",
        ),
    )

    @validates("platform")
    def validate_platform(self, key: str, value: str) -> str:
        """Validate platform is supported."""
        valid_platforms = {"linkedin", "twitter", "facebook", "instagram"}
        if value.lower() not in valid_platforms:
            raise ValueError(
                f"Invalid platform: {value}. Must be one of {valid_platforms}"
            )
        return value.lower()

    @validates("handle")
    def validate_handle(self, key: str, value: str) -> str:
        """Validate handle is not empty."""
        if not value or not value.strip():
            raise ValueError("Handle cannot be empty")
        return value.strip()

    def __repr__(self) -> str:
        return f"<Channel(id={self.id}, platform='{self.platform}', handle='{self.handle}')>"


class Post(Base, TimestampMixin):
    """
    Social media post.

    Represents a post scheduled or published through Buffer.
    """

    __tablename__ = "posts"

    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channels.id", ondelete="CASCADE"), index=True, nullable=False
    )
    buffer_post_id: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), index=True, default="draft", nullable=False
    )
    scheduled_at: Mapped[Optional[datetime]] = mapped_column(index=True, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(index=True, nullable=True)
    likes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comments: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shares: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    clicks: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    impressions: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    engagement_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    channel: Mapped["Channel"] = relationship(back_populates="posts")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="post", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_post_channel_status", "channel_id", "status"),
        Index("idx_post_channel_published", "channel_id", "published_at"),
        Index("idx_post_scheduled", "scheduled_at", "status"),
        CheckConstraint(
            "status IN ('draft', 'scheduled', 'sent', 'failed')",
            name="check_post_status",
        ),
        CheckConstraint("likes >= 0", name="check_likes_positive"),
        CheckConstraint("comments >= 0", name="check_comments_positive"),
        CheckConstraint("shares >= 0", name="check_shares_positive"),
        CheckConstraint("clicks >= 0", name="check_clicks_positive"),
        CheckConstraint("impressions >= 0", name="check_impressions_positive"),
        CheckConstraint(
            "engagement_rate >= 0 AND engagement_rate <= 1",
            name="check_engagement_rate_range",
        ),
    )

    @validates("status")
    def validate_status(self, key: str, value: str) -> str:
        """Validate post status."""
        valid_statuses = {"draft", "scheduled", "sent", "failed"}
        if value.lower() not in valid_statuses:
            raise ValueError(
                f"Invalid status: {value}. Must be one of {valid_statuses}"
            )
        return value.lower()

    @validates("content")
    def validate_content(self, key: str, value: str) -> str:
        """Validate content is not empty."""
        if not value or not value.strip():
            raise ValueError("Content cannot be empty")
        return value.strip()

    @validates("engagement_rate")
    def validate_engagement_rate(
        self, key: str, value: Optional[float]
    ) -> Optional[float]:
        """Validate engagement rate is between 0 and 1."""
        if value is not None and (value < 0 or value > 1):
            raise ValueError(f"Engagement rate must be between 0 and 1, got {value}")
        return value

    @property
    def total_engagement(self) -> int:
        """Calculate total engagement (likes + comments + shares)."""
        return (self.likes or 0) + (self.comments or 0) + (self.shares or 0)

    def __repr__(self) -> str:
        return f"<Post(id={self.id}, buffer_post_id='{self.buffer_post_id}', status='{self.status}')>"


class Prediction(Base, TimestampMixin):
    """
    ML prediction for a post.

    Stores predicted and actual engagement metrics.
    """

    __tablename__ = "predictions"

    post_id: Mapped[int] = mapped_column(
        ForeignKey("posts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True, nullable=False
    )
    predicted_engagement_score: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_likes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_comments: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_shares: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    actual_engagement_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    actual_likes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    actual_comments: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    actual_shares: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prediction_error: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_accurate: Mapped[Optional[bool]] = mapped_column(nullable=True)
    features_used: Mapped[str] = mapped_column(Text, nullable=False)
    prediction_made_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), index=True, nullable=False
    )
    actual_recorded_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    post: Mapped["Post"] = relationship(back_populates="predictions")
    model_version: Mapped["ModelVersion"] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("idx_prediction_post_model", "post_id", "model_version_id"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="check_confidence_range"
        ),
        CheckConstraint("predicted_likes >= 0", name="check_predicted_likes_positive"),
        CheckConstraint(
            "predicted_comments >= 0", name="check_predicted_comments_positive"
        ),
        CheckConstraint(
            "predicted_shares >= 0", name="check_predicted_shares_positive"
        ),
    )

    @validates("confidence")
    def validate_confidence(self, key: str, value: float) -> float:
        """Validate confidence is between 0 and 1."""
        if value < 0 or value > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {value}")
        return value

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, post_id={self.post_id}, score={self.predicted_engagement_score:.2f})>"


class ModelVersion(Base, TimestampMixin):
    """
    ML model version.

    Tracks different versions of ML models with performance metrics.
    """

    __tablename__ = "model_versions"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    version: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    training_r2: Mapped[float] = mapped_column(Float, nullable=False)
    training_mae: Mapped[float] = mapped_column(Float, nullable=False)
    training_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    validation_r2: Mapped[float] = mapped_column(Float, nullable=False)
    validation_mae: Mapped[float] = mapped_column(Float, nullable=False)
    validation_rmse: Mapped[float] = mapped_column(Float, nullable=False)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    feature_names: Mapped[str] = mapped_column(Text, nullable=False)
    hyperparameters: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(default=False, index=True, nullable=False)
    training_data_size: Mapped[int] = mapped_column(Integer, nullable=False)
    training_data_date_range: Mapped[str] = mapped_column(String(100), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    deployed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    retired_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    user: Mapped["User"] = relationship(back_populates="model_versions")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="model_version"
    )

    __table_args__ = (
        Index("idx_model_user_version", "user_id", "version"),
        Index("idx_model_active", "is_active", "user_id"),
        CheckConstraint("training_data_size > 0", name="check_training_size_positive"),
    )

    @validates("version")
    def validate_version(self, key: str, value: str) -> str:
        """Validate version format (e.g., 1.0.0)."""
        parts = value.split(".")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError(f"Invalid version format: {value}. Expected format: X.Y.Z")
        return value

    def __repr__(self) -> str:
        return f"<ModelVersion(id={self.id}, version='{self.version}', r2={self.validation_r2:.3f})>"


class VoiceProfile(Base, TimestampMixin):
    """
    User's writing style profile.

    Captures linguistic and stylistic features of user's content.
    """

    __tablename__ = "voice_profiles"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        index=True,
        nullable=False,
    )
    avg_post_length: Mapped[float] = mapped_column(Float, nullable=False)
    avg_word_count: Mapped[float] = mapped_column(Float, nullable=False)
    avg_sentence_length: Mapped[float] = mapped_column(Float, nullable=False)
    vocabulary_size: Mapped[int] = mapped_column(Integer, nullable=False)
    common_words: Mapped[str] = mapped_column(Text, nullable=False)
    tone: Mapped[str] = mapped_column(String(50), nullable=False)
    formality_score: Mapped[float] = mapped_column(Float, nullable=False)
    emoji_usage_rate: Mapped[float] = mapped_column(Float, nullable=False)
    hashtag_usage_rate: Mapped[float] = mapped_column(Float, nullable=False)
    question_usage_rate: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_centroid: Mapped[str] = mapped_column(Text, nullable=False)
    posts_analyzed: Mapped[int] = mapped_column(Integer, nullable=False)

    user: Mapped["User"] = relationship(back_populates="voice_profile")

    __table_args__ = (
        CheckConstraint("avg_post_length >= 0", name="check_avg_length_positive"),
        CheckConstraint("avg_word_count >= 0", name="check_word_count_positive"),
        CheckConstraint("vocabulary_size >= 0", name="check_vocab_size_positive"),
        CheckConstraint(
            "formality_score >= 0 AND formality_score <= 1",
            name="check_formality_range",
        ),
        CheckConstraint(
            "emoji_usage_rate >= 0 AND emoji_usage_rate <= 1",
            name="check_emoji_rate_range",
        ),
        CheckConstraint(
            "hashtag_usage_rate >= 0 AND hashtag_usage_rate <= 1",
            name="check_hashtag_rate_range",
        ),
        CheckConstraint(
            "question_usage_rate >= 0 AND question_usage_rate <= 1",
            name="check_question_rate_range",
        ),
        CheckConstraint("posts_analyzed > 0", name="check_posts_analyzed_positive"),
    )

    @validates(
        "formality_score",
        "emoji_usage_rate",
        "hashtag_usage_rate",
        "question_usage_rate",
    )
    def validate_rate(self, key: str, value: float) -> float:
        """Validate rate is between 0 and 1."""
        if value < 0 or value > 1:
            raise ValueError(f"{key} must be between 0 and 1, got {value}")
        return value

    def __repr__(self) -> str:
        return (
            f"<VoiceProfile(id={self.id}, user_id={self.user_id}, tone='{self.tone}')>"
        )


class ContentGap(Base, TimestampMixin):
    """
    Identified content opportunity.

    Represents topics with high performance potential that are underutilized.
    """

    __tablename__ = "content_gaps"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    topic: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    topic_keywords: Mapped[str] = mapped_column(Text, nullable=False)
    gap_type: Mapped[str] = mapped_column(String(50), nullable=False)
    performance_score: Mapped[float] = mapped_column(Float, nullable=False)
    frequency_score: Mapped[float] = mapped_column(Float, nullable=False)
    opportunity_score: Mapped[float] = mapped_column(Float, nullable=False)
    suggested_angles: Mapped[str] = mapped_column(Text, nullable=False)
    identified_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    is_addressed: Mapped[bool] = mapped_column(default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="content_gaps")

    __table_args__ = (
        Index("idx_gap_user_score", "user_id", "opportunity_score"),
        CheckConstraint(
            "gap_type IN ('underused_high_performer', 'declining', 'emerging')",
            name="check_gap_type",
        ),
        CheckConstraint(
            "performance_score >= 0 AND performance_score <= 1",
            name="check_performance_score_range",
        ),
        CheckConstraint(
            "frequency_score >= 0 AND frequency_score <= 1",
            name="check_frequency_score_range",
        ),
        CheckConstraint(
            "opportunity_score >= 0 AND opportunity_score <= 1",
            name="check_opportunity_score_range",
        ),
    )

    @validates("gap_type")
    def validate_gap_type(self, key: str, value: str) -> str:
        """Validate gap type."""
        valid_types = {"underused_high_performer", "declining", "emerging"}
        if value not in valid_types:
            raise ValueError(f"Invalid gap type: {value}. Must be one of {valid_types}")
        return value

    def __repr__(self) -> str:
        return (
            f"<ContentGap(id={self.id}, topic='{self.topic}', type='{self.gap_type}')>"
        )


class SyncJob(Base, TimestampMixin):
    """
    Data synchronization job.

    Tracks sync operations with Buffer API.
    """

    __tablename__ = "sync_jobs"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50), index=True, default="pending", nullable=False
    )
    total_items: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)

    user: Mapped["User"] = relationship(back_populates="sync_jobs")

    __table_args__ = (
        Index("idx_sync_user_status", "user_id", "status"),
        CheckConstraint(
            "job_type IN ('initial', 'incremental')", name="check_job_type"
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="check_sync_status",
        ),
        CheckConstraint("processed_items >= 0", name="check_processed_positive"),
        CheckConstraint("failed_items >= 0", name="check_failed_positive"),
    )

    @validates("status")
    def validate_status(self, key: str, value: str) -> str:
        """Validate sync status."""
        valid_statuses = {"pending", "running", "completed", "failed"}
        if value.lower() not in valid_statuses:
            raise ValueError(
                f"Invalid status: {value}. Must be one of {valid_statuses}"
            )
        return value.lower()

    @validates("job_type")
    def validate_job_type(self, key: str, value: str) -> str:
        """Validate job type."""
        valid_types = {"initial", "incremental"}
        if value.lower() not in valid_types:
            raise ValueError(f"Invalid job type: {value}. Must be one of {valid_types}")
        return value.lower()

    @property
    def success_rate(self) -> float:
        """Calculate success rate of processed items."""
        if self.processed_items == 0:
            return 0.0
        successful = self.processed_items - self.failed_items
        return successful / self.processed_items

    def __repr__(self) -> str:
        return (
            f"<SyncJob(id={self.id}, type='{self.job_type}', status='{self.status}')>"
        )
