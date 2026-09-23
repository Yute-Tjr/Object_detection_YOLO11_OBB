from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from terminal_web.database import Base
from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus


def utc_now() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ModelRecord(TimestampMixin, Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        UniqueConstraint("model_type", "name", "version", name="uq_model_identity"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    weights_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    load_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)


class InspectionTask(TimestampMixin, Base):
    __tablename__ = "inspection_tasks"
    __table_args__ = (Index("ix_inspection_tasks_status_created", "status", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    display_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=TaskStatus.queued.value, nullable=False
    )
    current_stage: Mapped[str] = mapped_column(
        String(32), default=ImageStage.pending.value, nullable=False
    )
    total_images: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completed_images: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    succeeded_images: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_images: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(128))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    detector_model_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_registry.id"), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    detector_model: Mapped[ModelRecord | None] = relationship()
    images: Mapped[list[InspectionImage]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="InspectionImage.sequence_no"
    )


class InspectionImage(TimestampMixin, Base):
    __tablename__ = "inspection_images"
    __table_args__ = (
        UniqueConstraint("task_id", "sequence_no", name="uq_task_image_sequence"),
        Index("ix_inspection_images_task_id", "task_id"),
        Index("ix_inspection_images_original_filename", "original_filename"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspection_tasks.id", ondelete="CASCADE"), nullable=False
    )
    sequence_no: Mapped[int] = mapped_column(Integer, nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_path: Mapped[str] = mapped_column(Text, nullable=False)
    result_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        String(32), default=ImageStatus.queued.value, nullable=False
    )
    stage: Mapped[str] = mapped_column(
        String(32), default=ImageStage.pending.value, nullable=False
    )
    overall_result: Mapped[str] = mapped_column(
        String(16), default=OverallResult.unknown.value, nullable=False
    )
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    task: Mapped[InspectionTask] = relationship(back_populates="images")
    detections: Mapped[list[Detection]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[list[ImageFeedback]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )


class Detection(TimestampMixin, Base):
    __tablename__ = "detections"
    __table_args__ = (Index("ix_detections_image_id", "image_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspection_images.id", ondelete="CASCADE"), nullable=False
    )
    region_label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    points: Mapped[list[list[float]]] = mapped_column(JSON, nullable=False)
    selected_for_classification: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    crop_path: Mapped[str | None] = mapped_column(Text)

    image: Mapped[InspectionImage] = relationship(back_populates="detections")
    classifications: Mapped[list[ClassificationResult]] = relationship(
        back_populates="detection", cascade="all, delete-orphan"
    )


class ClassificationResult(TimestampMixin, Base):
    __tablename__ = "classification_results"
    __table_args__ = (Index("ix_classification_results_detection_id", "detection_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    detection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("detections.id", ondelete="CASCADE"), nullable=False
    )
    classifier_type: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict[str, float] | None] = mapped_column(JSON)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("model_registry.id"), nullable=True
    )

    detection: Mapped[Detection] = relationship(back_populates="classifications")
    model: Mapped[ModelRecord | None] = relationship()


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    sessions: Mapped[list[UserSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    feedbacks: Mapped[list[ImageFeedback]] = relationship(
        back_populates="user", passive_deletes=True
    )


class UserSession(TimestampMixin, Base):
    __tablename__ = "user_sessions"
    __table_args__ = (Index("ix_user_sessions_user_id", "user_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    user: Mapped[User] = relationship(back_populates="sessions")


class ImageFeedback(TimestampMixin, Base):
    __tablename__ = "image_feedbacks"
    __table_args__ = (
        UniqueConstraint("user_id", "image_id", name="uq_feedback_user_image"),
        Index("ix_image_feedbacks_user_id", "user_id"),
        Index("ix_image_feedbacks_image_id", "image_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("inspection_images.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    image: Mapped[InspectionImage] = relationship(back_populates="feedbacks")
    user: Mapped[User] = relationship(back_populates="feedbacks")
    items: Mapped[list[ImageFeedbackItem]] = relationship(
        back_populates="feedback", cascade="all, delete-orphan"
    )
    misses: Mapped[list[ImageFeedbackMiss]] = relationship(
        back_populates="feedback", cascade="all, delete-orphan"
    )


class ImageFeedbackItem(TimestampMixin, Base):
    __tablename__ = "image_feedback_items"
    __table_args__ = (
        CheckConstraint(
            "source IN ('manual', 'model', 'unreviewed')",
            name="ck_feedback_item_source",
        ),
        UniqueConstraint(
            "feedback_id", "detection_id", name="uq_feedback_detection"
        ),
        Index("ix_image_feedback_items_feedback_id", "feedback_id"),
        Index("ix_image_feedback_items_detection_id", "detection_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    feedback_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("image_feedbacks.id", ondelete="CASCADE"), nullable=False
    )
    detection_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("detections.id", ondelete="RESTRICT"), nullable=False
    )
    region_label: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="manual", nullable=False)
    verdict: Mapped[str | None] = mapped_column(String(2))
    color: Mapped[str | None] = mapped_column(String(1))

    feedback: Mapped[ImageFeedback] = relationship(back_populates="items")
    detection: Mapped[Detection] = relationship()


class ImageFeedbackMiss(TimestampMixin, Base):
    __tablename__ = "image_feedback_misses"
    __table_args__ = (
        UniqueConstraint(
            "feedback_id", "logical_region", name="uq_feedback_missed_region"
        ),
        Index("ix_image_feedback_misses_feedback_id", "feedback_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    feedback_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("image_feedbacks.id", ondelete="CASCADE"), nullable=False
    )
    logical_region: Mapped[str] = mapped_column(String(16), nullable=False)

    feedback: Mapped[ImageFeedback] = relationship(back_populates="misses")
