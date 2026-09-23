from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus


def to_camel(value: str) -> str:
    first, *rest = value.split("_")
    return first + "".join(part.capitalize() for part in rest)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


class ModelHealth(ApiModel):
    model_type: str
    name: str
    version: str
    ready: bool
    sha256: str | None = None
    error: str | None = None


class HealthResponse(ApiModel):
    api_ready: bool
    database_ready: bool
    worker_ready: bool
    models_ready: bool


class LoginRequest(ApiModel):
    username: str
    password: str


class CurrentUserResponse(ApiModel):
    id: uuid.UUID
    username: str


class ClassificationResponse(ApiModel):
    classifier_type: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float] | None = None


class DetectionResponse(ApiModel):
    id: uuid.UUID
    region_label: str
    confidence: float
    points: list[list[float]]
    selected_for_classification: bool
    crop_url: str | None = None
    anomaly: ClassificationResponse | None = None
    color: ClassificationResponse | None = None


class ImageSummary(ApiModel):
    id: uuid.UUID
    sequence_no: int
    original_filename: str
    status: ImageStatus
    stage: ImageStage
    overall_result: OverallResult
    width: int
    height: int
    size_bytes: int
    original_url: str
    result_url: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class ImageDetail(ImageSummary):
    detections: list[DetectionResponse]


FeedbackVerdict = Literal["OK", "NG"]
FeedbackColor = Literal["B", "G", "R", "W"]
LogicalRegion = Literal[
    "label1", "label2", "label3", "label4", "label5", "label6"
]


class FeedbackDetectionResponse(ApiModel):
    detection_id: uuid.UUID
    region_label: str
    logical_region: str
    anomaly: ClassificationResponse | None = None
    color: ClassificationResponse | None = None


class FeedbackItemInput(ApiModel):
    detection_id: uuid.UUID
    verdict: FeedbackVerdict
    color: FeedbackColor | None = None


class FeedbackUpdateRequest(ApiModel):
    items: list[FeedbackItemInput]
    missed_regions: list[LogicalRegion]


class FeedbackItemResponse(ApiModel):
    detection_id: uuid.UUID
    region_label: str
    logical_region: str
    source: Literal["manual", "model", "unreviewed"]
    verdict: FeedbackVerdict | None = None
    color: FeedbackColor | None = None


class FeedbackRecordResponse(ApiModel):
    id: uuid.UUID
    items: list[FeedbackItemResponse]
    missed_regions: list[LogicalRegion]
    created_at: datetime
    updated_at: datetime


class ImageFeedbackView(ApiModel):
    image_id: uuid.UUID
    original_filename: str
    status: ImageStatus
    detections: list[FeedbackDetectionResponse]
    missed_region_candidates: list[LogicalRegion]
    feedback: FeedbackRecordResponse | None = None


class TaskSummary(ApiModel):
    id: uuid.UUID
    display_id: str
    status: TaskStatus
    current_stage: ImageStage
    total_images: int
    completed_images: int
    succeeded_images: int
    failed_images: int
    has_feedback: bool
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TaskDetail(TaskSummary):
    images: list[ImageSummary]


class TaskPage(ApiModel):
    items: list[TaskSummary]
    total: int
    offset: int
    limit: int


class ImagePage(ApiModel):
    items: list[ImageSummary]
    total: int
    offset: int
    limit: int


class ErrorResponse(ApiModel):
    detail: str
    context: dict[str, Any] = Field(default_factory=dict)
