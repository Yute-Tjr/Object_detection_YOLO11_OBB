from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

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
    models: list[ModelHealth]


class ClassificationResponse(ApiModel):
    classifier_type: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float] | None = None
    model_name: str | None = None
    model_version: str | None = None


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


class TaskSummary(ApiModel):
    id: uuid.UUID
    display_id: str
    name: str | None = None
    note: str | None = None
    status: TaskStatus
    current_stage: ImageStage
    total_images: int
    completed_images: int
    succeeded_images: int
    failed_images: int
    detector_model: str
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
