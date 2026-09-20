from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from terminal_web.domain import (
    ImageOutcome,
    ImageStage,
    ImageStatus,
    OverallResult,
    TaskStatus,
    aggregate_task_status,
)
from terminal_web.models import Detection, InspectionImage, InspectionTask


class TaskRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_task(
        self,
        task_id: uuid.UUID,
        display_id: str,
        images: Sequence[InspectionImage],
        *,
        name: str | None = None,
        note: str | None = None,
    ) -> InspectionTask:
        task = InspectionTask(
            id=task_id,
            display_id=display_id,
            name=name,
            note=note,
            status=TaskStatus.queued,
            current_stage=ImageStage.pending,
            total_images=len(images),
            images=list(images),
        )
        self.session.add(task)
        self.session.flush()
        return task

    def get_task(self, task_id: uuid.UUID) -> InspectionTask | None:
        statement = (
            select(InspectionTask)
            .where(InspectionTask.id == task_id)
            .options(
                selectinload(InspectionTask.detector_model),
                selectinload(InspectionTask.images),
            )
        )
        return self.session.scalar(statement)

    def list_tasks(
        self,
        statuses: Sequence[TaskStatus | str] | None,
        query: str | None,
        offset: int,
        limit: int,
    ) -> tuple[list[InspectionTask], int]:
        filters = []
        status_values = self._expand_statuses(statuses)
        if status_values:
            filters.append(InspectionTask.status.in_(status_values))
        if query:
            pattern = f"%{query.strip()}%"
            image_match = (
                select(InspectionImage.id)
                .where(
                    InspectionImage.task_id == InspectionTask.id,
                    InspectionImage.original_filename.ilike(pattern),
                )
                .exists()
            )
            filters.append(
                or_(
                    InspectionTask.display_id.ilike(pattern),
                    InspectionTask.name.ilike(pattern),
                    image_match,
                )
            )

        total = self.session.scalar(
            select(func.count()).select_from(InspectionTask).where(*filters)
        ) or 0
        statement = (
            select(InspectionTask)
            .where(*filters)
            .options(selectinload(InspectionTask.detector_model))
            .order_by(InspectionTask.created_at.desc(), InspectionTask.id.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(self.session.scalars(statement)), total

    def get_image(self, image_id: uuid.UUID) -> InspectionImage | None:
        statement = (
            select(InspectionImage)
            .where(InspectionImage.id == image_id)
            .options(
                selectinload(InspectionImage.detections).selectinload(
                    Detection.classifications
                )
            )
        )
        return self.session.scalar(statement)

    def recompute_progress(self, task_id: uuid.UUID) -> InspectionTask:
        task = self.session.get(InspectionTask, task_id)
        if task is None:
            raise LookupError(f"inspection task not found: {task_id}")

        succeeded = sum(
            image.status == ImageStatus.succeeded.value for image in task.images
        )
        failed = sum(image.status == ImageStatus.failed.value for image in task.images)
        task.succeeded_images = succeeded
        task.failed_images = failed
        task.completed_images = succeeded + failed

        if task.total_images and task.completed_images == task.total_images:
            outcomes = [ImageOutcome.succeeded] * succeeded + [ImageOutcome.failed] * failed
            task.status = aggregate_task_status(outcomes)
            task.current_stage = ImageStage.complete
            task.finished_at = datetime.now(UTC)
            task.worker_id = None
            task.heartbeat_at = None
        self.session.flush()
        return task

    def reset_failed_image(self, image_id: uuid.UUID) -> InspectionImage:
        image = self.session.get(InspectionImage, image_id)
        if image is None:
            raise LookupError(f"inspection image not found: {image_id}")
        if image.status != ImageStatus.failed.value:
            raise ValueError("only failed images can be retried")

        image.status = ImageStatus.queued
        image.stage = ImageStage.pending
        image.overall_result = OverallResult.unknown
        image.result_path = None
        image.error_code = None
        image.error_message = None
        image.started_at = None
        image.finished_at = None
        image.detections.clear()

        task = image.task
        task.status = TaskStatus.queued
        task.current_stage = ImageStage.pending
        task.worker_id = None
        task.heartbeat_at = None
        task.finished_at = None
        self.recompute_progress(task.id)
        self.session.flush()
        return image

    @staticmethod
    def _expand_statuses(
        statuses: Sequence[TaskStatus | str] | None,
    ) -> list[str]:
        if not statuses:
            return []
        values = {TaskStatus(status).value for status in statuses}
        if TaskStatus.failed.value in values:
            values.add(TaskStatus.partial_failed.value)
        return sorted(values)
