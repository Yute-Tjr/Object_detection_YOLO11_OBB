from __future__ import annotations

import uuid
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from terminal_web.domain import ImageStage, TaskStatus
from terminal_web.models import InspectionTask


def claim_next_task(
    session: Session,
    worker_id: str,
    now: datetime,
) -> InspectionTask | None:
    statement = (
        select(InspectionTask)
        .where(InspectionTask.status == TaskStatus.queued.value)
        .order_by(InspectionTask.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    task = session.scalar(statement)
    if task is None:
        return None

    task.status = TaskStatus.running
    task.current_stage = ImageStage.object_detection
    task.attempt_count += 1
    task.worker_id = worker_id
    task.heartbeat_at = now
    if task.started_at is None:
        task.started_at = now
    session.flush()
    return task


def recover_stale_tasks(
    session: Session,
    now: datetime,
    timeout_seconds: int,
    max_attempts: int,
) -> list[uuid.UUID]:
    cutoff = now - timedelta(seconds=timeout_seconds)
    statement = (
        select(InspectionTask)
        .where(
            InspectionTask.status == TaskStatus.running.value,
            or_(
                InspectionTask.heartbeat_at.is_(None),
                InspectionTask.heartbeat_at < cutoff,
            ),
        )
        .with_for_update(skip_locked=True)
    )
    tasks = list(session.scalars(statement))
    for task in tasks:
        task.worker_id = None
        task.heartbeat_at = None
        if task.attempt_count < max_attempts:
            task.status = TaskStatus.queued
            task.current_stage = ImageStage.pending
        else:
            task.status = TaskStatus.failed
            task.current_stage = ImageStage.complete
            task.finished_at = now
    session.flush()
    return [task.id for task in tasks]
