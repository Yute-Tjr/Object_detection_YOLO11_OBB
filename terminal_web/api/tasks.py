from __future__ import annotations

import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.exc import SQLAlchemyError

from terminal_web.api.dependencies import (
    ReadinessProvider,
    get_app_settings,
    get_readiness,
    get_repository,
    get_storage,
)
from terminal_web.api.presenters import image_summary, task_detail, task_summary
from terminal_web.domain import TaskStatus
from terminal_web.models import InspectionImage
from terminal_web.repositories import TaskRepository
from terminal_web.schemas import ImagePage, TaskDetail, TaskPage
from terminal_web.storage import ArtifactStorage, InvalidImageError


router = APIRouter(prefix="/tasks", tags=["tasks"])


def _statuses_for_filter(value: str | None) -> list[TaskStatus] | None:
    if value is None or value == "all":
        return None
    if value == "ongoing":
        return [TaskStatus.queued, TaskStatus.running]
    if value == "success":
        return [TaskStatus.succeeded]
    if value == "failed":
        return [TaskStatus.failed, TaskStatus.partial_failed]
    try:
        return [TaskStatus(value)]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid task status filter") from exc


@router.post("", response_model=TaskDetail, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    files: Annotated[list[UploadFile], File()],
    repository: Annotated[TaskRepository, Depends(get_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_storage)],
    readiness: Annotated[ReadinessProvider, Depends(get_readiness)],
    operator: Annotated[str, Form(min_length=1, max_length=128)],
    settings=Depends(get_app_settings),
    name: Annotated[str | None, Form()] = None,
    note: Annotated[str | None, Form()] = None,
) -> TaskDetail:
    operator = operator.strip()
    if not operator:
        raise HTTPException(status_code=422, detail="操作员不能为空")

    health = readiness.snapshot()
    if not (
        health.api_ready
        and health.database_ready
        and health.worker_ready
        and health.models_ready
    ):
        raise HTTPException(status_code=503, detail="inspection service is not ready")
    if not 1 <= len(files) <= settings.max_images_per_task:
        raise HTTPException(
            status_code=400,
            detail=f"upload must contain 1-{settings.max_images_per_task} images",
        )

    task_id = uuid.uuid4()
    staged_root = Path(tempfile.mkdtemp(prefix=f".staging-{task_id}-", dir=storage.root))
    staged_storage = ArtifactStorage(staged_root)
    staged_uploads = []
    image_ids = []
    final_task_dir = storage.root / "tasks" / str(task_id)
    moved_to_final = False
    committed = False
    try:
        for upload in files:
            image_id = uuid.uuid4()
            content = await upload.read()
            staged_uploads.append(
                staged_storage.save_upload(
                    task_id,
                    image_id,
                    upload.filename or "",
                    content,
                )
            )
            image_ids.append(image_id)

        staged_task_dir = staged_root / "tasks" / str(task_id)
        final_task_dir.parent.mkdir(parents=True, exist_ok=True)
        staged_task_dir.replace(final_task_dir)
        moved_to_final = True

        images = [
            InspectionImage(
                id=image_id,
                sequence_no=index,
                original_filename=stored.original_filename,
                stored_filename=stored.stored_filename,
                original_path=stored.relative_path,
                width=stored.width,
                height=stored.height,
                size_bytes=stored.size_bytes,
            )
            for index, (image_id, stored) in enumerate(zip(image_ids, staged_uploads))
        ]
        display_id = f"T{datetime.now(UTC):%Y%m%d}-{task_id.hex[:8].upper()}"
        task = repository.create_task(
            task_id,
            display_id,
            images,
            operator=operator,
            name=name,
            note=note,
        )
        repository.session.commit()
        committed = True
        return task_detail(task)
    except InvalidImageError as exc:
        repository.session.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        repository.session.rollback()
        raise HTTPException(status_code=503, detail="database is unavailable") from exc
    finally:
        shutil.rmtree(staged_root, ignore_errors=True)
        if moved_to_final and not committed:
            shutil.rmtree(final_task_dir, ignore_errors=True)


@router.get("", response_model=TaskPage)
def list_tasks(
    repository: Annotated[TaskRepository, Depends(get_repository)],
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    query: str | None = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> TaskPage:
    tasks, total = repository.list_tasks(
        _statuses_for_filter(status_filter), query, offset, limit
    )
    return TaskPage(
        items=[task_summary(task) for task in tasks],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.get("/{task_id}", response_model=TaskDetail)
def get_task(
    task_id: uuid.UUID,
    repository: Annotated[TaskRepository, Depends(get_repository)],
) -> TaskDetail:
    task = repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task_detail(task)


@router.get("/{task_id}/images", response_model=ImagePage)
def list_task_images(
    task_id: uuid.UUID,
    repository: Annotated[TaskRepository, Depends(get_repository)],
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> ImagePage:
    task = repository.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    images = task.images[offset : offset + limit]
    return ImagePage(
        items=[image_summary(image) for image in images],
        total=len(task.images),
        offset=offset,
        limit=limit,
    )
