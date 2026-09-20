import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from terminal_web.api.dependencies import get_repository, get_storage
from terminal_web.api.presenters import image_detail, image_summary
from terminal_web.repositories import TaskRepository
from terminal_web.schemas import ImageDetail, ImageSummary
from terminal_web.storage import ArtifactStorage


router = APIRouter(prefix="/images", tags=["images"])


@router.get("/{image_id}", response_model=ImageDetail)
def get_image(
    image_id: uuid.UUID,
    repository: Annotated[TaskRepository, Depends(get_repository)],
) -> ImageDetail:
    image = repository.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image not found")
    return image_detail(image)


@router.get("/{image_id}/original", response_class=FileResponse)
def get_original(
    image_id: uuid.UUID,
    repository: Annotated[TaskRepository, Depends(get_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_storage)],
) -> FileResponse:
    image = repository.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image not found")
    path = storage.resolve(image.original_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="original image is missing")
    return FileResponse(path, filename=image.original_filename)


@router.get("/{image_id}/result", response_class=FileResponse)
def get_result(
    image_id: uuid.UUID,
    repository: Annotated[TaskRepository, Depends(get_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_storage)],
) -> FileResponse:
    image = repository.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image not found")
    if not image.result_path:
        raise HTTPException(status_code=409, detail="result image is not ready")
    path = storage.resolve(image.result_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="result image is missing")
    return FileResponse(path)


@router.post("/{image_id}/retry", response_model=ImageSummary)
def retry_image(
    image_id: uuid.UUID,
    repository: Annotated[TaskRepository, Depends(get_repository)],
) -> ImageSummary:
    try:
        image = repository.reset_failed_image(image_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="image not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    repository.session.commit()
    return image_summary(image)
