from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from terminal_web.api.dependencies import CurrentUser, get_repository
from terminal_web.api.presenters import detection_response
from terminal_web.domain import ImageStage, ImageStatus
from terminal_web.feedback import (
    FeedbackConflictError,
    FeedbackValidationError,
    FeedbackValue,
    LOGICAL_REGION_ORDER,
    LOGICAL_REGIONS,
    feedback_region_order,
    logical_region,
    validate_feedback,
)
from terminal_web.feedback_repository import FeedbackRepository
from terminal_web.models import ImageFeedback, InspectionImage
from terminal_web.repositories import TaskRepository
from terminal_web.schemas import (
    FeedbackDetectionResponse,
    FeedbackItemResponse,
    FeedbackRecordResponse,
    FeedbackUpdateRequest,
    ImageFeedbackView,
)


router = APIRouter(prefix="/images", tags=["feedback"])


def _feedback_view(
    image: InspectionImage,
    feedback: ImageFeedback | None,
) -> ImageFeedbackView:
    detections = []
    ordered_detections = sorted(
        image.detections,
        key=lambda item: feedback_region_order(item.region_label),
    )
    for detection in ordered_detections:
        presented = detection_response(detection)
        detections.append(
            FeedbackDetectionResponse(
                detection_id=detection.id,
                region_label=detection.region_label,
                logical_region=logical_region(detection.region_label),
                anomaly=presented.anomaly,
                color=presented.color,
            )
        )
    detected_regions = {item.logical_region for item in detections}
    candidates = [
        region for region in LOGICAL_REGION_ORDER if region not in detected_regions
    ]

    feedback_response = None
    if feedback is not None:
        item_by_detection = {item.detection_id: item for item in feedback.items}
        ordered_items = []
        for detection in ordered_detections:
            item = item_by_detection.get(detection.id)
            if item is None:
                continue
            ordered_items.append(
                FeedbackItemResponse(
                    detection_id=item.detection_id,
                    region_label=item.region_label,
                    logical_region=logical_region(item.region_label),
                    source=item.source,
                    verdict=item.verdict,
                    color=item.color,
                )
            )
        feedback_response = FeedbackRecordResponse(
            id=feedback.id,
            items=ordered_items,
            missed_regions=sorted(item.logical_region for item in feedback.misses),
            created_at=feedback.created_at,
            updated_at=feedback.updated_at,
        )

    return ImageFeedbackView(
        image_id=image.id,
        original_filename=image.original_filename,
        status=image.status,
        detections=detections,
        missed_region_candidates=candidates,
        feedback=feedback_response,
    )


def _load_image(
    image_id: uuid.UUID,
    repository: TaskRepository,
) -> InspectionImage:
    image = repository.get_image(image_id)
    if image is None:
        raise HTTPException(status_code=404, detail="image not found")
    return image


@router.get("/{image_id}/feedback", response_model=ImageFeedbackView)
def get_feedback(
    image_id: uuid.UUID,
    user: CurrentUser,
    repository: Annotated[TaskRepository, Depends(get_repository)],
) -> ImageFeedbackView:
    image = _load_image(image_id, repository)
    feedback = FeedbackRepository(repository.session).get_for_user_image(
        user.id, image.id
    )
    return _feedback_view(image, feedback)


@router.put("/{image_id}/feedback", response_model=ImageFeedbackView)
def update_feedback(
    image_id: uuid.UUID,
    payload: FeedbackUpdateRequest,
    user: CurrentUser,
    repository: Annotated[TaskRepository, Depends(get_repository)],
) -> ImageFeedbackView:
    image = _load_image(image_id, repository)
    if not (
        image.status == ImageStatus.succeeded.value
        and image.stage == ImageStage.complete.value
    ):
        raise HTTPException(status_code=409, detail="image is not ready for feedback")

    submitted = tuple(
        FeedbackValue(
            detection_id=item.detection_id,
            verdict=item.verdict,
            color=item.color,
            source="manual",
        )
        for item in payload.items
    )
    try:
        items, missed_regions = validate_feedback(
            image.detections,
            submitted,
            payload.missed_regions,
        )
    except FeedbackConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FeedbackValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    feedback_repository = FeedbackRepository(repository.session)
    feedback_repository.upsert_feedback(
        user.id,
        image.id,
        items,
        {detection.id: detection.region_label for detection in image.detections},
        missed_regions,
    )
    repository.session.commit()
    feedback = feedback_repository.get_for_user_image(user.id, image.id)
    return _feedback_view(image, feedback)


@router.delete("/{image_id}/feedback", response_model=ImageFeedbackView)
def delete_feedback(
    image_id: uuid.UUID,
    user: CurrentUser,
    repository: Annotated[TaskRepository, Depends(get_repository)],
) -> ImageFeedbackView:
    image = _load_image(image_id, repository)
    feedback_repository = FeedbackRepository(repository.session)
    feedback_repository.delete_for_user_image(user.id, image.id)
    repository.session.commit()
    return _feedback_view(image, None)
