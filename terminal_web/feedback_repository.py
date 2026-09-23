from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from terminal_web.feedback import FeedbackValue
from terminal_web.models import (
    ImageFeedback,
    ImageFeedbackItem,
    ImageFeedbackMiss,
    InspectionImage,
    utc_now,
)


class FeedbackRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_for_user_image(
        self,
        user_id: uuid.UUID,
        image_id: uuid.UUID,
    ) -> ImageFeedback | None:
        statement = (
            select(ImageFeedback)
            .where(
                ImageFeedback.user_id == user_id,
                ImageFeedback.image_id == image_id,
            )
            .options(
                selectinload(ImageFeedback.items),
                selectinload(ImageFeedback.misses),
            )
        )
        return self.session.scalar(statement)

    def upsert_feedback(
        self,
        user_id: uuid.UUID,
        image_id: uuid.UUID,
        items: Sequence[FeedbackValue],
        region_labels: dict[uuid.UUID, str],
        missed_regions: Sequence[str],
    ) -> ImageFeedback:
        statement = (
            select(ImageFeedback)
            .where(
                ImageFeedback.user_id == user_id,
                ImageFeedback.image_id == image_id,
            )
            .with_for_update()
        )
        feedback = self.session.scalar(statement)
        if feedback is None:
            try:
                with self.session.begin_nested():
                    feedback = ImageFeedback(user_id=user_id, image_id=image_id)
                    self.session.add(feedback)
                    self.session.flush()
            except IntegrityError:
                feedback = self.session.scalar(statement)
                if feedback is None:
                    raise

        self.session.execute(
            delete(ImageFeedbackItem).where(
                ImageFeedbackItem.feedback_id == feedback.id
            )
        )
        self.session.execute(
            delete(ImageFeedbackMiss).where(
                ImageFeedbackMiss.feedback_id == feedback.id
            )
        )
        self.session.flush()
        self.session.add_all(
            ImageFeedbackItem(
                feedback_id=feedback.id,
                detection_id=item.detection_id,
                region_label=region_labels[item.detection_id],
                source=item.source,
                verdict=item.verdict,
                color=item.color,
            )
            for item in items
        )
        self.session.add_all(
            ImageFeedbackMiss(feedback_id=feedback.id, logical_region=region)
            for region in missed_regions
        )
        feedback.updated_at = utc_now()
        self.session.flush()
        return feedback

    def task_has_feedback(self, task_id: uuid.UUID) -> bool:
        statement = select(
            select(ImageFeedback.id)
            .join(InspectionImage, ImageFeedback.image_id == InspectionImage.id)
            .where(InspectionImage.task_id == task_id)
            .exists()
        )
        return bool(self.session.scalar(statement))
