from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from terminal_web.models import Detection


LOGICAL_REGIONS = frozenset(
    {"label1", "label2", "label3", "label4", "label5", "label6"}
)
LABEL1_VARIANTS = frozenset({"label1_thin", "label1_thick"})
ALLOWED_COLORS = frozenset({"B", "G", "R", "W"})
ALLOWED_VERDICTS = frozenset({"OK", "NG"})


class FeedbackValidationError(ValueError):
    pass


class FeedbackConflictError(ValueError):
    pass


@dataclass(frozen=True)
class FeedbackValue:
    detection_id: uuid.UUID
    verdict: str
    color: str | None


def logical_region(region_label: str) -> str:
    return "label1" if region_label in LABEL1_VARIANTS else region_label


def validate_feedback(
    detections: Sequence[Detection],
    submitted_items: Sequence[FeedbackValue],
    missed_regions: Sequence[str],
) -> tuple[tuple[FeedbackValue, ...], tuple[str, ...]]:
    detection_by_id = {detection.id: detection for detection in detections}
    submitted_ids = [item.detection_id for item in submitted_items]
    if len(submitted_ids) != len(set(submitted_ids)):
        raise FeedbackValidationError("detection IDs must be unique")
    if set(submitted_ids) != set(detection_by_id):
        raise FeedbackConflictError("detections changed; reload feedback before saving")

    for item in submitted_items:
        if item.verdict not in ALLOWED_VERDICTS:
            raise FeedbackValidationError("verdict must be OK or NG")
        detection = detection_by_id[item.detection_id]
        if detection.region_label in LABEL1_VARIANTS:
            if item.color not in ALLOWED_COLORS:
                raise FeedbackValidationError("label1 feedback requires B/G/R/W color")
        elif item.color is not None:
            raise FeedbackValidationError("color is only allowed for label1")

    if len(missed_regions) != len(set(missed_regions)):
        raise FeedbackValidationError("missed regions must be unique")
    missed_set = set(missed_regions)
    if not missed_set.issubset(LOGICAL_REGIONS):
        raise FeedbackValidationError("invalid missed region")
    detected_regions = {logical_region(item.region_label) for item in detections}
    if missed_set & detected_regions:
        raise FeedbackValidationError("detected regions cannot be marked as missed")
    if not detections and not missed_set:
        raise FeedbackValidationError(
            "an image without detections requires at least one missed region"
        )

    return tuple(submitted_items), tuple(sorted(missed_set))
