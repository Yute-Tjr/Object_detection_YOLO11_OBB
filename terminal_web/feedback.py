from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from terminal_web.models import Detection


LOGICAL_REGIONS = frozenset(
    {"label1", "label2", "label3", "label4", "label5", "label6"}
)
LOGICAL_REGION_ORDER = (
    "label1",
    "label2",
    "label3",
    "label4",
    "label5",
    "label6",
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
    verdict: str | None
    color: str | None
    source: str = "manual"


def logical_region(region_label: str) -> str:
    return "label1" if region_label in LABEL1_VARIANTS else region_label


def feedback_region_order(region_label: str) -> tuple[int, str]:
    region = logical_region(region_label)
    try:
        return LOGICAL_REGION_ORDER.index(region), region_label
    except ValueError:
        return len(LOGICAL_REGION_ORDER), region_label


def _model_feedback(detection: Detection) -> tuple[str | None, str | None]:
    classifications = {
        item.classifier_type: item.predicted_label
        for item in detection.classifications
    }
    verdict = classifications.get("anomaly")
    if verdict not in ALLOWED_VERDICTS:
        verdict = None
    color = classifications.get("color")
    if logical_region(detection.region_label) != "label1" or color not in ALLOWED_COLORS:
        color = None
    return verdict, color


def validate_feedback(
    detections: Sequence[Detection],
    submitted_items: Sequence[FeedbackValue],
    missed_regions: Sequence[str],
) -> tuple[tuple[FeedbackValue, ...], tuple[str, ...]]:
    detection_by_id = {detection.id: detection for detection in detections}
    submitted_ids = [item.detection_id for item in submitted_items]
    if len(submitted_ids) != len(set(submitted_ids)):
        raise FeedbackValidationError("detection IDs must be unique")
    if not set(submitted_ids).issubset(detection_by_id):
        raise FeedbackConflictError("detections changed; reload feedback before saving")

    if not submitted_items and not missed_regions:
        raise FeedbackValidationError("feedback requires a reviewed or missed region")

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
    submitted_by_id = {item.detection_id: item for item in submitted_items}
    snapshot = []
    for detection in sorted(detections, key=lambda item: feedback_region_order(item.region_label)):
        manual = submitted_by_id.get(detection.id)
        if manual is not None:
            snapshot.append(manual)
            continue
        verdict, color = _model_feedback(detection)
        snapshot.append(
            FeedbackValue(
                detection_id=detection.id,
                verdict=verdict,
                color=color,
                source="model" if verdict is not None or color is not None else "unreviewed",
            )
        )

    return tuple(snapshot), tuple(
        sorted(missed_set, key=feedback_region_order)
    )
