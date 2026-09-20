from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from terminal_web.domain import OverallResult


@dataclass(frozen=True)
class ClassificationPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class RegionPrediction:
    region_label: str
    detection_confidence: float
    points: Sequence[tuple[float, float]]
    selected_for_classification: bool
    anomaly: ClassificationPrediction | None
    color: ClassificationPrediction | None
    crop_path: str | None
    error: str | None


@dataclass(frozen=True)
class ImagePrediction:
    regions: Sequence[RegionPrediction]
    overall_result: OverallResult
    warnings: Sequence[str]
