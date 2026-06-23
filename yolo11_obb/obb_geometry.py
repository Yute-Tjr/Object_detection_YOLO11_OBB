from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


@dataclass(frozen=True)
class ObbBox:
    class_id: int
    points: Polygon
    confidence: Optional[float] = None


@dataclass(frozen=True)
class ObbMatch:
    ground_truth: ObbBox
    prediction: Optional[ObbBox]
    iou: float
    passed: bool


def parse_obb_line(
    line: str,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> ObbBox:
    parts = line.split()
    if len(parts) not in {9, 10}:
        raise ValueError(f"expected 9 or 10 YOLO-OBB fields, got {len(parts)}")

    values = [float(value) for value in parts]
    class_id = int(values[0])
    coords = values[1:9]
    points: List[Point] = []
    for idx in range(0, len(coords), 2):
        points.append((coords[idx] * image_width, coords[idx + 1] * image_height))
    confidence = values[9] if len(values) == 10 else None
    return ObbBox(class_id=class_id, points=tuple(points), confidence=confidence)  # type: ignore[arg-type]


def top_edge_width(points: Sequence[Point]) -> float:
    if len(points) != 4:
        raise ValueError(f"expected 4 OBB points, got {len(points)}")

    top_edge_index = min(
        range(4),
        key=lambda idx: (points[idx][1] + points[(idx + 1) % 4][1]) / 2.0,
    )
    start = points[top_edge_index]
    end = points[(top_edge_index + 1) % 4]
    return hypot(end[0] - start[0], end[1] - start[1])


def _as_cv_polygon(points: Sequence[Point]) -> np.ndarray:
    if len(points) != 4:
        raise ValueError(f"expected 4 OBB points, got {len(points)}")
    return np.array(points, dtype=np.float32)


def polygon_iou(left: Sequence[Point], right: Sequence[Point]) -> float:
    left_poly = _as_cv_polygon(left)
    right_poly = _as_cv_polygon(right)
    left_area = abs(float(cv2.contourArea(left_poly)))
    right_area = abs(float(cv2.contourArea(right_poly)))
    if left_area <= 0 or right_area <= 0:
        return 0.0

    intersection_area, _ = cv2.intersectConvexConvex(left_poly, right_poly)
    intersection = max(float(intersection_area), 0.0)
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def match_ground_truths(
    ground_truths: Sequence[ObbBox],
    predictions: Sequence[ObbBox],
    iou_threshold: float,
) -> List[ObbMatch]:
    matches: List[ObbMatch] = []
    for ground_truth in ground_truths:
        best_prediction: Optional[ObbBox] = None
        best_iou = 0.0
        for prediction in predictions:
            if prediction.class_id != ground_truth.class_id:
                continue
            iou = polygon_iou(ground_truth.points, prediction.points)
            if iou > best_iou:
                best_iou = iou
                best_prediction = prediction
        matches.append(
            ObbMatch(
                ground_truth=ground_truth,
                prediction=best_prediction,
                iou=best_iou,
                passed=best_iou >= iou_threshold,
            )
        )
    return matches
