from __future__ import annotations

from math import hypot
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def ordered_polygon_points(points: Sequence[Sequence[float]]) -> list[Point]:
    if len(points) != 4:
        raise ValueError(f"expected 4 points, got {len(points)}")
    array = np.array(points, dtype=np.float32)
    sums = array.sum(axis=1)
    diffs = np.diff(array, axis=1).reshape(-1)
    top_left = array[int(np.argmin(sums))]
    bottom_right = array[int(np.argmax(sums))]
    top_right = array[int(np.argmin(diffs))]
    bottom_left = array[int(np.argmax(diffs))]
    return [
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    ]


def _target_size(points: Sequence[Point]) -> tuple[int, int]:
    top_width = hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    bottom_width = hypot(points[2][0] - points[3][0], points[2][1] - points[3][1])
    left_height = hypot(points[3][0] - points[0][0], points[3][1] - points[0][1])
    right_height = hypot(points[2][0] - points[1][0], points[2][1] - points[1][1])
    width = max(int(round(max(top_width, bottom_width))), 1)
    height = max(int(round(max(left_height, right_height))), 1)
    return width, height


def rectify_obb_crop(image: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    ordered = ordered_polygon_points(points)
    width, height = _target_size(ordered)
    source = np.array(ordered, dtype=np.float32)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def save_obb_crop(image: np.ndarray, points: Sequence[Sequence[float]], output: Path) -> None:
    crop = rectify_obb_crop(image, points)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), crop):
        raise RuntimeError(f"failed to write crop: {output}")
