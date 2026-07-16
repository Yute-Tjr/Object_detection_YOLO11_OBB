from __future__ import annotations

from math import cos, sin
from typing import Iterable, List, Sequence, Tuple


def rbox_to_corners(rbox: Sequence[float]) -> List[Tuple[float, float]]:
    if len(rbox) != 5:
        raise ValueError(f"expected rbox [cx, cy, width, height, angle], got {rbox!r}")
    cx, cy, width, height, angle = (float(value) for value in rbox)
    half_width = width / 2.0
    half_height = height / 2.0
    base = [(-half_width, -half_height), (half_width, -half_height), (half_width, half_height), (-half_width, half_height)]
    c, s = cos(angle), sin(angle)
    return [(round(cx + x * c - y * s, 6), round(cy + x * s + y * c, 6)) for x, y in base]


def rboxes_to_yolo_lines(
    rboxes: Iterable[Sequence[float]],
    labels: Iterable[int],
    scores: Iterable[float],
    image_width: float,
    image_height: float,
) -> List[str]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image width and height must be positive")
    lines: List[str] = []
    for rbox, label, score in zip(rboxes, labels, scores):
        corners = rbox_to_corners(rbox)
        normalized = []
        for x, y in corners:
            normalized.extend([x / image_width, y / image_height])
        coords = " ".join(f"{value:.6f}" for value in normalized)
        lines.append(f"{int(label)} {coords} {float(score):.6f}")
    return lines
