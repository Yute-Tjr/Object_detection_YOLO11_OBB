from __future__ import annotations

import pickle
from math import cos, sin
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


def index_images_by_stem(image_paths: Iterable[Path]) -> Dict[str, Path]:
    indexed: Dict[str, Path] = {}
    for image_path in image_paths:
        stem = image_path.stem
        if stem in indexed:
            raise ValueError(f"duplicate image stem {stem!r}: {indexed[stem]} and {image_path}")
        indexed[stem] = image_path
    return indexed


def rbox_to_corners(rbox: Sequence[float]) -> List[Tuple[float, float]]:
    if len(rbox) != 5:
        raise ValueError(f"expected rbox [cx, cy, width, height, angle], got {rbox!r}")
    cx, cy, width, height, angle = (float(value) for value in rbox)
    half_width = width / 2.0
    half_height = height / 2.0
    base = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    ]
    c, s = cos(angle), sin(angle)
    return [
        (round(cx + x * c - y * s, 6), round(cy + x * s + y * c, 6))
        for x, y in base
    ]


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
        normalized: list[float] = []
        for x, y in corners:
            normalized.extend([x / image_width, y / image_height])
        coords = " ".join(f"{value:.6f}" for value in normalized)
        lines.append(f"{int(label)} {coords} {float(score):.6f}")
    return lines


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _numpy(value: Any) -> np.ndarray:
    value = _field(value, "tensor", value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sample_path(sample: Any) -> Path:
    metainfo = _field(sample, "metainfo", {})
    image_path = _field(metainfo, "img_path")
    if image_path is None:
        image_path = _field(sample, "img_path")
    if image_path is None:
        raise ValueError("MMRotate prediction sample has no img_path")
    return Path(str(image_path))


def prediction_samples(payload: Any) -> list[Any]:
    """Handle a direct sample list and common MMEngine ``--out`` wrappers."""
    if isinstance(payload, Mapping):
        for key in ("predictions", "results"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise ValueError("MMRotate prediction pickle must contain prediction samples")
    return list(payload)


def export_mmrotate_predictions(
    *,
    predictions_path: Path,
    image_paths: Iterable[Path],
    output_dir: Path,
    min_conf: float = 0.001,
) -> int:
    """Convert MMRotate ``DetDataSample`` predictions to YOLO OBB labels."""
    if not 0.0 <= min_conf <= 1.0:
        raise ValueError("min_conf must be between 0 and 1")
    predictions_path = predictions_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    indexed_images = index_images_by_stem(image_paths)
    with predictions_path.open("rb") as handle:
        samples = prediction_samples(pickle.load(handle))

    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for sample in samples:
        prediction_image = _sample_path(sample)
        image_path = indexed_images.get(prediction_image.stem)
        if image_path is None:
            raise ValueError(f"prediction image is outside the requested split: {prediction_image}")
        if image_path.stem in seen:
            raise ValueError(f"duplicate prediction sample: {image_path.stem}")
        seen.add(image_path.stem)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        instances = _field(sample, "pred_instances")
        if instances is None:
            raise ValueError(f"prediction sample has no pred_instances: {prediction_image}")
        rboxes = _numpy(_field(instances, "bboxes"))
        labels = _numpy(_field(instances, "labels")).astype(int)
        scores = _numpy(_field(instances, "scores")).astype(float)
        keep = scores >= min_conf
        lines = rboxes_to_yolo_lines(rboxes[keep], labels[keep], scores[keep], width, height)
        (output_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    missing = sorted(set(indexed_images) - seen)
    if missing:
        raise ValueError(f"predictions are missing {len(missing)} images, first entries: {missing[:5]}")
    return len(seen)

