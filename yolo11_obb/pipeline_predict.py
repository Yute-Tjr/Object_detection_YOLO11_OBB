from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
ObbPoints = Tuple[Point, Point, Point, Point]
PIPELINE_LABELS = ("label3", "label5")


@dataclass
class PipelineDetection:
    image_path: Path
    image_name: str
    det_index: int
    det_label: str
    det_conf: float
    points: ObbPoints
    crop_path: str = ""
    classifier: str = ""
    predicted_label: str = ""
    cls_confidence: float | None = None
    probabilities: Dict[str, float] = field(default_factory=dict)


def selected_detection_by_label(
    detections: Sequence[PipelineDetection],
    required_labels: Iterable[str] = PIPELINE_LABELS,
) -> tuple[Dict[str, PipelineDetection], List[str]]:
    selected: Dict[str, PipelineDetection] = {}
    warnings: List[str] = []
    for label in required_labels:
        matches = [detection for detection in detections if detection.det_label == label]
        if not matches:
            warnings.append(f"missing {label}")
            continue
        if len(matches) > 1:
            warnings.append(f"duplicate {label} count={len(matches)}")
        selected[label] = max(matches, key=lambda detection: detection.det_conf)
    return selected, warnings


def final_result_from_selected(selected_labels: Mapping[str, str]) -> str:
    if any(value == "NG" for value in selected_labels.values()):
        return "NG"
    if all(selected_labels.get(label) == "OK" for label in PIPELINE_LABELS):
        return "OK"
    return "UNKNOWN"


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def format_detection_row(detection: PipelineDetection) -> Dict[str, str]:
    row = {
        "image_path": str(detection.image_path),
        "image_name": detection.image_name,
        "det_index": str(detection.det_index),
        "det_label": detection.det_label,
        "det_conf": f"{detection.det_conf:.6f}",
        "crop_path": detection.crop_path,
        "classifier": detection.classifier,
        "predicted_label": detection.predicted_label,
        "cls_confidence": _format_optional_float(detection.cls_confidence),
        "prob_NG": _format_optional_float(detection.probabilities.get("NG")),
        "prob_OK": _format_optional_float(detection.probabilities.get("OK")),
    }
    for index, point in enumerate(detection.points, 1):
        row[f"x{index}"] = f"{point[0]:.6f}"
        row[f"y{index}"] = f"{point[1]:.6f}"
    return row


def summary_row_for_image(
    image_path: Path,
    detections: Sequence[PipelineDetection],
    visualization_path: Path,
) -> Dict[str, str]:
    selected, warnings = selected_detection_by_label(detections, PIPELINE_LABELS)
    selected_labels = {
        label: detection.predicted_label
        for label, detection in selected.items()
        if detection.predicted_label
    }
    row = {
        "image_path": str(image_path),
        "image_name": Path(image_path).name,
        "detections": str(len(detections)),
        "label3_pred": "",
        "label3_conf": "",
        "label3_det_conf": "",
        "label5_pred": "",
        "label5_conf": "",
        "label5_det_conf": "",
        "final_result": final_result_from_selected(selected_labels),
        "warnings": "; ".join(warnings),
        "visualization_path": str(visualization_path),
    }
    for label in PIPELINE_LABELS:
        detection = selected.get(label)
        if detection is None:
            continue
        row[f"{label}_pred"] = detection.predicted_label
        row[f"{label}_conf"] = _format_optional_float(detection.cls_confidence)
        row[f"{label}_det_conf"] = f"{detection.det_conf:.6f}"
    return row


def _box_color(detection: PipelineDetection) -> tuple[int, int, int]:
    if detection.predicted_label == "NG":
        return (0, 0, 255)
    if detection.predicted_label == "OK":
        return (0, 180, 0)
    return (0, 200, 255)


def _visualization_text(detection: PipelineDetection) -> str:
    text = f"{detection.det_label} det={detection.det_conf:.2f}"
    if detection.predicted_label:
        text += f" {detection.predicted_label}"
        if detection.cls_confidence is not None:
            text += f" cls={detection.cls_confidence:.2f}"
    return text


def draw_visualization(
    image_path: Path,
    detections: Sequence[PipelineDetection],
    output: Path,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")

    for detection in detections:
        color = _box_color(detection)
        points = np.array(detection.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
        label_origin = tuple(points.reshape(-1, 2)[0])
        text_origin = (int(label_origin[0]), max(int(label_origin[1]) - 6, 12))
        cv2.putText(
            image,
            _visualization_text(detection),
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write visualization: {output}")
