from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .classification_inference import (
    classification_transform,
    collect_image_paths,
    load_resnet18_checkpoint,
    select_device,
)
from .obb_crop import rectify_obb_crop


Point = Tuple[float, float]
ObbPoints = Tuple[Point, Point, Point, Point]
PIPELINE_LABELS = ("label3", "label5")
DETECTION_FIELDNAMES = [
    "image_path",
    "image_name",
    "det_index",
    "det_label",
    "det_conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
    "crop_path",
    "classifier",
    "predicted_label",
    "cls_confidence",
    "prob_NG",
    "prob_OK",
]
SUMMARY_FIELDNAMES = [
    "image_path",
    "image_name",
    "detections",
    "label3_pred",
    "label3_conf",
    "label3_det_conf",
    "label5_pred",
    "label5_conf",
    "label5_det_conf",
    "final_result",
    "warnings",
    "visualization_path",
]


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


@dataclass(frozen=True)
class PipelineClassifier:
    label: str
    weights: Path
    model: torch.nn.Module
    classes: List[str]
    device: torch.device
    transform: object


def _to_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _points_from_flat(flat_points: Sequence[float]) -> ObbPoints:
    if len(flat_points) == 4 and all(isinstance(point, (list, tuple)) for point in flat_points):
        pairs = flat_points
    else:
        if len(flat_points) != 8:
            raise ValueError(f"expected 8 OBB coordinate values, got {len(flat_points)}")
        pairs = [
            (flat_points[0], flat_points[1]),
            (flat_points[2], flat_points[3]),
            (flat_points[4], flat_points[5]),
            (flat_points[6], flat_points[7]),
        ]
    return tuple((float(point[0]), float(point[1])) for point in pairs)  # type: ignore[return-value]


def detections_from_yolo_result(result) -> List[PipelineDetection]:
    image_path = Path(str(result.path))
    obb = getattr(result, "obb", None)
    if obb is None:
        return []

    classes = _to_list(getattr(obb, "cls", []))
    confidences = _to_list(getattr(obb, "conf", []))
    polygons = _to_list(getattr(obb, "xyxyxyxy", []))
    names = getattr(result, "names", {})
    detections: List[PipelineDetection] = []

    for index, (class_id, confidence, polygon) in enumerate(zip(classes, confidences, polygons)):
        class_index = int(class_id)
        label = str(names.get(class_index, class_index))
        detections.append(
            PipelineDetection(
                image_path=image_path,
                image_name=image_path.name,
                det_index=index,
                det_label=label,
                det_conf=float(confidence),
                points=_points_from_flat(polygon),
            )
        )
    return detections


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
    from terminal_web.domain import OverallResult
    from terminal_web.inference.rendering import render_prediction
    from terminal_web.inference.types import (
        ClassificationPrediction,
        ImagePrediction,
        RegionPrediction,
    )

    regions = []
    for detection in detections:
        anomaly = None
        if detection.predicted_label:
            anomaly = ClassificationPrediction(
                label=detection.predicted_label,
                confidence=detection.cls_confidence or 0.0,
                probabilities=dict(detection.probabilities),
            )
        regions.append(
            RegionPrediction(
                region_label=detection.det_label,
                detection_confidence=detection.det_conf,
                points=detection.points,
                selected_for_classification=anomaly is not None,
                anomaly=anomaly,
                color=None,
                crop_path=detection.crop_path or None,
                error=None,
            )
        )
    selected_labels = {
        detection.det_label: detection.predicted_label
        for detection in detections
        if detection.predicted_label
    }
    render_prediction(
        image_path,
        ImagePrediction(
            regions=tuple(regions),
            overall_result=OverallResult(final_result_from_selected(selected_labels)),
            warnings=(),
        ),
        output,
    )


def load_pipeline_classifiers(
    label_weights: Mapping[str, Path],
    device_name: Optional[str],
    imgsz: int,
) -> Dict[str, PipelineClassifier]:
    device = select_device(device_name)
    transform = classification_transform(imgsz)
    classifiers: Dict[str, PipelineClassifier] = {}
    for label, weights_path in label_weights.items():
        model, classes, _ = load_resnet18_checkpoint(Path(weights_path), device)
        classifiers[label] = PipelineClassifier(
            label=label,
            weights=Path(weights_path),
            model=model,
            classes=classes,
            device=device,
            transform=transform,
        )
    return classifiers


def classify_detection_crop(
    image: np.ndarray,
    detection: PipelineDetection,
    classifier: PipelineClassifier,
    crop_output: Path,
) -> None:
    crop = rectify_obb_crop(image, detection.points)
    crop_output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_output), crop):
        raise RuntimeError(f"failed to write crop: {crop_output}")

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    tensor = classifier.transform(pil_image).unsqueeze(0).to(classifier.device)  # type: ignore[operator]
    with torch.no_grad():
        output = classifier.model(tensor)
        probabilities = torch.softmax(output, dim=1).detach().cpu().tolist()[0]
    predicted_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
    detection.crop_path = str(crop_output)
    detection.classifier = classifier.label
    detection.predicted_label = classifier.classes[predicted_index]
    detection.cls_confidence = float(probabilities[predicted_index])
    detection.probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(classifier.classes, probabilities)
    }


def _load_yolo_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed. Install it with python3 -m pip install ultralytics") from exc
    return YOLO(str(weights))


def _safe_stem(index: int, image_path: Path) -> str:
    return f"{index:06d}_{image_path.stem}"


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    det_weights: Path,
    source: Path,
    label3_weights: Path,
    label5_weights: Path,
    output: Path,
    imgsz: int,
    det_conf: float,
    det_device: Optional[str],
    cls_device: Optional[str],
    cls_imgsz: int,
    exist_ok: bool,
) -> Dict[str, object]:
    from terminal_web.inference.pipeline import InferenceArtifacts, LoadedInferencePipeline

    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not exist_ok:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    pipeline = LoadedInferencePipeline(
        detector_weights=Path(det_weights).expanduser().resolve(),
        label3_weights=Path(label3_weights).expanduser().resolve(),
        label5_weights=Path(label5_weights).expanduser().resolve(),
        detection_imgsz=imgsz,
        detection_confidence=det_conf,
        detection_device=det_device,
        classification_imgsz=cls_imgsz,
        classification_device=cls_device,
    )

    all_detection_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []
    images = 0
    detections_total = 0
    classified_total = 0

    for result_index, image_path in enumerate(collect_image_paths(Path(source))):
        key = _safe_stem(result_index, image_path)
        visualization_path = output / "visualizations" / f"{key}.jpg"
        prediction = pipeline.predict_image(
            image_path,
            InferenceArtifacts(
                crop_dir=output / "crops" / key,
                result_path=visualization_path,
            ),
            lambda stage: None,
        )
        detections: list[PipelineDetection] = []
        for det_index, region in enumerate(prediction.regions):
            detection = PipelineDetection(
                image_path=image_path,
                image_name=image_path.name,
                det_index=det_index,
                det_label=region.region_label,
                det_conf=region.detection_confidence,
                points=tuple(region.points),  # type: ignore[arg-type]
            )
            if region.crop_path:
                detection.crop_path = str(Path(region.crop_path).relative_to(output))
            if region.anomaly:
                detection.classifier = region.region_label
                detection.predicted_label = region.anomaly.label
                detection.cls_confidence = region.anomaly.confidence
                detection.probabilities = dict(region.anomaly.probabilities)
                classified_total += 1
            detections.append(detection)

        all_detection_rows.extend(format_detection_row(detection) for detection in detections)
        summary_rows.append(
            summary_row_for_image(
                image_path=image_path,
                detections=detections,
                visualization_path=visualization_path.relative_to(output),
            )
        )
        images += 1
        detections_total += len(detections)

    _write_csv(output / "detections.csv", DETECTION_FIELDNAMES, all_detection_rows)
    _write_csv(output / "summary.csv", SUMMARY_FIELDNAMES, summary_rows)
    report = {
        "output": str(output),
        "images": images,
        "detections": detections_total,
        "classified": classified_total,
        "classifiers": sorted(pipeline.classifiers),
    }
    lines = [f"{key}: {value}" for key, value in report.items()]
    (output / "pipeline_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
