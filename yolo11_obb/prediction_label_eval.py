from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import cv2

from .config import IMAGE_EXTENSIONS, load_dataset_config
from .obb_geometry import ObbBox, parse_obb_line, polygon_iou


IOU_THRESHOLDS = {
    "mAP50": 0.50,
    "mAP80": 0.80,
    "mAP85": 0.85,
    "mAP90": 0.90,
    "mAP95": 0.95,
}
CSV_COLUMNS = ("class", "precision", "recall", "mAP50", "mAP80", "mAP85", "mAP90", "mAP95")


@dataclass(frozen=True)
class Prediction:
    image: str
    box: ObbBox


def _label_dir_for_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def _image_files(image_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_boxes(label_path: Path, image_width: int, image_height: int) -> List[ObbBox]:
    if not label_path.exists():
        return []
    boxes: List[ObbBox] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            boxes.append(parse_obb_line(line, image_width=image_width, image_height=image_height))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no}: {exc}") from exc
    return boxes


def _average_precision(
    predictions: Sequence[Prediction],
    gt_by_image: Mapping[str, Sequence[ObbBox]],
    iou_threshold: float,
) -> float:
    gt_count = sum(len(items) for items in gt_by_image.values())
    if gt_count == 0:
        return 0.0
    matched = {image: [False] * len(items) for image, items in gt_by_image.items()}
    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: prediction.box.confidence if prediction.box.confidence is not None else 1.0,
        reverse=True,
    )
    tp: List[float] = []
    fp: List[float] = []

    for prediction in sorted_predictions:
        gt_boxes = gt_by_image.get(prediction.image, [])
        best_iou = 0.0
        best_index = -1
        for index, gt_box in enumerate(gt_boxes):
            if matched[prediction.image][index]:
                continue
            iou = polygon_iou(prediction.box.points, gt_box.points)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[prediction.image][best_index] = True
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)

    if not tp:
        return 0.0

    cumulative_tp: List[float] = []
    cumulative_fp: List[float] = []
    total_tp = 0.0
    total_fp = 0.0
    for tp_value, fp_value in zip(tp, fp):
        total_tp += tp_value
        total_fp += fp_value
        cumulative_tp.append(total_tp)
        cumulative_fp.append(total_fp)

    recalls = [value / gt_count for value in cumulative_tp]
    precisions = [
        cumulative_tp[index] / max(cumulative_tp[index] + cumulative_fp[index], 1e-12)
        for index in range(len(cumulative_tp))
    ]
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def _precision_recall_at_50(
    predictions: Sequence[Prediction],
    gt_by_image: Mapping[str, Sequence[ObbBox]],
) -> Tuple[float, float]:
    gt_count = sum(len(items) for items in gt_by_image.values())
    if gt_count == 0:
        return 0.0, 0.0
    matched = {image: [False] * len(items) for image, items in gt_by_image.items()}
    tp = 0
    fp = 0
    for prediction in sorted(
        predictions,
        key=lambda item: item.box.confidence if item.box.confidence is not None else 1.0,
        reverse=True,
    ):
        gt_boxes = gt_by_image.get(prediction.image, [])
        best_iou = 0.0
        best_index = -1
        for index, gt_box in enumerate(gt_boxes):
            if matched[prediction.image][index]:
                continue
            iou = polygon_iou(prediction.box.points, gt_box.points)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= 0.50:
            matched[prediction.image][best_index] = True
            tp += 1
        else:
            fp += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / gt_count
    return precision, recall


def evaluate_prediction_labels(
    data_yaml: Union[str, Path],
    pred_labels: Union[str, Path],
    split: str = "test",
) -> List[Dict[str, float]]:
    dataset = load_dataset_config(data_yaml)
    image_dir = dataset.splits[split]
    gt_label_dir = _label_dir_for_image_dir(image_dir)
    pred_labels = Path(pred_labels)

    gt_by_class: Dict[int, Dict[str, List[ObbBox]]] = {class_id: {} for class_id in dataset.names}
    pred_by_class: Dict[int, List[Prediction]] = {class_id: [] for class_id in dataset.names}

    for image_path in _image_files(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        gt_boxes = _read_boxes(gt_label_dir / f"{image_path.stem}.txt", width, height)
        pred_boxes = _read_boxes(pred_labels / f"{image_path.stem}.txt", width, height)
        for box in gt_boxes:
            gt_by_class.setdefault(box.class_id, {}).setdefault(image_path.name, []).append(box)
        for box in pred_boxes:
            pred_by_class.setdefault(box.class_id, []).append(Prediction(image_path.name, box))

    class_rows: List[Dict[str, float]] = []
    for class_id in sorted(dataset.names):
        gt_for_class = gt_by_class.get(class_id, {})
        predictions = pred_by_class.get(class_id, [])
        precision, recall = _precision_recall_at_50(predictions, gt_for_class)
        row: Dict[str, float] = {
            "class": dataset.names[class_id],  # type: ignore[assignment]
            "precision": precision,
            "recall": recall,
        }
        for column, threshold in IOU_THRESHOLDS.items():
            row[column] = _average_precision(predictions, gt_for_class, threshold)
        class_rows.append(row)

    all_row: Dict[str, float] = {"class": "all"}  # type: ignore[assignment]
    for column in CSV_COLUMNS[1:]:
        values = [float(row[column]) for row in class_rows]
        all_row[column] = sum(values) / len(values) if values else 0.0
    return [all_row] + class_rows


def write_prediction_metrics_csv(rows: Sequence[Mapping[str, object]], output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: row[column] if column == "class" else f"{float(row[column]):.6f}"
                    for column in CSV_COLUMNS
                }
            )
    return output_path
