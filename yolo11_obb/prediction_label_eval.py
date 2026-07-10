from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Union

import cv2
import numpy as np

from .config import IMAGE_EXTENSIONS, load_dataset_config
from .obb_geometry import ObbBox, ObbMatch, parse_obb_line


ULTRALYTICS_IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


class UltralyticsObbMetricBackend:
    def __init__(self) -> None:
        try:
            import torch
            from ultralytics.utils.metrics import OBBMetrics, batch_probiou
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics OBB metric evaluation requires torch and ultralytics. "
                "Run this on the same environment used for YOLO evaluation."
            ) from exc
        self._torch = torch
        self._OBBMetrics = OBBMetrics
        self._batch_probiou = batch_probiou

    def create_metrics(self, names: Mapping[int, str]):
        return self._OBBMetrics(dict(names))

    def probiou(self, gt_xywhr: np.ndarray, pred_xywhr: np.ndarray):
        gt = self._torch.as_tensor(gt_xywhr, dtype=self._torch.float32)
        pred = self._torch.as_tensor(pred_xywhr, dtype=self._torch.float32)
        return self._batch_probiou(gt, pred)


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


def _boxes_to_xywhr(boxes: Sequence[ObbBox]) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)

    rboxes: List[List[float]] = []
    for box in boxes:
        pts = np.asarray(box.points, dtype=np.float32)
        (cx, cy), (width, height), angle = cv2.minAreaRect(pts)
        theta = angle / 180.0 * np.pi
        if width < height:
            width, height = height, width
            theta += np.pi / 2.0
        while theta >= 3.0 * np.pi / 4.0:
            theta -= np.pi
        while theta < -np.pi / 4.0:
            theta += np.pi
        rboxes.append([cx, cy, width, height, theta])
    return np.asarray(rboxes, dtype=np.float32)


def _classes_array(boxes: Sequence[ObbBox]) -> np.ndarray:
    return np.asarray([box.class_id for box in boxes], dtype=np.float32)


def _confidence_array(boxes: Sequence[ObbBox]) -> np.ndarray:
    return np.asarray(
        [1.0 if box.confidence is None else box.confidence for box in boxes],
        dtype=np.float32,
    )


def _to_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def _match_predictions(
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
    iou: np.ndarray,
    iou_thresholds: np.ndarray = ULTRALYTICS_IOU_THRESHOLDS,
) -> np.ndarray:
    correct = np.zeros((pred_classes.shape[0], iou_thresholds.shape[0]), dtype=bool)
    if pred_classes.shape[0] == 0 or true_classes.shape[0] == 0:
        return correct

    iou = np.asarray(iou) * (true_classes[:, None] == pred_classes[None, :])
    for idx, threshold in enumerate(iou_thresholds.tolist()):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), idx] = True
    return correct


def probiou_matrix(
    ground_truths: Sequence[ObbBox],
    predictions: Sequence[ObbBox],
    metric_backend=None,
) -> np.ndarray:
    if not ground_truths or not predictions:
        return np.zeros((len(ground_truths), len(predictions)), dtype=np.float32)
    backend = metric_backend or UltralyticsObbMetricBackend()
    return _to_numpy(backend.probiou(_boxes_to_xywhr(ground_truths), _boxes_to_xywhr(predictions))).astype(np.float32)


def match_ground_truths_ultralytics(
    ground_truths: Sequence[ObbBox],
    predictions: Sequence[ObbBox],
    iou_threshold: float,
    metric_backend=None,
) -> List[ObbMatch]:
    scores = probiou_matrix(ground_truths, predictions, metric_backend=metric_backend)
    matches: List[ObbMatch] = []
    for gt_index, ground_truth in enumerate(ground_truths):
        best_prediction = None
        best_iou = 0.0
        for pred_index, prediction in enumerate(predictions):
            if prediction.class_id != ground_truth.class_id:
                continue
            iou = float(scores[gt_index, pred_index])
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


def evaluate_prediction_labels_ultralytics(
    data_yaml: Union[str, Path],
    pred_labels: Union[str, Path],
    split: str = "test",
    save_dir: Union[str, Path, None] = None,
    plot: bool = False,
    metric_backend=None,
):
    dataset = load_dataset_config(data_yaml)
    image_dir = dataset.splits[split]
    gt_label_dir = _label_dir_for_image_dir(image_dir)
    pred_labels = Path(pred_labels)
    output_dir = Path(save_dir) if save_dir is not None else pred_labels.parent
    backend = metric_backend or UltralyticsObbMetricBackend()
    metrics = backend.create_metrics(dataset.names)

    for image_path in _image_files(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        gt_boxes = _read_boxes(gt_label_dir / f"{image_path.stem}.txt", width, height)
        pred_boxes = _read_boxes(pred_labels / f"{image_path.stem}.txt", width, height)

        target_cls = _classes_array(gt_boxes)
        pred_cls = _classes_array(pred_boxes)
        conf = _confidence_array(pred_boxes)
        if not gt_boxes or not pred_boxes:
            tp = np.zeros((len(pred_boxes), ULTRALYTICS_IOU_THRESHOLDS.shape[0]), dtype=bool)
        else:
            iou = _to_numpy(backend.probiou(_boxes_to_xywhr(gt_boxes), _boxes_to_xywhr(pred_boxes)))
            tp = _match_predictions(pred_classes=pred_cls, true_classes=target_cls, iou=iou)

        no_pred = len(pred_boxes) == 0
        metrics.update_stats(
            {
                "tp": tp,
                "target_cls": target_cls,
                "target_img": np.unique(target_cls),
                "conf": np.zeros(0, dtype=np.float32) if no_pred else conf,
                "pred_cls": np.zeros(0, dtype=np.float32) if no_pred else pred_cls,
                "im_name": image_path.name,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.process(save_dir=output_dir, plot=plot)
    metrics.save_dir = output_dir
    return metrics
