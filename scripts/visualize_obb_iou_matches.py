#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import IMAGE_EXTENSIONS, load_dataset_config, resolve_from_root
from yolo11_obb.obb_geometry import ObbBox, match_ground_truths, parse_obb_line


DEFAULT_DATA = ROOT / "datasets" / "154843_obb_converted_label1_6_train_test" / "data.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "analysis" / "iou85_overlay"

GT_COLOR = (255, 0, 0)
PRED_COLOR = (0, 255, 255)
PASS_COLOR = (0, 180, 0)
FAIL_COLOR = (0, 0, 255)
UNMATCHED_PRED_COLOR = (255, 0, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw GT and prediction OBBs together and annotate same-class IoU matches.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pred-labels", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of images to process.")
    return parser.parse_args()


def label_dir_for_image_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return image_dir.parent.parent / "labels" / image_dir.name


def image_files(image_dir: Path) -> Iterable[Path]:
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_boxes(label_path: Path, width: int, height: int) -> List[ObbBox]:
    if not label_path.exists():
        return []
    boxes = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            boxes.append(parse_obb_line(line, image_width=width, image_height=height))
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no}: {exc}") from exc
    return boxes


def draw_polygon(image: np.ndarray, box: ObbBox, color: Sequence[int], thickness: int) -> None:
    points = np.array(box.points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [points], isClosed=True, color=tuple(color), thickness=thickness)


def text_origin(box: ObbBox, row_offset: int = 0) -> tuple[int, int]:
    min_x = min(point[0] for point in box.points)
    min_y = min(point[1] for point in box.points)
    return max(int(min_x), 0), max(int(min_y) - 8 + row_offset * 18, 14)


def draw_text(image: np.ndarray, text: str, origin: tuple[int, int], color: Sequence[int]) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        tuple(color),
        1,
        cv2.LINE_AA,
    )


def class_name(names: Dict[int, str], class_id: int) -> str:
    return str(names.get(class_id, class_id))


def safe_dir_name(name: str) -> str:
    return name.replace("/", "_")


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def main() -> None:
    args = parse_args()
    data_yaml = resolve_from_root(args.data, ROOT)
    pred_labels = resolve_from_root(args.pred_labels, ROOT)
    output = resolve_from_root(args.output, ROOT)
    dataset = load_dataset_config(data_yaml)
    image_dir = dataset.splits[args.split]
    gt_label_dir = label_dir_for_image_dir(image_dir)

    if not pred_labels.is_dir():
        raise FileNotFoundError(f"Prediction labels directory not found: {pred_labels}")

    all_dir = output / "all"
    failed_dir = output / "failed_iou85"
    success_dir = output / "success_iou85"
    matches_csv = output / "matches.csv"
    output.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    processed = 0

    for image_path in image_files(image_dir):
        if args.limit and processed >= args.limit:
            break
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        gt_boxes = read_boxes(gt_label_dir / f"{image_path.stem}.txt", width, height)
        pred_boxes = read_boxes(pred_labels / f"{image_path.stem}.txt", width, height)
        matches = match_ground_truths(gt_boxes, pred_boxes, iou_threshold=args.iou_threshold)
        matched_predictions = {match.prediction for match in matches if match.prediction is not None}

        has_failed_match = False
        has_success_match = False
        failed_class_names = set()
        success_class_names = set()

        for match_index, match in enumerate(matches):
            draw_polygon(image, match.ground_truth, GT_COLOR, 2)
            if match.prediction is not None:
                draw_polygon(image, match.prediction, PRED_COLOR, 2)
            name = class_name(dataset.names, match.ground_truth.class_id)
            color = PASS_COLOR if match.passed else FAIL_COLOR
            confidence = "" if match.prediction is None or match.prediction.confidence is None else f" conf={match.prediction.confidence:.3f}"
            text = f"{name} IoU={match.iou:.3f}{confidence}"
            draw_text(image, text, text_origin(match.ground_truth, match_index % 3), color)

            if not match.passed:
                has_failed_match = True
                failed_class_names.add(name)
            else:
                has_success_match = True
                success_class_names.add(name)

            rows.append(
                {
                    "image": image_path.name,
                    "class": name,
                    "gt_index": match_index,
                    "pred_found": match.prediction is not None,
                    "iou": f"{match.iou:.6f}",
                    "passed_iou": match.passed,
                    "confidence": "" if match.prediction is None or match.prediction.confidence is None else f"{match.prediction.confidence:.6f}",
                }
            )

        for prediction in pred_boxes:
            if prediction in matched_predictions:
                continue
            draw_polygon(image, prediction, UNMATCHED_PRED_COLOR, 1)
            name = class_name(dataset.names, prediction.class_id)
            confidence = "" if prediction.confidence is None else f" conf={prediction.confidence:.3f}"
            draw_text(image, f"unmatched {name}{confidence}", text_origin(prediction), UNMATCHED_PRED_COLOR)

        output_name = f"{image_path.stem}.jpg"
        save_image(all_dir / output_name, image)
        if has_failed_match:
            save_image(failed_dir / output_name, image)
            for failed_class_name in failed_class_names:
                save_image(output / f"{safe_dir_name(failed_class_name)}_failed_iou85" / output_name, image)
        if has_success_match:
            save_image(success_dir / output_name, image)
            for success_class_name in success_class_names:
                save_image(output / f"{safe_dir_name(success_class_name)}_success_iou85" / output_name, image)
        processed += 1

    with matches_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image",
                "class",
                "gt_index",
                "pred_found",
                "iou",
                "passed_iou",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"images processed: {processed}")
    print(f"overlays: {all_dir}")
    print(f"failed overlays: {failed_dir}")
    print(f"success overlays: {success_dir}")
    print(f"matches: {matches_csv}")


if __name__ == "__main__":
    main()
