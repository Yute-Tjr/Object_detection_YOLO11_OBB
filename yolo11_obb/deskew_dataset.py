from __future__ import annotations

import csv
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from .anylabeling_to_obb import format_obb_number
from .config import IMAGE_EXTENSIONS, DatasetConfig, load_dataset_config
from .obb_geometry import ObbBox, parse_obb_line


Point = Tuple[float, float]


@dataclass(frozen=True)
class RotatedBoxes:
    boxes: List[ObbBox]
    width: int
    height: int


def _normalize_half_turn_angle(angle_degrees: float) -> float:
    return ((angle_degrees + 90.0) % 180.0) - 90.0


def signed_top_edge_angle(points: Sequence[Point]) -> float:
    if len(points) != 4:
        raise ValueError(f"expected 4 OBB points, got {len(points)}")
    top_edge_index = min(
        range(4),
        key=lambda idx: (points[idx][1] + points[(idx + 1) % 4][1]) / 2.0,
    )
    start = points[top_edge_index]
    end = points[(top_edge_index + 1) % 4]
    angle = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    return _normalize_half_turn_angle(angle)


def estimate_deskew_angle(
    boxes: Sequence[ObbBox],
    angle_class_ids: Optional[Set[int]] = None,
) -> float:
    selected = [
        box
        for box in boxes
        if angle_class_ids is None or box.class_id in angle_class_ids
    ]
    if not selected:
        selected = list(boxes)
    if not selected:
        return 0.0
    return statistics.median(signed_top_edge_angle(box.points) for box in selected)


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


def _rotation_matrix(width: int, height: int, angle_degrees: float) -> Tuple[np.ndarray, int, int]:
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(float(matrix[0, 0]))
    sin = abs(float(matrix[0, 1]))
    output_width = int(round(height * sin + width * cos))
    output_height = int(round(height * cos + width * sin))
    matrix[0, 2] += output_width / 2.0 - center[0]
    matrix[1, 2] += output_height / 2.0 - center[1]
    return matrix, output_width, output_height


def _transform_point(matrix: np.ndarray, point: Point) -> Point:
    x, y = point
    return (
        float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
        float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
    )


def rotate_image_and_boxes(
    image: np.ndarray,
    boxes: Sequence[ObbBox],
    angle_degrees: float,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, RotatedBoxes]:
    height, width = image.shape[:2]
    matrix, output_width, output_height = _rotation_matrix(width, height, angle_degrees)
    rotated_image = cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderValue=border_value,
    )
    rotated_boxes = [
        ObbBox(
            class_id=box.class_id,
            points=tuple(_transform_point(matrix, point) for point in box.points),  # type: ignore[arg-type]
            confidence=box.confidence,
        )
        for box in boxes
    ]
    return rotated_image, RotatedBoxes(rotated_boxes, output_width, output_height)


def _format_box_line(box: ObbBox, image_width: int, image_height: int) -> str:
    values = [str(box.class_id)]
    for x, y in box.points:
        normalized_x = min(max(x / image_width, 0.0), 1.0)
        normalized_y = min(max(y / image_height, 0.0), 1.0)
        values.extend([format_obb_number(normalized_x), format_obb_number(normalized_y)])
    return " ".join(values)


def _portable_dataset_path(output: Path) -> str:
    if output.parent.name == "datasets":
        return f"datasets/{output.name}"
    return "."


def _write_data_yaml(
    output: Path,
    names: Mapping[int, str],
    split_targets: Mapping[str, str],
) -> None:
    lines = [
        f"path: {_portable_dataset_path(output)}",
        f"train: images/{split_targets['train']}",
        f"val: images/{split_targets['val']}",
        f"test: images/{split_targets['test']}",
        "names:",
    ]
    for class_id in sorted(names):
        lines.append(f"  {class_id}: {names[class_id]}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _split_targets(dataset: DatasetConfig) -> Dict[str, str]:
    path_to_output: Dict[Path, str] = {}
    targets: Dict[str, str] = {}
    for split in ("train", "test", "val"):
        split_path = dataset.splits[split].resolve()
        if split_path not in path_to_output:
            path_to_output[split_path] = split
        targets[split] = path_to_output[split_path]
    return targets


def create_deskewed_obb_dataset(
    source_data: Union[str, Path],
    output: Union[str, Path],
    angle_class_ids: Optional[Set[int]] = None,
    border_value: Tuple[int, int, int] = (0, 0, 0),
) -> Dict[str, Dict[str, object]]:
    source_data = Path(source_data).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")

    dataset = load_dataset_config(source_data)
    split_targets = _split_targets(dataset)
    output.mkdir(parents=True)

    report: Dict[str, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    processed_output_splits: Set[str] = set()

    for split in ("train", "test", "val"):
        output_split = split_targets[split]
        if output_split in processed_output_splits:
            continue
        processed_output_splits.add(output_split)

        image_dir = dataset.splits[split]
        label_dir = _label_dir_for_image_dir(image_dir)
        split_report = {
            "images": 0,
            "labels": 0,
            "objects": 0,
            "mean_abs_angle": 0.0,
        }
        angles: List[float] = []

        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for image: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"failed to read image: {image_path}")
            height, width = image.shape[:2]
            boxes = _read_boxes(label_path, width, height)
            angle = estimate_deskew_angle(boxes, angle_class_ids=angle_class_ids)
            rotated_image, rotated = rotate_image_and_boxes(
                image,
                boxes,
                angle_degrees=angle,
                border_value=border_value,
            )

            output_image = output / "images" / output_split / image_path.name
            output_label = output / "labels" / output_split / f"{image_path.stem}.txt"
            output_image.parent.mkdir(parents=True, exist_ok=True)
            output_label.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_image), rotated_image):
                raise RuntimeError(f"failed to write image: {output_image}")
            output_label.write_text(
                "\n".join(_format_box_line(box, rotated.width, rotated.height) for box in rotated.boxes)
                + ("\n" if rotated.boxes else ""),
                encoding="utf-8",
            )

            split_report["images"] = int(split_report["images"]) + 1
            split_report["labels"] = int(split_report["labels"]) + 1
            split_report["objects"] = int(split_report["objects"]) + len(rotated.boxes)
            angles.append(abs(angle))
            rows.append(
                {
                    "split": output_split,
                    "image": image_path.name,
                    "label": f"{image_path.stem}.txt",
                    "deskew_angle_degrees": f"{angle:.6f}",
                    "objects": len(rotated.boxes),
                    "source_width": width,
                    "source_height": height,
                    "output_width": rotated.width,
                    "output_height": rotated.height,
                }
            )

        if angles:
            split_report["mean_abs_angle"] = sum(angles) / len(angles)
        report[output_split] = split_report

    _write_data_yaml(output, dataset.names, split_targets)

    with (output / "deskew_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "image",
                "label",
                "deskew_angle_degrees",
                "objects",
                "source_width",
                "source_height",
                "output_width",
                "output_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"source_data: {source_data}",
        f"output: {output}",
        "angle_class_ids: "
        + (",".join(str(class_id) for class_id in sorted(angle_class_ids)) if angle_class_ids else "all"),
    ]
    for split in sorted(report):
        lines.append(f"{split}: {report[split]}")
    (output / "deskew_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    shutil.copy2(source_data, output / "source_data.yaml")
    return report
