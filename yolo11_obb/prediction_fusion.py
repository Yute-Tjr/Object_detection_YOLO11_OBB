from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

from .anylabeling_to_obb import format_obb_number
from .deskew_dataset import _rotation_matrix
from .obb_geometry import ObbBox, parse_obb_line


Point = Tuple[float, float]


@dataclass(frozen=True)
class DeskewTransform:
    image: str
    label: str
    angle_degrees: float
    source_width: int
    source_height: int
    output_width: int
    output_height: int


def _transform_point(matrix: np.ndarray, point: Point) -> Point:
    x, y = point
    return (
        float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
        float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
    )


def inverse_deskew_box(
    box: ObbBox,
    source_width: int,
    source_height: int,
    deskew_width: int,
    deskew_height: int,
    deskew_angle_degrees: float,
) -> ObbBox:
    matrix, _, _ = _rotation_matrix(source_width, source_height, deskew_angle_degrees)
    inverse = cv2.invertAffineTransform(matrix)
    return ObbBox(
        class_id=box.class_id,
        points=tuple(_transform_point(inverse, point) for point in box.points),  # type: ignore[arg-type]
        confidence=box.confidence,
    )


def write_obb_prediction_line(
    box: ObbBox,
    image_width: float = 1.0,
    image_height: float = 1.0,
) -> str:
    values = [str(box.class_id)]
    for x, y in box.points:
        values.extend(
            [
                format_obb_number(min(max(x / image_width, 0.0), 1.0)),
                format_obb_number(min(max(y / image_height, 0.0), 1.0)),
            ]
        )
    if box.confidence is not None:
        values.append(format_obb_number(box.confidence))
    return " ".join(values)


def read_prediction_boxes(label_path: Path, image_width: int, image_height: int) -> List[ObbBox]:
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


def load_deskew_transforms(report_path: Union[str, Path], split: Optional[str] = "test") -> Dict[str, DeskewTransform]:
    report_path = Path(report_path)
    transforms: Dict[str, DeskewTransform] = {}
    with report_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if split is not None and row.get("split") != split:
                continue
            label = row["label"]
            transforms[Path(label).stem] = DeskewTransform(
                image=row["image"],
                label=label,
                angle_degrees=float(row["deskew_angle_degrees"]),
                source_width=int(row["source_width"]),
                source_height=int(row["source_height"]),
                output_width=int(row["output_width"]),
                output_height=int(row["output_height"]),
            )
    return transforms


def _selected(boxes: Sequence[ObbBox], class_ids: Set[int]) -> List[ObbBox]:
    return [box for box in boxes if box.class_id in class_ids]


def fuse_prediction_labels(
    baseline_labels: Union[str, Path],
    deskew_labels: Union[str, Path],
    deskew_report: Union[str, Path],
    output_labels: Union[str, Path],
    baseline_class_ids: Set[int],
    deskew_class_ids: Set[int],
    split: str = "test",
) -> List[Dict[str, object]]:
    baseline_labels = Path(baseline_labels)
    deskew_labels = Path(deskew_labels)
    output_labels = Path(output_labels)
    transforms = load_deskew_transforms(deskew_report, split=split)
    if not transforms:
        raise ValueError(f"No deskew rows found for split={split!r} in {deskew_report}")

    output_labels.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for stem, transform in sorted(transforms.items()):
        baseline_boxes = read_prediction_boxes(
            baseline_labels / f"{stem}.txt",
            image_width=transform.source_width,
            image_height=transform.source_height,
        )
        deskew_boxes = read_prediction_boxes(
            deskew_labels / f"{stem}.txt",
            image_width=transform.output_width,
            image_height=transform.output_height,
        )
        restored_deskew_boxes = [
            inverse_deskew_box(
                box,
                source_width=transform.source_width,
                source_height=transform.source_height,
                deskew_width=transform.output_width,
                deskew_height=transform.output_height,
                deskew_angle_degrees=transform.angle_degrees,
            )
            for box in deskew_boxes
        ]
        fused = _selected(baseline_boxes, baseline_class_ids) + _selected(restored_deskew_boxes, deskew_class_ids)
        fused.sort(key=lambda box: (box.class_id, -(box.confidence or 0.0)))

        output_file = output_labels / f"{stem}.txt"
        output_file.write_text(
            "\n".join(
                write_obb_prediction_line(
                    box,
                    image_width=transform.source_width,
                    image_height=transform.source_height,
                )
                for box in fused
            )
            + ("\n" if fused else ""),
            encoding="utf-8",
        )
        rows.append(
            {
                "image": transform.image,
                "label": transform.label,
                "baseline_boxes": len(baseline_boxes),
                "deskew_boxes": len(deskew_boxes),
                "fused_boxes": len(fused),
            }
        )

    return rows


def write_fusion_manifest(rows: Sequence[Dict[str, object]], output_path: Union[str, Path]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image", "label", "baseline_boxes", "deskew_boxes", "fused_boxes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path
