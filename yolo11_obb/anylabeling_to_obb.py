from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


LABEL_TO_CLASS = {
    "label1": 0,
    "label2": 1,
    "label3": 2,
    "label4": 3,
    "label5": 4,
    "label6": 5,
    "label7": 6,
    "lable7": 7,
    "other": 8,
}


@dataclass(frozen=True)
class ConversionReport:
    json_files: int
    labels_written: int
    objects: int
    images_copied: int


def format_obb_number(value: float) -> str:
    if abs(value) < 0.0000005:
        value = 0.0
    return f"{value:.6f}".rstrip("0").rstrip(".")


def shape_to_obb_line(
    shape: Mapping[str, Any],
    image_width: float,
    image_height: float,
    label_to_class: Mapping[str, int] = LABEL_TO_CLASS,
    clip: bool = True,
) -> str:
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size: {image_width}x{image_height}")

    label = shape.get("label")
    if label not in label_to_class:
        raise ValueError(f"unknown label: {label!r}")

    points = shape.get("points")
    if not isinstance(points, Sequence) or len(points) != 4:
        raise ValueError(f"shape {label!r} must contain exactly 4 points")

    values: List[str] = [str(label_to_class[label])]
    for point in points:
        if not isinstance(point, Sequence) or len(point) < 2:
            raise ValueError(f"invalid point in shape {label!r}: {point!r}")
        x = float(point[0]) / image_width
        y = float(point[1]) / image_height
        if clip:
            x = min(max(x, 0.0), 1.0)
            y = min(max(y, 0.0), 1.0)
        values.extend([format_obb_number(x), format_obb_number(y)])

    return " ".join(values)


def load_annotation(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"annotation must be a JSON object: {json_path}")
    return data


def annotation_to_obb_lines(
    annotation: Mapping[str, Any],
    label_to_class: Mapping[str, int] = LABEL_TO_CLASS,
    clip: bool = True,
) -> List[str]:
    try:
        image_width = float(annotation["imageWidth"])
        image_height = float(annotation["imageHeight"])
    except KeyError as exc:
        raise ValueError(f"missing image dimension: {exc.args[0]}") from exc

    shapes = annotation.get("shapes", [])
    if not isinstance(shapes, Iterable):
        raise ValueError("annotation field 'shapes' must be iterable")

    return [
        shape_to_obb_line(shape, image_width, image_height, label_to_class, clip=clip)
        for shape in shapes
    ]


def write_obb_label(json_path: Path, output_txt: Path, clip: bool = True) -> int:
    annotation = load_annotation(json_path)
    lines = annotation_to_obb_lines(annotation, clip=clip)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def resolve_image_path(source: Path, json_path: Path, annotation: Mapping[str, Any]) -> Path:
    image_path = annotation.get("imagePath")
    candidates = []
    if image_path:
        raw_path = Path(str(image_path))
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(source / raw_path)
            candidates.append(source / raw_path.name)
    candidates.append(json_path.with_suffix(".bmp"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"missing image for {json_path}")


def convert_anylabeling_directory(
    source: Path,
    output: Path,
    copy_images: bool = True,
    overwrite: bool = False,
    clip: bool = True,
) -> ConversionReport:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")

    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    json_paths = sorted(source.glob("*.json"))
    objects = 0
    images_copied = 0

    for json_path in json_paths:
        annotation = load_annotation(json_path)
        lines = annotation_to_obb_lines(annotation, clip=clip)
        output_txt = output / f"{json_path.stem}.txt"
        output_txt.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        objects += len(lines)

        if copy_images:
            image_source = resolve_image_path(source, json_path, annotation)
            shutil.copy2(image_source, output / image_source.name)
            images_copied += 1

    return ConversionReport(
        json_files=len(json_paths),
        labels_written=len(json_paths),
        objects=objects,
        images_copied=images_copied,
    )
