from __future__ import annotations

import csv
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

import cv2

from .config import DatasetConfig, load_dataset_config
from .obb_geometry import parse_obb_line


@dataclass(frozen=True)
class RhinoDatasetReport:
    output: Path
    images_by_split: Dict[str, int]
    objects_by_split: Dict[str, int]
    image_mode: str


def _label_dir(image_dir: Path) -> Path:
    return image_dir.parent.parent / "labels" / image_dir.name


def _image_paths(image_dir: Path) -> Iterable[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file())


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if mode not in {"link", "copy"}:
        raise ValueError("image_mode must be 'link' or 'copy'")
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _dota_lines(label_path: Path, width: int, height: int, names: Dict[int, str]) -> List[str]:
    if not label_path.exists():
        return []
    converted: List[str] = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            box = parse_obb_line(line, image_width=width, image_height=height)
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no}: {exc}") from exc
        if box.class_id not in names:
            raise ValueError(f"{label_path}:{line_no}: unknown class id {box.class_id}")
        coords = " ".join(f"{value:.4f}" for point in box.points for value in point)
        converted.append(f"{coords} {names[box.class_id]} 0")
    return converted


def _prepare_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"RHINO dataset output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _convert_split(dataset: DatasetConfig, split: str, output: Path, image_mode: str) -> tuple[int, int, List[dict]]:
    image_dir = dataset.splits[split]
    label_dir = _label_dir(image_dir)
    destination_images = output / split / "images"
    destination_labels = output / split / "annfiles"
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0
    manifest_rows: List[dict] = []
    for image_path in _image_paths(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        label_path = label_dir / f"{image_path.stem}.txt"
        lines = _dota_lines(label_path, width, height, dataset.names)
        _link_or_copy(image_path, destination_images / image_path.name, image_mode)
        (destination_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        image_count += 1
        object_count += len(lines)
        manifest_rows.append(
            {
                "split": split,
                "image": image_path.name,
                "source_image": str(image_path),
                "source_label": str(label_path),
                "objects": len(lines),
            }
        )
    return image_count, object_count, manifest_rows


def create_rhino_dataset(
    source_data: Union[str, Path],
    output: Union[str, Path],
    image_mode: str = "link",
) -> RhinoDatasetReport:
    """Convert the current YOLO-OBB train/test split to RHINO's DOTA annfile layout."""
    dataset = load_dataset_config(source_data)
    output_path = Path(output).expanduser().resolve()
    _prepare_output(output_path)

    images_by_split: Dict[str, int] = {}
    objects_by_split: Dict[str, int] = {}
    manifest_rows: List[dict] = []
    for split in ("train", "test"):
        images, objects, rows = _convert_split(dataset, split, output_path, image_mode)
        images_by_split[split] = images
        objects_by_split[split] = objects
        manifest_rows.extend(rows)

    (output_path / "classes.txt").write_text(
        "\n".join(dataset.names[index] for index in sorted(dataset.names)) + "\n",
        encoding="utf-8",
    )
    with (output_path / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "image", "source_image", "source_label", "objects"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    (output_path / "README.txt").write_text(
        "\n".join(
            [
                "format: DOTA-style quadrilateral annfiles for RHINO/MMRotate",
                f"source_data: {dataset.data_yaml}",
                f"image_mode: {image_mode}",
                f"train_images: {images_by_split['train']}",
                f"test_images: {images_by_split['test']}",
                f"train_objects: {objects_by_split['train']}",
                f"test_objects: {objects_by_split['test']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return RhinoDatasetReport(
        output=output_path,
        images_by_split=images_by_split,
        objects_by_split=objects_by_split,
        image_mode=image_mode,
    )
