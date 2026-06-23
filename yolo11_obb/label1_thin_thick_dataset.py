from __future__ import annotations

import csv
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple, Union

import cv2

from .config import IMAGE_EXTENSIONS
from .obb_geometry import parse_obb_line, top_edge_width


LABEL1_THIN_THICK_NAMES = {
    0: "label1_thin",
    1: "label1_thick",
    2: "label2",
    3: "label3",
    4: "label4",
    5: "label5",
    6: "label6",
}


def parent_group_key(stem: str) -> str:
    match = re.match(r"^(?P<parent>.+)-\d+$", stem)
    return match.group("parent") if match else stem


def _portable_dataset_path(output: Path) -> str:
    if output.parent.name == "datasets":
        return f"datasets/{output.name}"
    return "."


def _write_data_yaml(output: Path, names: Mapping[int, str]) -> None:
    lines = [
        f"path: {_portable_dataset_path(output)}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        "names:",
    ]
    for class_id in sorted(names):
        lines.append(f"  {class_id}: {names[class_id]}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_grouped_images(source: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for image in sorted(source.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            groups.setdefault(parent_group_key(image.stem), []).append(image)
    if not groups:
        raise ValueError(f"no images found in source directory: {source}")
    return groups


def _split_group_keys(
    group_keys: Sequence[str],
    train_ratio: float,
    seed: int,
) -> Tuple[Set[str], Set[str]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    shuffled = list(group_keys)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    if len(shuffled) > 1:
        train_count = min(max(train_count, 1), len(shuffled) - 1)
    return set(shuffled[:train_count]), set(shuffled[train_count:])


def _image_size(image_path: Path) -> Tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    height, width = image.shape[:2]
    return width, height


def label1_top_edge_width_px(label_line: str, image_width: int, image_height: int) -> float:
    box = parse_obb_line(label_line, image_width=image_width, image_height=image_height)
    if box.class_id != 0:
        raise ValueError("label1_top_edge_width_px requires original class 0")
    return top_edge_width(box.points)


def _remap_label_line(
    raw_line: str,
    image_width: int,
    image_height: int,
    top_edge_threshold_px: float,
) -> Tuple[str, int, float]:
    parts = raw_line.split()
    if len(parts) != 9:
        raise ValueError(f"expected 9 YOLO-OBB fields, got {len(parts)}")

    original_class = int(parts[0])
    if original_class == 0:
        width = label1_top_edge_width_px(raw_line, image_width, image_height)
        new_class = 0 if width < top_edge_threshold_px else 1
    elif 1 <= original_class <= 5:
        width = 0.0
        new_class = original_class + 1
    else:
        width = 0.0
        new_class = -1

    return " ".join([str(new_class)] + parts[1:]), new_class, width


def _copy_image(source_image: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, output_dir / source_image.name)


def _remap_label_file(
    source_label: Path,
    output_label: Path,
    image_width: int,
    image_height: int,
    top_edge_threshold_px: float,
) -> Tuple[Dict[str, int], List[Dict[str, str]]]:
    kept: List[str] = []
    width_rows: List[Dict[str, str]] = []
    counts: Dict[str, int] = {
        "total": 0,
        "kept": 0,
        "removed": 0,
    }
    class_counts = {class_id: 0 for class_id in LABEL1_THIN_THICK_NAMES}

    for line_no, raw in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        counts["total"] += 1
        remapped, new_class, width = _remap_label_line(
            line,
            image_width=image_width,
            image_height=image_height,
            top_edge_threshold_px=top_edge_threshold_px,
        )
        if new_class == -1:
            counts["removed"] += 1
            continue
        if new_class not in class_counts:
            raise ValueError(f"{source_label}:{line_no}: unsupported class id {new_class}")
        kept.append(remapped)
        counts["kept"] += 1
        class_counts[new_class] += 1
        if int(line.split()[0]) == 0:
            width_rows.append(
                {
                    "label": source_label.name,
                    "line": str(line_no),
                    "top_edge_width_px": f"{width:.6f}",
                    "new_class": str(new_class),
                    "new_name": LABEL1_THIN_THICK_NAMES[new_class],
                }
            )

    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    for class_id, count in class_counts.items():
        counts[f"class_{class_id}"] = count
    return counts, width_rows


def create_label1_thin_thick_dataset(
    source: Union[str, Path],
    output: Union[str, Path],
    top_edge_threshold_px: float,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, object]]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    if top_edge_threshold_px <= 0:
        raise ValueError("top_edge_threshold_px must be positive")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source}")
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")

    groups = _collect_grouped_images(source)
    train_groups, test_groups = _split_group_keys(sorted(groups), train_ratio, seed)
    split_groups = {"train": train_groups, "test": test_groups}

    report: Dict[str, Dict[str, object]] = {}
    manifest_rows: List[Dict[str, object]] = []
    width_rows: List[Dict[str, str]] = []

    for split in ("train", "test"):
        split_report: MutableMapping[str, object] = {
            "groups": len(split_groups[split]),
            "images": 0,
            "labels": 0,
            "total_objects": 0,
            "kept_objects": 0,
            "removed_objects": 0,
            "empty_labels": 0,
            "class_counts": {class_id: 0 for class_id in LABEL1_THIN_THICK_NAMES},
        }
        class_counts = split_report["class_counts"]
        if not isinstance(class_counts, dict):
            raise TypeError("class_counts must be a dict")

        for group_key in sorted(split_groups[split]):
            for source_image in sorted(groups[group_key]):
                source_label = source / f"{source_image.stem}.txt"
                if not source_label.exists():
                    raise FileNotFoundError(f"Missing label for image: {source_image}")

                image_width, image_height = _image_size(source_image)
                output_image_dir = output / "images" / split
                output_label = output / "labels" / split / f"{source_image.stem}.txt"
                _copy_image(source_image, output_image_dir)
                counts, image_width_rows = _remap_label_file(
                    source_label,
                    output_label,
                    image_width=image_width,
                    image_height=image_height,
                    top_edge_threshold_px=top_edge_threshold_px,
                )

                split_report["images"] = int(split_report["images"]) + 1
                split_report["labels"] = int(split_report["labels"]) + 1
                split_report["total_objects"] = int(split_report["total_objects"]) + counts["total"]
                split_report["kept_objects"] = int(split_report["kept_objects"]) + counts["kept"]
                split_report["removed_objects"] = int(split_report["removed_objects"]) + counts["removed"]
                if counts["kept"] == 0:
                    split_report["empty_labels"] = int(split_report["empty_labels"]) + 1
                for class_id in LABEL1_THIN_THICK_NAMES:
                    class_counts[class_id] += counts[f"class_{class_id}"]

                for row in image_width_rows:
                    row["split"] = split
                    row["image"] = source_image.name
                    width_rows.append(row)

                manifest_rows.append(
                    {
                        "split": split,
                        "group": group_key,
                        "image": source_image.name,
                        "label": f"{source_image.stem}.txt",
                        "source_image": str(source_image),
                        "source_label": str(source_label),
                        "kept_objects": counts["kept"],
                        "removed_objects": counts["removed"],
                    }
                )
        report[split] = dict(split_report)

    _write_data_yaml(output, LABEL1_THIN_THICK_NAMES)
    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "group",
                "image",
                "label",
                "source_image",
                "source_label",
                "kept_objects",
                "removed_objects",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (output / "label1_top_edge_widths.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "image", "label", "line", "top_edge_width_px", "new_class", "new_name"],
        )
        writer.writeheader()
        writer.writerows(width_rows)

    lines = [
        f"source: {source}",
        f"output: {output}",
        f"seed: {seed}",
        f"train_ratio: {train_ratio}",
        f"top_edge_threshold_px: {top_edge_threshold_px}",
        "names:",
    ]
    for class_id in sorted(LABEL1_THIN_THICK_NAMES):
        lines.append(f"  {class_id}: {LABEL1_THIN_THICK_NAMES[class_id]}")
    for split in ("train", "test"):
        lines.append(f"{split}: {report[split]}")
    (output / "split_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
