from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Set, Union

from .config import IMAGE_EXTENSIONS


SPLITS = ("train", "val", "test")


def _copy_images(source_dir: Path, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, Path] = {}
    for image in sorted(source_dir.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            target = output_dir / image.name
            shutil.copy2(image, target)
            copied[image.stem] = image
    return copied


def _filter_label_file(
    source_label: Path,
    output_label: Path,
    keep_classes: Set[int],
) -> Dict[str, int]:
    kept = []
    removed = 0
    total = 0

    for line_no, raw in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 9:
            raise ValueError(f"{source_label}:{line_no}: expected 9 YOLO-OBB fields")
        cls = int(parts[0])
        total += 1
        if cls in keep_classes:
            kept.append(line)
        else:
            removed += 1

    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_label.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return {"total": total, "kept": len(kept), "removed": removed}


def _write_data_yaml(output: Path, names: Mapping[int, str]) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_label_subset_dataset(
    source: Union[str, Path],
    output: Union[str, Path],
    keep_classes: Set[int],
    names: Mapping[int, str],
) -> Dict[str, Dict[str, int]]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if sorted(keep_classes) != sorted(names.keys()):
        raise ValueError("keep_classes must match the class ids in names")

    report: Dict[str, Dict[str, int]] = {}
    manifest_rows = []

    for split in SPLITS:
        source_images = source / "images" / split
        source_labels = source / "labels" / split
        output_images = output / "images" / split
        output_labels = output / "labels" / split

        if not source_images.exists():
            raise FileNotFoundError(f"Missing image split directory: {source_images}")
        if not source_labels.exists():
            raise FileNotFoundError(f"Missing label split directory: {source_labels}")

        copied_images = _copy_images(source_images, output_images)
        split_report: MutableMapping[str, int] = {
            "images": len(copied_images),
            "labels": 0,
            "total_objects": 0,
            "kept_objects": 0,
            "removed_objects": 0,
            "empty_labels": 0,
        }

        for stem, source_image in copied_images.items():
            source_label = source_labels / f"{stem}.txt"
            if not source_label.exists():
                raise FileNotFoundError(f"Missing label for image: {source_image}")
            output_label = output_labels / f"{stem}.txt"
            counts = _filter_label_file(source_label, output_label, keep_classes)
            split_report["labels"] += 1
            split_report["total_objects"] += counts["total"]
            split_report["kept_objects"] += counts["kept"]
            split_report["removed_objects"] += counts["removed"]
            if counts["kept"] == 0:
                split_report["empty_labels"] += 1
            manifest_rows.append(
                {
                    "split": split,
                    "image": f"{stem}{source_image.suffix}",
                    "label": f"{stem}.txt",
                    "source_image": str(source_image),
                    "source_label": str(source_label),
                    "kept_objects": counts["kept"],
                    "removed_objects": counts["removed"],
                }
            )

        report[split] = dict(split_report)

    _write_data_yaml(output, names)

    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "split",
            "image",
            "label",
            "source_image",
            "source_label",
            "kept_objects",
            "removed_objects",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    lines = [
        f"source: {source}",
        f"output: {output}",
        f"keep_classes: {sorted(keep_classes)}",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    for split in SPLITS:
        lines.append(f"{split}: {report[split]}")
    (output / "subset_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
