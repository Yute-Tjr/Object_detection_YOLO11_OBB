from __future__ import annotations

import csv
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple, Union

from .config import IMAGE_EXTENSIONS


def parent_group_key(stem: str) -> str:
    match = re.match(r"^(?P<parent>.+)-\d+$", stem)
    return match.group("parent") if match else stem


def _copy_image(source_image: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, output_dir / source_image.name)


def _filter_label_file(
    source_label: Path,
    output_label: Path,
    keep_classes: Set[int],
) -> Dict[str, int]:
    kept: List[str] = []
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
        f"path: {output.resolve()}",
        "train: images/train",
        "val: images/test",
        "test: images/test",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    (output / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    train_groups = set(shuffled[:train_count])
    test_groups = set(shuffled[train_count:])
    return train_groups, test_groups


def _collect_grouped_images(source: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for image in sorted(source.iterdir()):
        if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
            groups.setdefault(parent_group_key(image.stem), []).append(image)
    if not groups:
        raise ValueError(f"no images found in source directory: {source}")
    return groups


def create_train_test_dataset(
    source: Union[str, Path],
    output: Union[str, Path],
    keep_classes: Set[int],
    names: Mapping[int, str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset directory not found: {source}")
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}")
    if sorted(keep_classes) != sorted(names.keys()):
        raise ValueError("keep_classes must match the class ids in names")

    groups = _collect_grouped_images(source)
    train_groups, test_groups = _split_group_keys(sorted(groups), train_ratio, seed)
    split_groups = {"train": train_groups, "test": test_groups}

    report: Dict[str, Dict[str, int]] = {}
    manifest_rows = []
    split_image_paths: Dict[str, List[Path]] = {"train": [], "test": []}

    for split in ("train", "test"):
        split_report: MutableMapping[str, int] = {
            "groups": len(split_groups[split]),
            "images": 0,
            "labels": 0,
            "total_objects": 0,
            "kept_objects": 0,
            "removed_objects": 0,
            "empty_labels": 0,
        }
        for group_key in sorted(split_groups[split]):
            for source_image in sorted(groups[group_key]):
                source_label = source / f"{source_image.stem}.txt"
                if not source_label.exists():
                    raise FileNotFoundError(f"Missing label for image: {source_image}")

                output_image_dir = output / "images" / split
                output_label = output / "labels" / split / f"{source_image.stem}.txt"
                _copy_image(source_image, output_image_dir)
                counts = _filter_label_file(source_label, output_label, keep_classes)

                output_image = output_image_dir / source_image.name
                split_image_paths[split].append(output_image.resolve())
                split_report["images"] += 1
                split_report["labels"] += 1
                split_report["total_objects"] += counts["total"]
                split_report["kept_objects"] += counts["kept"]
                split_report["removed_objects"] += counts["removed"]
                if counts["kept"] == 0:
                    split_report["empty_labels"] += 1
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

    _write_data_yaml(output, names)
    (output / "splits").mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        lines = [str(path) for path in sorted(split_image_paths[split])]
        (output / "splits" / f"{split}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )

    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "split",
            "group",
            "image",
            "label",
            "source_image",
            "source_label",
            "kept_objects",
            "removed_objects",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    lines = [
        f"source: {source}",
        f"output: {output}",
        f"seed: {seed}",
        f"train_ratio: {train_ratio}",
        f"keep_classes: {sorted(keep_classes)}",
        "names:",
    ]
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    for split in ("train", "test"):
        lines.append(f"{split}: {report[split]}")
    (output / "split_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report
