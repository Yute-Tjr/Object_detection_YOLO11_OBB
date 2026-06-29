from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2

from .classification_labels import LabelSample, split_samples_by_group, write_manifest_csv
from .obb_crop import save_obb_crop


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"annotation must be an object: {path}")
    return data


def _resolve_image(source: Path, json_path: Path, data: Mapping[str, object]) -> Path:
    image_path = data.get("imagePath")
    candidates = []
    if image_path:
        raw = Path(str(image_path))
        candidates.append(raw if raw.is_absolute() else source / raw)
        candidates.append(source / raw.name)
    candidates.append(json_path.with_suffix(".bmp"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing image for {json_path}")


def _shape_points(data: Mapping[str, object], label_name: str, json_path: Path) -> Sequence[Sequence[float]]:
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"shapes must be a list: {json_path}")
    for shape in shapes:
        if isinstance(shape, dict) and shape.get("label") == label_name:
            points = shape.get("points")
            if not isinstance(points, list):
                raise ValueError(f"missing points for {label_name}: {json_path}")
            return points
    raise ValueError(f"missing shape {label_name}: {json_path}")


def build_classification_dataset(
    source: Path,
    output: Path,
    samples: Sequence[LabelSample],
    label_name: str,
    train_ratio: float,
    seed: int,
    overwrite: bool,
) -> Dict[str, object]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    split_samples = split_samples_by_group(samples, train_ratio=train_ratio, seed=seed)
    manifest_rows = []
    class_counts: Counter[str] = Counter()
    split_counts: Dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}

    for split, split_rows in split_samples.items():
        for sample in split_rows:
            json_path = source / f"{sample.image_name}.json"
            if not json_path.exists():
                raise FileNotFoundError(f"missing annotation: {json_path}")
            data = _load_json(json_path)
            image_path = _resolve_image(source, json_path, data)
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"failed to read image: {image_path}")
            points = _shape_points(data, label_name, json_path)
            crop_rel = Path("images") / split / sample.class_name / f"{sample.image_name}.png"
            crop_path = output / crop_rel
            save_obb_crop(image, points, crop_path)
            class_counts[sample.class_name] += 1
            split_counts[split][sample.class_name] += 1
            manifest_rows.append(
                {
                    "image_name": sample.image_name,
                    "class_name": sample.class_name,
                    "group": sample.group,
                    "split": split,
                    "source_json": str(json_path),
                    "source_image": str(image_path),
                    "crop_path": str(crop_rel),
                }
            )

    write_manifest_csv(output / "manifest.csv", manifest_rows)
    lines = [
        f"source: {source}",
        f"output: {output}",
        f"label_name: {label_name}",
        f"train_ratio: {train_ratio}",
        f"seed: {seed}",
        f"total_samples: {len(manifest_rows)}",
        f"classes: {dict(sorted(class_counts.items()))}",
        f"train: {dict(sorted(split_counts['train'].items()))}",
        f"test: {dict(sorted(split_counts['test'].items()))}",
    ]
    (output / "split_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "total_samples": len(manifest_rows),
        "classes": dict(class_counts),
        "train": dict(split_counts["train"]),
        "test": dict(split_counts["test"]),
    }
