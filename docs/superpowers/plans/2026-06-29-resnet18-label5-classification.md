# ResNet18 Label5 Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first ResNet18 classification pipeline for `label5` `OK/NG` using manually annotated AnyLabeling OBB crops and the Excel description workbook as the label source.

**Architecture:** Add small focused modules for label parsing, OBB crop rectification, dataset construction, and classification training metrics. Keep CLIs thin: one script builds the crop dataset and one script trains/evaluates ResNet18. Use group-aware `8:2` splitting by parent image stem to avoid related child-index leakage.

**Tech Stack:** Python, unittest, OpenCV, NumPy, openpyxl for Excel reading, PyTorch/torchvision ResNet18 for training, PyYAML for argument snapshots, CSV/YAML/text outputs for reproducibility.

---

## File Structure

Create or modify these files:

| path | responsibility |
| --- | --- |
| `yolo11_obb/classification_labels.py` | Load and normalize Excel label rows, expose parent grouping and split helpers. |
| `tests/test_classification_labels.py` | Unit tests for status normalization, row parsing, group splitting, and CSV writing. |
| `yolo11_obb/obb_crop.py` | Rectify a four-point OBB polygon into an upright crop image. |
| `tests/test_obb_crop.py` | Unit tests for crop geometry and invalid polygons. |
| `yolo11_obb/classification_dataset.py` | Build the `label5` classification dataset folder and manifests from Excel rows plus AnyLabeling JSON/images. |
| `scripts/create_label_classification_dataset.py` | CLI wrapper for dataset construction. |
| `tests/test_classification_dataset.py` | Unit tests for dataset layout, manifest rows, and split report counts. |
| `yolo11_obb/classification_training.py` | Pure helper functions for class discovery, metrics, confusion matrix, and args output. |
| `scripts/train_resnet18_classifier.py` | CLI for ResNet18 train/evaluate/checkpoint workflow. |
| `tests/test_classification_training.py` | Unit tests for classification metrics without requiring a GPU. |
| `requirements.txt` | Add `openpyxl`, `torchvision`, and `PyYAML` so the new scripts declare their imports. |

Do not modify the YOLO detection training scripts in this phase.

## Task 1: Label Parsing And Group Split Helpers

**Files:**
- Create: `yolo11_obb/classification_labels.py`
- Create: `tests/test_classification_labels.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classification_labels.py`:

```python
import tempfile
import unittest
from pathlib import Path

from yolo11_obb.classification_labels import (
    LabelSample,
    normalize_status,
    parent_group_key,
    rows_to_label_samples,
    split_samples_by_group,
    write_manifest_csv,
)


class ClassificationLabelsTests(unittest.TestCase):
    def test_normalize_status_accepts_ok_and_ng_only(self) -> None:
        self.assertEqual(normalize_status("OK"), "OK")
        self.assertEqual(normalize_status("ok"), "OK")
        self.assertEqual(normalize_status("NG"), "NG")
        self.assertEqual(normalize_status(" ng "), "NG")
        self.assertIsNone(normalize_status(""))
        self.assertIsNone(normalize_status(None))
        self.assertIsNone(normalize_status("OTHER"))

    def test_parent_group_key_strips_trailing_child_index(self) -> None:
        self.assertEqual(
            parent_group_key("CropImage_20260126092714769_F3-I0_OK-5"),
            "CropImage_20260126092714769_F3-I0_OK",
        )
        self.assertEqual(parent_group_key("sample_without_child"), "sample_without_child")

    def test_rows_to_label_samples_filters_invalid_statuses(self) -> None:
        rows = [
            {"images_name": "parent-2", "tag1": "OK"},
            {"images_name": "parent-3", "tag1": "NG"},
            {"images_name": "parent-4", "tag1": "OTHER"},
            {"images_name": "", "tag1": "OK"},
        ]

        samples = rows_to_label_samples(rows, target_column="tag1")

        self.assertEqual(
            samples,
            [
                LabelSample(image_name="parent-2", class_name="OK", group="parent"),
                LabelSample(image_name="parent-3", class_name="NG", group="parent"),
            ],
        )

    def test_split_samples_by_group_keeps_related_images_together(self) -> None:
        samples = [
            LabelSample("group-a-2", "OK", "group-a"),
            LabelSample("group-a-3", "NG", "group-a"),
            LabelSample("group-b-2", "OK", "group-b"),
            LabelSample("group-c-2", "NG", "group-c"),
            LabelSample("group-d-2", "OK", "group-d"),
        ]

        split = split_samples_by_group(samples, train_ratio=0.5, seed=7)
        train_groups = {sample.group for sample in split["train"]}
        test_groups = {sample.group for sample in split["test"]}

        self.assertTrue(train_groups)
        self.assertTrue(test_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(len(split["train"]) + len(split["test"]), len(samples))

    def test_write_manifest_csv_writes_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            rows = [
                {
                    "image_name": "sample-2",
                    "class_name": "OK",
                    "group": "sample",
                    "split": "train",
                    "source_json": "sample-2.json",
                    "source_image": "sample-2.bmp",
                    "crop_path": "images/train/OK/sample-2.png",
                }
            ]

            write_manifest_csv(path, rows)

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    "image_name,class_name,group,split,source_json,source_image,crop_path",
                    "sample-2,OK,sample,train,sample-2.json,sample-2.bmp,images/train/OK/sample-2.png",
                ],
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run:

```bash
.venv/bin/python tests/test_classification_labels.py
```

Expected: fails with `ModuleNotFoundError: No module named 'yolo11_obb.classification_labels'`.

- [ ] **Step 3: Implement `classification_labels.py`**

Create `yolo11_obb/classification_labels.py`:

```python
from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class LabelSample:
    image_name: str
    class_name: str
    group: str


def normalize_status(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"OK", "NG"}:
        return text
    return None


def parent_group_key(stem: str) -> str:
    match = re.match(r"^(?P<parent>.+)-\d+$", Path(stem).stem)
    return match.group("parent") if match else Path(stem).stem


def rows_to_label_samples(rows: Iterable[Mapping[str, object]], target_column: str) -> List[LabelSample]:
    samples: List[LabelSample] = []
    for row in rows:
        image_value = row.get("images_name")
        if image_value is None:
            continue
        image_name = str(image_value).strip()
        if not image_name:
            continue
        class_name = normalize_status(row.get(target_column))
        if class_name is None:
            continue
        samples.append(
            LabelSample(
                image_name=Path(image_name).stem,
                class_name=class_name,
                group=parent_group_key(image_name),
            )
        )
    return samples


def split_samples_by_group(
    samples: Sequence[LabelSample],
    train_ratio: float,
    seed: int,
) -> Dict[str, List[LabelSample]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    groups = sorted({sample.group for sample in samples})
    if not groups:
        raise ValueError("no samples to split")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_ratio)
    if len(shuffled) > 1:
        train_count = min(max(train_count, 1), len(shuffled) - 1)
    train_groups = set(shuffled[:train_count])
    return {
        "train": [sample for sample in samples if sample.group in train_groups],
        "test": [sample for sample in samples if sample.group not in train_groups],
    }


def load_sheet_rows(xlsx_path: Path, sheet_name: str) -> List[Dict[str, object]]:
    try:
        from openpyxl import load_workbook
    except ModuleNotFoundError as exc:
        raise RuntimeError("openpyxl is required to read Excel label files") from exc

    workbook = load_workbook(xlsx_path, read_only=True, data_only=True)
    if sheet_name not in workbook.sheetnames:
        raise ValueError(f"sheet not found in {xlsx_path}: {sheet_name}")
    worksheet = workbook[sheet_name]
    rows = list(worksheet.iter_rows(values_only=True))
    if not rows:
        return []
    headers = ["" if value is None else str(value).strip() for value in rows[0]]
    return [
        {headers[index]: value for index, value in enumerate(row) if index < len(headers) and headers[index]}
        for row in rows[1:]
    ]


def write_manifest_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    fieldnames = [
        "image_name",
        "class_name",
        "group",
        "split",
        "source_json",
        "source_image",
        "crop_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
```

- [ ] **Step 4: Add dependencies to `requirements.txt`**

Change `requirements.txt` from:

```text
ultralytics
```

to:

```text
ultralytics
openpyxl
torchvision
PyYAML
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run:

```bash
.venv/bin/python tests/test_classification_labels.py
```

Expected: `Ran 5 tests` and `OK`.

- [ ] **Step 6: Commit Task 1**

Run:

```bash
git add requirements.txt yolo11_obb/classification_labels.py tests/test_classification_labels.py
git commit -m "feat: add classification label parsing helpers"
```

## Task 2: OBB Crop Rectification

**Files:**
- Create: `yolo11_obb/obb_crop.py`
- Create: `tests/test_obb_crop.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_obb_crop.py`:

```python
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.obb_crop import ordered_polygon_points, rectify_obb_crop, save_obb_crop


class ObbCropTests(unittest.TestCase):
    def test_ordered_polygon_points_returns_top_left_clockwise(self) -> None:
        points = [(40, 30), (10, 10), (40, 10), (10, 30)]

        ordered = ordered_polygon_points(points)

        self.assertEqual(ordered, [(10.0, 10.0), (40.0, 10.0), (40.0, 30.0), (10.0, 30.0)])

    def test_rectify_obb_crop_returns_expected_size_for_rectangle(self) -> None:
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        image[10:30, 10:40] = (0, 255, 0)

        crop = rectify_obb_crop(image, [(10, 10), (40, 10), (40, 30), (10, 30)])

        self.assertEqual(crop.shape[:2], (20, 30))
        self.assertGreater(int(crop[:, :, 1].mean()), 200)

    def test_rectify_obb_crop_rejects_bad_polygon(self) -> None:
        image = np.zeros((50, 60, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "expected 4 points"):
            rectify_obb_crop(image, [(10, 10), (40, 10), (40, 30)])

    def test_save_obb_crop_writes_png(self) -> None:
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        image[10:30, 10:40] = (255, 0, 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "crop.png"

            save_obb_crop(image, [(10, 10), (40, 10), (40, 30), (10, 30)], output)

            saved = cv2.imread(str(output))
            self.assertIsNotNone(saved)
            self.assertEqual(saved.shape[:2], (20, 30))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run:

```bash
.venv/bin/python tests/test_obb_crop.py
```

Expected: fails with `ModuleNotFoundError: No module named 'yolo11_obb.obb_crop'`.

- [ ] **Step 3: Implement `obb_crop.py`**

Create `yolo11_obb/obb_crop.py`:

```python
from __future__ import annotations

from math import hypot
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


def ordered_polygon_points(points: Sequence[Sequence[float]]) -> list[Point]:
    if len(points) != 4:
        raise ValueError(f"expected 4 points, got {len(points)}")
    array = np.array(points, dtype=np.float32)
    sums = array.sum(axis=1)
    diffs = np.diff(array, axis=1).reshape(-1)
    top_left = array[int(np.argmin(sums))]
    bottom_right = array[int(np.argmax(sums))]
    top_right = array[int(np.argmin(diffs))]
    bottom_left = array[int(np.argmax(diffs))]
    return [
        (float(top_left[0]), float(top_left[1])),
        (float(top_right[0]), float(top_right[1])),
        (float(bottom_right[0]), float(bottom_right[1])),
        (float(bottom_left[0]), float(bottom_left[1])),
    ]


def _target_size(points: Sequence[Point]) -> tuple[int, int]:
    top_width = hypot(points[1][0] - points[0][0], points[1][1] - points[0][1])
    bottom_width = hypot(points[2][0] - points[3][0], points[2][1] - points[3][1])
    left_height = hypot(points[3][0] - points[0][0], points[3][1] - points[0][1])
    right_height = hypot(points[2][0] - points[1][0], points[2][1] - points[1][1])
    width = max(int(round(max(top_width, bottom_width))), 1)
    height = max(int(round(max(left_height, right_height))), 1)
    return width, height


def rectify_obb_crop(image: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    ordered = ordered_polygon_points(points)
    width, height = _target_size(ordered)
    source = np.array(ordered, dtype=np.float32)
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, matrix, (width, height))


def save_obb_crop(image: np.ndarray, points: Sequence[Sequence[float]], output: Path) -> None:
    crop = rectify_obb_crop(image, points)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), crop):
        raise RuntimeError(f"failed to write crop: {output}")
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run:

```bash
.venv/bin/python tests/test_obb_crop.py
```

Expected: `Ran 4 tests` and `OK`.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add yolo11_obb/obb_crop.py tests/test_obb_crop.py
git commit -m "feat: add obb crop rectification"
```

## Task 3: Classification Dataset Builder

**Files:**
- Create: `yolo11_obb/classification_dataset.py`
- Create: `scripts/create_label_classification_dataset.py`
- Create: `tests/test_classification_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_classification_dataset.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.classification_dataset import build_classification_dataset
from yolo11_obb.classification_labels import LabelSample


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:50, 10:60] = (0, 255, 0)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def write_annotation(path: Path, label: str = "label5") -> None:
    write_image(path.with_suffix(".bmp"))
    path.write_text(
        json.dumps(
            {
                "imagePath": path.with_suffix(".bmp").name,
                "imageWidth": 80,
                "imageHeight": 80,
                "shapes": [
                    {
                        "label": label,
                        "shape_type": "polygon",
                        "points": [[10, 20], [60, 20], [60, 50], [10, 50]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class ClassificationDatasetTests(unittest.TestCase):
    def test_build_classification_dataset_writes_crops_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "anylabeling"
            output = root / "dataset"
            write_annotation(source / "group-a-2.json")
            write_annotation(source / "group-b-2.json")
            write_annotation(source / "group-c-2.json")
            samples = [
                LabelSample("group-a-2", "OK", "group-a"),
                LabelSample("group-b-2", "NG", "group-b"),
                LabelSample("group-c-2", "NG", "group-c"),
            ]

            report = build_classification_dataset(
                source=source,
                output=output,
                samples=samples,
                label_name="label5",
                train_ratio=0.67,
                seed=3,
                overwrite=False,
            )

            self.assertEqual(report["total_samples"], 3)
            self.assertEqual(report["classes"], {"OK": 1, "NG": 2})
            self.assertTrue((output / "manifest.csv").exists())
            self.assertTrue((output / "split_report.txt").exists())
            crops = list((output / "images").glob("*/*/*.png"))
            self.assertEqual(len(crops), 3)
            for crop in crops:
                image = cv2.imread(str(crop))
                self.assertIsNotNone(image)
                self.assertEqual(image.shape[:2], (30, 50))

    def test_build_classification_dataset_fails_when_label_shape_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "anylabeling"
            output = root / "dataset"
            write_annotation(source / "group-a-2.json", label="label4")
            samples = [LabelSample("group-a-2", "OK", "group-a")]

            with self.assertRaisesRegex(ValueError, "missing shape label5"):
                build_classification_dataset(
                    source=source,
                    output=output,
                    samples=samples,
                    label_name="label5",
                    train_ratio=0.8,
                    seed=42,
                    overwrite=False,
                )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run:

```bash
.venv/bin/python tests/test_classification_dataset.py
```

Expected: fails with `ModuleNotFoundError: No module named 'yolo11_obb.classification_dataset'`.

- [ ] **Step 3: Implement `classification_dataset.py`**

Create `yolo11_obb/classification_dataset.py`:

```python
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

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
```

- [ ] **Step 4: Implement dataset CLI**

Create `scripts/create_label_classification_dataset.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.classification_dataset import build_classification_dataset
from yolo11_obb.classification_labels import load_sheet_rows, rows_to_label_samples
from yolo11_obb.config import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cropped classification dataset from AnyLabeling OBB annotations.")
    parser.add_argument("--excel", type=Path, default=Path("outputs/label1_6_description.xlsx"))
    parser.add_argument("--source", type=Path, default=Path("已打标的数据202604/user1_2026-03-16_154843_anylabeling"))
    parser.add_argument("--output", type=Path, default=Path("datasets/classification/label5_ok_ng"))
    parser.add_argument("--label", default="label5")
    parser.add_argument("--target-column", default="tag1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excel = resolve_from_root(args.excel, ROOT)
    source = resolve_from_root(args.source, ROOT)
    output = resolve_from_root(args.output, ROOT)
    rows = load_sheet_rows(excel, args.label)
    samples = rows_to_label_samples(rows, target_column=args.target_column)
    report = build_classification_dataset(
        source=source,
        output=output,
        samples=samples,
        label_name=args.label,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"dataset: {output}")
    print(f"total_samples: {report['total_samples']}")
    print(f"classes: {report['classes']}")
    print(f"train: {report['train']}")
    print(f"test: {report['test']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run:

```bash
.venv/bin/python tests/test_classification_dataset.py
```

Expected: `Ran 2 tests` and `OK`.

- [ ] **Step 6: Build the real label5 dataset**

Run with the Python environment that has `openpyxl`, OpenCV, and NumPy:

```bash
python3 scripts/create_label_classification_dataset.py \
  --excel outputs/label1_6_description.xlsx \
  --source '已打标的数据202604/user1_2026-03-16_154843_anylabeling' \
  --output datasets/classification/label5_ok_ng \
  --label label5 \
  --target-column tag1 \
  --train-ratio 0.8 \
  --seed 42 \
  --overwrite
```

Expected: prints `total_samples: 251`, class counts containing `OK: 98` and `NG: 153`, and creates `datasets/classification/label5_ok_ng/manifest.csv`.

- [ ] **Step 7: Inspect the real dataset report**

Run:

```bash
sed -n '1,80p' datasets/classification/label5_ok_ng/split_report.txt
```

Expected: output includes `label_name: label5`, `total_samples: 251`, and non-empty train/test class counts.

- [ ] **Step 8: Commit Task 3 code only**

Do not commit generated dataset files.

Run:

```bash
git add yolo11_obb/classification_dataset.py scripts/create_label_classification_dataset.py tests/test_classification_dataset.py
git commit -m "feat: build label classification crops"
```

## Task 4: ResNet18 Training, Metrics, And Smoke Run

**Files:**
- Create: `yolo11_obb/classification_training.py`
- Create: `scripts/train_resnet18_classifier.py`
- Create: `tests/test_classification_training.py`

- [ ] **Step 1: Write the pure helper tests**

Create `tests/test_classification_training.py`:

```python
import unittest

from yolo11_obb.classification_training import (
    classification_metrics,
    confusion_counts,
    discover_classes,
)


class ClassificationTrainingTests(unittest.TestCase):
    def test_discover_classes_sorts_expected_class_names(self) -> None:
        self.assertEqual(discover_classes(["NG", "OK"]), {"NG": 0, "OK": 1})

    def test_confusion_counts_counts_pairs(self) -> None:
        counts = confusion_counts(["OK", "OK", "NG", "NG"], ["OK", "NG", "NG", "OK"], ["NG", "OK"])

        self.assertEqual(
            counts,
            {
                ("NG", "NG"): 1,
                ("NG", "OK"): 1,
                ("OK", "NG"): 1,
                ("OK", "OK"): 1,
            },
        )

    def test_classification_metrics_returns_macro_f1(self) -> None:
        metrics = classification_metrics(["OK", "OK", "NG", "NG"], ["OK", "NG", "NG", "OK"], ["NG", "OK"])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["per_class"]["OK"]["precision"], 0.5)
        self.assertEqual(metrics["per_class"]["NG"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run:

```bash
.venv/bin/python tests/test_classification_training.py
```

Expected: fails with `ModuleNotFoundError: No module named 'yolo11_obb.classification_training'`.

- [ ] **Step 3: Implement pure training helpers**

Create `yolo11_obb/classification_training.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def discover_classes(class_names: Iterable[str]) -> Dict[str, int]:
    names = sorted(set(class_names))
    if len(names) < 2:
        raise ValueError("at least two classes are required")
    return {name: index for index, name in enumerate(names)}


def confusion_counts(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> Dict[Tuple[str, str], int]:
    counts = {(true_name, pred_name): 0 for true_name in class_names for pred_name in class_names}
    for true_name, pred_name in zip(true_labels, predicted_labels):
        counts[(true_name, pred_name)] += 1
    return counts


def classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, object]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    total = len(true_labels)
    if total == 0:
        raise ValueError("no predictions to score")
    counts = confusion_counts(true_labels, predicted_labels, class_names)
    correct = sum(counts[(name, name)] for name in class_names)
    per_class: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    for name in class_names:
        tp = counts[(name, name)]
        fp = sum(counts[(other, name)] for other in class_names if other != name)
        fn = sum(counts[(name, other)] for other in class_names if other != name)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def write_confusion_matrix_csv(
    path: Path,
    counts: Mapping[Tuple[str, str], int],
    class_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for true_name in class_names:
            writer.writerow([true_name, *[counts[(true_name, pred_name)] for pred_name in class_names]])
```

- [ ] **Step 4: Implement ResNet18 training CLI**

Create `scripts/train_resnet18_classifier.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from yolo11_obb.classification_training import (
    classification_metrics,
    confusion_counts,
    write_confusion_matrix_csv,
)
from yolo11_obb.config import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 classifier on cropped OBB regions.")
    parser.add_argument("--data", type=Path, default=Path("datasets/classification/label5_ok_ng"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/classification"))
    parser.add_argument("--name", default="label5_resnet18")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_labels = []
    pred_labels = []
    image_paths = []
    class_names = loader.dataset.classes
    with torch.set_grad_enabled(training):
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            predictions = outputs.argmax(dim=1).detach().cpu().tolist()
            true_labels.extend(class_names[index] for index in targets.detach().cpu().tolist())
            pred_labels.extend(class_names[index] for index in predictions)
            image_paths.extend(paths)
    metrics = classification_metrics(true_labels, pred_labels, class_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, true_labels, pred_labels, image_paths


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    project = resolve_from_root(args.project, ROOT)
    run_dir = project / args.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ImageFolderWithPaths(data / "images" / "train", transform=transform)
    test_dataset = ImageFolderWithPaths(data / "images" / "test", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=len(train_dataset.classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args_payload = {
        "data": str(data),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "imgsz": args.imgsz,
        "device": str(device),
        "workers": args.workers,
        "pretrained": not args.no_pretrained,
        "model_selection": "best test macro_f1 on requested 8:2 split; optimistic because no separate validation split",
        "classes": train_dataset.classes,
    }
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    metrics_path = run_dir / "metrics.csv"
    best_score = -1.0
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_accuracy", "train_macro_f1", "test_loss", "test_accuracy", "test_macro_f1"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics, _, _, _ = epoch_pass(model, train_loader, criterion, device, optimizer=optimizer)
            test_metrics, true_labels, pred_labels, _ = epoch_pass(model, test_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "train_macro_f1": f"{train_metrics['macro_f1']:.6f}",
                "test_loss": f"{test_metrics['loss']:.6f}",
                "test_accuracy": f"{test_metrics['accuracy']:.6f}",
                "test_macro_f1": f"{test_metrics['macro_f1']:.6f}",
            }
            writer.writerow(row)
            handle.flush()
            score = float(test_metrics["macro_f1"])
            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "classes": train_dataset.classes, "epoch": epoch, "score": best_score}, run_dir / "weights" / "best.pt")
            torch.save({"model": model.state_dict(), "classes": train_dataset.classes, "epoch": epoch, "score": score}, run_dir / "weights" / "last.pt")

    final_metrics, true_labels, pred_labels, image_paths = epoch_pass(model, test_loader, criterion, device)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_label", "predicted_label"])
        writer.writeheader()
        for image_path, true_name, pred_name in zip(image_paths, true_labels, pred_labels):
            writer.writerow({"image_path": image_path, "true_label": true_name, "predicted_label": pred_name})
    counts = confusion_counts(true_labels, pred_labels, test_dataset.classes)
    write_confusion_matrix_csv(run_dir / "confusion_matrix.csv", counts, test_dataset.classes)
    print(f"run: {run_dir}")
    print(f"best_macro_f1: {best_score:.6f}")
    print(f"final_test_accuracy: {final_metrics['accuracy']:.6f}")
    print(f"final_test_macro_f1: {final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run pure helper tests and confirm they pass**

Run:

```bash
.venv/bin/python tests/test_classification_training.py
```

Expected: `Ran 3 tests` and `OK`.

- [ ] **Step 6: Run all new unit tests**

Run:

```bash
.venv/bin/python tests/test_classification_labels.py
.venv/bin/python tests/test_obb_crop.py
.venv/bin/python tests/test_classification_dataset.py
.venv/bin/python tests/test_classification_training.py
```

Expected: all four commands finish with `OK`.

- [ ] **Step 7: Run a CPU smoke train on the real dataset**

Use `--no-pretrained` to avoid network downloads.

Run:

```bash
.venv/bin/python scripts/train_resnet18_classifier.py \
  --data datasets/classification/label5_ok_ng \
  --epochs 1 \
  --batch 8 \
  --workers 0 \
  --device cpu \
  --name label5_resnet18_smoke \
  --no-pretrained \
  --exist-ok
```

Expected: creates `runs/classification/label5_resnet18_smoke/weights/best.pt`, `metrics.csv`, `predictions.csv`, `confusion_matrix.csv`, and `args.yaml`.

- [ ] **Step 8: Inspect smoke outputs**

Run:

```bash
sed -n '1,5p' runs/classification/label5_resnet18_smoke/metrics.csv
sed -n '1,20p' runs/classification/label5_resnet18_smoke/confusion_matrix.csv
```

Expected: `metrics.csv` contains one epoch row and `confusion_matrix.csv` contains `OK` and `NG` rows.

- [ ] **Step 9: Commit Task 4 code only**

Do not commit generated run outputs.

Run:

```bash
git add yolo11_obb/classification_training.py scripts/train_resnet18_classifier.py tests/test_classification_training.py
git commit -m "feat: train resnet18 region classifier"
```

## Task 5: Final Verification And User Commands

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short README classification section**

Append this section near the existing dataset/training instructions in `README.md`:

````markdown
## Label5 ResNet18 Classification

Create the first cropped classification dataset from manually annotated AnyLabeling OBB boxes:

```bash
python3 scripts/create_label_classification_dataset.py \
  --excel outputs/label1_6_description.xlsx \
  --source '已打标的数据202604/user1_2026-03-16_154843_anylabeling' \
  --output datasets/classification/label5_ok_ng \
  --label label5 \
  --target-column tag1 \
  --train-ratio 0.8 \
  --seed 42 \
  --overwrite
```

Run a smoke train without downloading pretrained weights:

```bash
.venv/bin/python scripts/train_resnet18_classifier.py \
  --data datasets/classification/label5_ok_ng \
  --epochs 1 \
  --batch 8 \
  --workers 0 \
  --device cpu \
  --name label5_resnet18_smoke \
  --no-pretrained \
  --exist-ok
```

Run a normal training job on the server:

```bash
python3 scripts/train_resnet18_classifier.py \
  --data datasets/classification/label5_ok_ng \
  --epochs 30 \
  --batch 32 \
  --device 0 \
  --name label5_resnet18_e30
```

The first version selects `best.pt` using test macro F1 on the requested 8:2 split. This is useful for the initial pipeline but is optimistic because there is no separate validation split.
````

- [ ] **Step 2: Run all targeted tests**

Run:

```bash
.venv/bin/python tests/test_classification_labels.py
.venv/bin/python tests/test_obb_crop.py
.venv/bin/python tests/test_classification_dataset.py
.venv/bin/python tests/test_classification_training.py
```

Expected: all commands finish with `OK`.

- [ ] **Step 3: Verify no generated datasets or runs are staged**

Run:

```bash
git status --short
```

Expected: generated paths such as `datasets/classification/` and `runs/classification/` may appear untracked, but only source files, tests, `requirements.txt`, and `README.md` should be staged for the next commit.

- [ ] **Step 4: Commit README update**

Run:

```bash
git add README.md
git commit -m "docs: add label5 classification commands"
```

- [ ] **Step 5: Final response evidence**

In the final response, report:

```text
Implemented label5 ResNet18 classification pipeline.
Verified:
- classification label tests
- OBB crop tests
- classification dataset tests
- classification metric tests
- real dataset build or exact blocker
- one-epoch smoke train or exact blocker
```

Mention generated artifact paths only if they exist locally:

```text
datasets/classification/label5_ok_ng
runs/classification/label5_resnet18_smoke
```
