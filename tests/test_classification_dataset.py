import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
