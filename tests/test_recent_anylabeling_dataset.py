import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.recent_anylabeling_dataset import (
    create_recent_label1_thin_thick_dataset,
    sample_order_key,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def write_annotation(path: Path, label1_width: int, extra_label: str = "label2") -> None:
    write_image(path.with_suffix(".bmp"))
    path.write_text(
        json.dumps(
            {
                "imagePath": path.with_suffix(".bmp").name,
                "imageWidth": 100,
                "imageHeight": 100,
                "shapes": [
                    {
                        "label": "label1",
                        "points": [[10, 10], [10 + label1_width, 10], [10 + label1_width, 30], [10, 30]],
                    },
                    {
                        "label": extra_label,
                        "points": [[40, 40], [70, 40], [70, 60], [40, 60]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


class RecentAnyLabelingDatasetTests(unittest.TestCase):
    def test_sample_order_key_uses_timestamp_and_trailing_index(self) -> None:
        self.assertEqual(
            sample_order_key("CropImage_20260126092651160_F3-I0_OK-7"),
            ("20260126092651160", 7),
        )
        self.assertEqual(
            sample_order_key("20260126092651160_F3-10_NG-7"),
            ("20260126092651160", 7),
        )
        self.assertEqual(
            sample_order_key("20260121210219803"),
            ("20260121210219803", 999),
        )

    def test_create_recent_dataset_excludes_marker_and_splits_label1_thin_thick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "anylabeling"
            converted = root / "converted"
            output = root / "datasets" / "recent_thin_thick"

            write_annotation(source / "CropImage_20260101000000000_F3-I0_OK-6.json", label1_width=10)
            write_annotation(source / "CropImage_20260101000000000_F3-I0_OK-7.json", label1_width=10)
            write_annotation(source / "CropImage_20260101000001000_F3-I0_OK-1.json", label1_width=10)
            write_annotation(source / "CropImage_20260101000002000_F3-I0_OK-1.json", label1_width=30, extra_label="label3")

            report = create_recent_label1_thin_thick_dataset(
                source=source,
                converted_output=converted,
                dataset_output=output,
                after_stem="20260101000000000_F3-10_NG-7",
                top_edge_threshold_px=20.0,
                train_ratio=0.5,
                seed=1,
            )

            self.assertEqual(report.selected_json_files, 2)
            self.assertFalse((converted / "CropImage_20260101000000000_F3-I0_OK-7.txt").exists())
            self.assertTrue((converted / "CropImage_20260101000001000_F3-I0_OK-1.txt").exists())

            class_counts = {
                class_id: report.dataset["train"]["class_counts"][class_id]
                + report.dataset["test"]["class_counts"][class_id]
                for class_id in range(7)
            }
            self.assertEqual(class_counts[0], 1)
            self.assertEqual(class_counts[1], 1)
            self.assertEqual(class_counts[2], 1)
            self.assertEqual(class_counts[3], 1)

            data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("path: datasets/recent_thin_thick", data_yaml)
            self.assertIn("  0: label1_thin", data_yaml)
            self.assertIn("  1: label1_thick", data_yaml)

    def test_create_recent_dataset_can_exclude_trailing_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "anylabeling"
            converted = root / "converted"
            output = root / "datasets" / "recent_thin_thick_no_index1"

            write_annotation(source / "CropImage_20260101000000000_F3-I0_OK-7.json", label1_width=10)
            write_annotation(source / "CropImage_20260101000001000_F3-I0_OK-1.json", label1_width=30)
            write_annotation(source / "CropImage_20260101000001000_F3-I0_OK-2.json", label1_width=10)
            write_annotation(source / "CropImage_20260101000001000_F3-I0_OK-3.json", label1_width=10)

            report = create_recent_label1_thin_thick_dataset(
                source=source,
                converted_output=converted,
                dataset_output=output,
                after_stem="20260101000000000_F3-I0_OK-7",
                top_edge_threshold_px=20.0,
                train_ratio=0.5,
                seed=1,
                exclude_indices=[1],
            )

            self.assertEqual(report.selected_json_files, 2)
            self.assertEqual(report.excluded_json_files, 1)
            self.assertFalse((converted / "CropImage_20260101000001000_F3-I0_OK-1.txt").exists())
            self.assertTrue((converted / "CropImage_20260101000001000_F3-I0_OK-2.txt").exists())

            manifest = (output / "split_manifest.csv").read_text(encoding="utf-8")
            self.assertNotIn("OK-1.bmp", manifest)
            self.assertIn("OK-2.bmp", manifest)


if __name__ == "__main__":
    unittest.main()
