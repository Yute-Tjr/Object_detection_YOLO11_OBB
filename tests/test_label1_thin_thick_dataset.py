import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.label1_thin_thick_dataset import (
    LABEL1_THIN_THICK_NAMES,
    create_label1_thin_thick_dataset,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Label1ThinThickDatasetTests(unittest.TestCase):
    def test_create_dataset_splits_label1_by_top_edge_width_and_remaps_other_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "project" / "datasets" / "thin_thick"

            thin_stem = "CropImage_20260101000000000_F3-I0_NG-1"
            thick_stem = "CropImage_20260101000000001_F3-I0_NG-1"
            write_image(source / f"{thin_stem}.bmp")
            write_image(source / f"{thick_stem}.bmp")
            write_label(
                source / f"{thin_stem}.txt",
                "\n".join(
                    [
                        "0 0.45 0.10 0.55 0.10 0.55 0.30 0.45 0.30",
                        "1 0.10 0.40 0.30 0.40 0.30 0.50 0.10 0.50",
                    ]
                )
                + "\n",
            )
            write_label(
                source / f"{thick_stem}.txt",
                "0 0.35 0.10 0.65 0.10 0.65 0.30 0.35 0.30\n",
            )

            report = create_label1_thin_thick_dataset(
                source=source,
                output=output,
                top_edge_threshold_px=20.0,
                train_ratio=0.5,
                seed=1,
            )

            selfEqual = self.assertEqual
            selfEqual(report["train"]["images"] + report["test"]["images"], 2)
            selfEqual(report["train"]["class_counts"][0] + report["test"]["class_counts"][0], 1)
            selfEqual(report["train"]["class_counts"][1] + report["test"]["class_counts"][1], 1)
            selfEqual(report["train"]["class_counts"][2] + report["test"]["class_counts"][2], 1)

            all_labels = "\n".join(
                path.read_text(encoding="utf-8").strip()
                for path in sorted((output / "labels").glob("*/*.txt"))
            )
            self.assertIn("0 0.45 0.10 0.55 0.10 0.55 0.30 0.45 0.30", all_labels)
            self.assertIn("1 0.35 0.10 0.65 0.10 0.65 0.30 0.35 0.30", all_labels)
            self.assertIn("2 0.10 0.40 0.30 0.40 0.30 0.50 0.10 0.50", all_labels)

            data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("path: datasets/thin_thick", data_yaml)
            for class_id, name in LABEL1_THIN_THICK_NAMES.items():
                self.assertIn(f"  {class_id}: {name}", data_yaml)


if __name__ == "__main__":
    unittest.main()
