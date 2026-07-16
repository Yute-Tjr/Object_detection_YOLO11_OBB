import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.rhino_dataset import create_rhino_dataset


def write_image(path: Path, width: int = 100, height: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def write_yolo_dataset(root: Path) -> Path:
    for split, class_id in (("train", 0), ("test", 1)):
        write_image(root / "images" / split / f"sample_{split}.bmp")
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split / f"sample_{split}.txt").write_text(
            f"{class_id} 0.10 0.10 0.50 0.10 0.50 0.40 0.10 0.40\n",
            encoding="utf-8",
        )
    data = root / "data.yaml"
    data.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/test",
                "test: images/test",
                "names:",
                "  0: label1_thin",
                "  1: label1_thick",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data


class RhinoDatasetTests(unittest.TestCase):
    def test_create_rhino_dataset_converts_yolo_obb_to_dota_annfiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_data = write_yolo_dataset(root / "obb_thin_thick")
            output = root / "rhino_obb"

            report = create_rhino_dataset(source_data, output)

            self.assertEqual(report.images_by_split, {"train": 1, "test": 1})
            self.assertEqual(report.objects_by_split, {"train": 1, "test": 1})
            self.assertEqual(report.image_format, "png")
            self.assertTrue((output / "train" / "images" / "sample_train.png").exists())
            self.assertFalse((output / "train" / "images" / "sample_train.bmp").exists())
            self.assertEqual(
                (output / "test" / "annfiles" / "sample_test.txt").read_text(encoding="utf-8"),
                "10.0000 8.0000 50.0000 8.0000 50.0000 32.0000 10.0000 32.0000 label1_thick 0\n",
            )
            self.assertIn("label1_thin", (output / "classes.txt").read_text(encoding="utf-8"))

    def test_create_rhino_dataset_rejects_nonempty_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_data = write_yolo_dataset(root / "obb_thin_thick")
            output = root / "rhino_obb"
            output.mkdir()
            (output / "existing.txt").write_text("existing\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                create_rhino_dataset(source_data, output)


if __name__ == "__main__":
    unittest.main()
