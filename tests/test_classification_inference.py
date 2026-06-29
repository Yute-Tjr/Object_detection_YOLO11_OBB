import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.classification_inference import (
    collect_image_paths,
    normalize_device_name,
    prediction_rows,
    select_device,
)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


class ClassificationInferenceTests(unittest.TestCase):
    def test_collect_image_paths_accepts_file_and_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "b.png"
            second = root / "nested" / "a.jpg"
            ignored = root / "notes.txt"
            write_image(first)
            write_image(second)
            ignored.write_text("ignore", encoding="utf-8")

            self.assertEqual(collect_image_paths(first), [first.resolve()])
            self.assertEqual(collect_image_paths(root), [first.resolve(), second.resolve()])

    def test_prediction_rows_include_confidence_and_per_class_probabilities(self) -> None:
        rows = prediction_rows(
            image_paths=[Path("b.png"), Path("a.png")],
            class_names=["NG", "OK"],
            probabilities=[[0.2, 0.8], [0.9, 0.1]],
        )

        self.assertEqual(
            rows,
            [
                {
                    "image_path": "b.png",
                    "predicted_label": "OK",
                    "confidence": "0.800000",
                    "prob_NG": "0.200000",
                    "prob_OK": "0.800000",
                },
                {
                    "image_path": "a.png",
                    "predicted_label": "NG",
                    "confidence": "0.900000",
                    "prob_NG": "0.900000",
                    "prob_OK": "0.100000",
                },
            ],
        )

    def test_select_device_accepts_cpu(self) -> None:
        self.assertEqual(select_device("cpu").type, "cpu")

    def test_normalize_device_name_treats_digit_as_cuda_index(self) -> None:
        self.assertEqual(normalize_device_name("0"), "cuda:0")
        self.assertEqual(normalize_device_name("cuda:1"), "cuda:1")
        self.assertIsNone(normalize_device_name(None))


if __name__ == "__main__":
    unittest.main()
