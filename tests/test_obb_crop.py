import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
