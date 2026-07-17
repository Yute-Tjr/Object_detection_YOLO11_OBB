import tempfile
import unittest
from pathlib import Path

from yolo11_obb.rhino_prediction import index_images_by_stem, rbox_to_corners, rboxes_to_yolo_lines


class RhinoPredictionTests(unittest.TestCase):
    def test_index_images_by_stem_matches_across_file_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.bmp"
            image_path.touch()

            indexed = index_images_by_stem([image_path])

            self.assertEqual(indexed[Path("sample.png").stem], image_path)

    def test_index_images_by_stem_rejects_duplicate_stems(self) -> None:
        with self.assertRaises(ValueError):
            index_images_by_stem([Path("sample.bmp"), Path("sample.png")])

    def test_rbox_to_corners_converts_center_width_height_angle(self) -> None:
        corners = rbox_to_corners([50.0, 40.0, 40.0, 20.0, 0.0])

        self.assertEqual(corners, [(30.0, 30.0), (70.0, 30.0), (70.0, 50.0), (30.0, 50.0)])

    def test_rboxes_to_yolo_lines_normalizes_points_and_keeps_confidence(self) -> None:
        lines = rboxes_to_yolo_lines(
            rboxes=[[50.0, 40.0, 40.0, 20.0, 0.0]],
            labels=[1],
            scores=[0.75],
            image_width=100,
            image_height=80,
        )

        self.assertEqual(lines, ["1 0.300000 0.375000 0.700000 0.375000 0.700000 0.625000 0.300000 0.625000 0.750000"])


if __name__ == "__main__":
    unittest.main()
