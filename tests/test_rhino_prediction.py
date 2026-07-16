import unittest

from yolo11_obb.rhino_prediction import rbox_to_corners, rboxes_to_yolo_lines


class RhinoPredictionTests(unittest.TestCase):
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
