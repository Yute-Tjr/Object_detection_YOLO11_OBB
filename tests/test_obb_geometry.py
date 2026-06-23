import unittest

from yolo11_obb.obb_geometry import (
    ObbBox,
    match_ground_truths,
    polygon_iou,
    top_edge_width,
)


class ObbGeometryTests(unittest.TestCase):
    def test_top_edge_width_uses_edge_with_highest_position(self) -> None:
        points = ((9.0, 7.0), (1.0, 8.0), (2.0, 5.0), (8.0, 4.0))

        self.assertAlmostEqual(top_edge_width(points), 6.0827625303)

    def test_polygon_iou_for_overlapping_rectangles(self) -> None:
        left = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
        right = ((1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0))

        self.assertAlmostEqual(polygon_iou(left, right), 1.0 / 3.0)

    def test_match_ground_truths_finds_best_same_class_prediction(self) -> None:
        gt = ObbBox(
            class_id=0,
            points=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
        )
        wrong_class = ObbBox(
            class_id=1,
            points=((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)),
            confidence=0.99,
        )
        same_class = ObbBox(
            class_id=0,
            points=((0.5, 0.0), (2.5, 0.0), (2.5, 2.0), (0.5, 2.0)),
            confidence=0.90,
        )

        matches = match_ground_truths([gt], [wrong_class, same_class], iou_threshold=0.85)

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].ground_truth, gt)
        self.assertEqual(matches[0].prediction, same_class)
        self.assertAlmostEqual(matches[0].iou, 0.6)
        self.assertFalse(matches[0].passed)


if __name__ == "__main__":
    unittest.main()
