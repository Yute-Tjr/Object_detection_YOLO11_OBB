from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yolo11_obb.obb_geometry import parse_obb_line
from yolo11_obb.prediction_fusion import (
    fuse_prediction_labels,
    inverse_deskew_box,
    write_obb_prediction_line,
)


class PredictionFusionTests(unittest.TestCase):
    def test_inverse_deskew_box_maps_rotated_box_back_to_source_coordinates(self) -> None:
        source_box = parse_obb_line(
            "2 0.3 0.2 0.7 0.2 0.7 0.4 0.3 0.4 0.91",
            image_width=100,
            image_height=80,
        )
        deskewed_box = parse_obb_line(
            "2 0.3 0.2 0.7 0.2 0.7 0.4 0.3 0.4 0.91",
            image_width=100,
            image_height=80,
        )

        restored = inverse_deskew_box(
            deskewed_box,
            source_width=100,
            source_height=80,
            deskew_width=100,
            deskew_height=80,
            deskew_angle_degrees=0.0,
        )

        for actual, expected in zip(restored.points, source_box.points):
            self.assertAlmostEqual(actual[0], expected[0], places=4)
            self.assertAlmostEqual(actual[1], expected[1], places=4)
        self.assertEqual(restored.confidence, source_box.confidence)

    def test_fuse_prediction_labels_selects_configured_classes_and_restores_deskew_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline"
            deskew = root / "deskew"
            output = root / "fused"
            baseline.mkdir()
            deskew.mkdir()
            stem = "sample"

            (baseline / f"{stem}.txt").write_text(
                "\n".join(
                    [
                        "1 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3 0.80",
                        "3 0.4 0.1 0.6 0.1 0.6 0.3 0.4 0.3 0.70",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (deskew / f"{stem}.txt").write_text(
                "3 0.4 0.1 0.6 0.1 0.6 0.3 0.4 0.3 0.95\n",
                encoding="utf-8",
            )
            report = root / "deskew_report.csv"
            report.write_text(
                "\n".join(
                    [
                        "split,image,label,deskew_angle_degrees,objects,source_width,source_height,output_width,output_height",
                        f"test,{stem}.bmp,{stem}.txt,0.0,1,100,80,100,80",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = fuse_prediction_labels(
                baseline_labels=baseline,
                deskew_labels=deskew,
                deskew_report=report,
                output_labels=output,
                baseline_class_ids={1},
                deskew_class_ids={3},
            )

            self.assertEqual(len(rows), 1)
            fused_lines = (output / f"{stem}.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(fused_lines), 2)
            self.assertTrue(fused_lines[0].startswith("1 "))
            self.assertTrue(fused_lines[1].startswith("3 "))
            restored = parse_obb_line(fused_lines[1], image_width=100, image_height=80)
            self.assertAlmostEqual(restored.points[0][0], 40.0, places=3)
            self.assertAlmostEqual(restored.points[0][1], 8.0, places=3)

    def test_write_obb_prediction_line_preserves_confidence(self) -> None:
        box = parse_obb_line("5 0 0 1 0 1 1 0 1 0.987654")

        self.assertEqual(write_obb_prediction_line(box), "5 0 0 1 0 1 1 0 1 0.987654")


if __name__ == "__main__":
    unittest.main()
