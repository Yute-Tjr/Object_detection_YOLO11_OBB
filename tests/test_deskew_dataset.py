from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.deskew_dataset import create_deskewed_obb_dataset, signed_top_edge_angle
from yolo11_obb.obb_geometry import parse_obb_line


def write_image(path: Path, width: int = 120, height: int = 160) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


def rotated_rect_points(
    center: tuple[float, float],
    width: float,
    height: float,
    angle_degrees: float,
) -> list[tuple[float, float]]:
    cx, cy = center
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    corners = [
        (-width / 2, -height / 2),
        (width / 2, -height / 2),
        (width / 2, height / 2),
        (-width / 2, height / 2),
    ]
    return [
        (cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a)
        for x, y in corners
    ]


def yolo_obb_line(class_id: int, points: list[tuple[float, float]], width: int, height: int) -> str:
    values = [str(class_id)]
    for x, y in points:
        values.extend([f"{x / width:.6f}", f"{y / height:.6f}"])
    return " ".join(values)


class DeskewDatasetTests(unittest.TestCase):
    def test_signed_top_edge_angle_normalizes_left_to_right_orientation(self) -> None:
        points = ((80.0, 40.0), (20.0, 50.0), (25.0, 80.0), (85.0, 70.0))

        self.assertAlmostEqual(signed_top_edge_angle(points), -9.462322, places=5)

    def test_create_deskewed_dataset_rotates_labels_to_horizontal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "project" / "datasets" / "tilted"
            output = root / "project" / "datasets" / "tilted_deskewed"
            image_width = 120
            image_height = 160
            stem = "sample"

            write_image(source / "images" / "train" / f"{stem}.bmp", image_width, image_height)
            write_image(source / "images" / "test" / f"{stem}.bmp", image_width, image_height)
            points = rotated_rect_points((60.0, 80.0), 50.0, 30.0, 12.0)
            label = yolo_obb_line(2, points, image_width, image_height)
            (source / "labels" / "train").mkdir(parents=True)
            (source / "labels" / "test").mkdir(parents=True)
            (source / "labels" / "train" / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
            (source / "labels" / "test" / f"{stem}.txt").write_text(label + "\n", encoding="utf-8")
            (source / "data.yaml").write_text(
                "\n".join(
                    [
                        "path: .",
                        "train: images/train",
                        "val: images/test",
                        "test: images/test",
                        "names:",
                        "  2: label2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = create_deskewed_obb_dataset(
                source_data=source / "data.yaml",
                output=output,
                angle_class_ids={2},
            )

            self.assertEqual(report["train"]["images"], 1)
            self.assertEqual(report["test"]["images"], 1)
            self.assertTrue((output / "images" / "train" / f"{stem}.bmp").exists())
            self.assertIn("path: datasets/tilted_deskewed", (output / "data.yaml").read_text(encoding="utf-8"))

            output_image = cv2.imread(str(output / "images" / "train" / f"{stem}.bmp"))
            self.assertIsNotNone(output_image)
            out_height, out_width = output_image.shape[:2]
            output_label = (output / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8").strip()
            output_box = parse_obb_line(output_label, image_width=out_width, image_height=out_height)

            self.assertAlmostEqual(signed_top_edge_angle(output_box.points), 0.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
