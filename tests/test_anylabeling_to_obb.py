import json
import tempfile
import unittest
from pathlib import Path

from yolo11_obb.anylabeling_to_obb import convert_anylabeling_directory, shape_to_obb_line


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


class AnyLabelingToObbTests(unittest.TestCase):
    def test_shape_to_obb_line_normalizes_four_points_and_maps_label(self) -> None:
        shape = {
            "label": "label2",
            "points": [
                [20, 10],
                [100, 10],
                [100, 30],
                [20, 30],
            ],
        }

        line = shape_to_obb_line(shape, image_width=200, image_height=100)

        self.assertEqual(line, "1 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3")

    def test_shape_to_obb_line_clips_coordinates_by_default(self) -> None:
        shape = {
            "label": "label1",
            "points": [
                [-20, 10],
                [220, 10],
                [220, 110],
                [-20, 110],
            ],
        }

        clipped = shape_to_obb_line(shape, image_width=200, image_height=100)
        unclipped = shape_to_obb_line(shape, image_width=200, image_height=100, clip=False)

        self.assertEqual(clipped, "0 0 0.1 1 0.1 1 1 0 1")
        self.assertEqual(unclipped, "0 -0.1 0.1 1.1 0.1 1.1 1.1 -0.1 1.1")

    def test_convert_anylabeling_directory_writes_labels_and_copies_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "anylabeling"
            output = root / "obb"
            image = source / "sample.bmp"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"fake-bmp")
            write_json(
                source / "sample.json",
                {
                    "imagePath": "sample.bmp",
                    "imageWidth": 200,
                    "imageHeight": 100,
                    "shapes": [
                        {
                            "label": "label1",
                            "shape_type": "rotation",
                            "points": [[0, 0], [100, 0], [100, 50], [0, 50]],
                        },
                        {
                            "label": "other",
                            "shape_type": "polygon",
                            "points": [[20, 10], [40, 10], [40, 30], [20, 30]],
                        },
                    ],
                },
            )

            report = convert_anylabeling_directory(source=source, output=output)

            self.assertEqual(report.json_files, 1)
            self.assertEqual(report.objects, 2)
            self.assertEqual((output / "sample.bmp").read_bytes(), b"fake-bmp")
            self.assertEqual(
                (output / "sample.txt").read_text(encoding="utf-8"),
                "\n".join(
                    [
                        "0 0 0 0.5 0 0.5 0.5 0 0.5",
                        "8 0.1 0.1 0.2 0.1 0.2 0.3 0.1 0.3",
                    ]
                )
                + "\n",
            )


if __name__ == "__main__":
    unittest.main()
