import tempfile
import unittest
from pathlib import Path

from yolo11_obb.dataset_filter import create_label_subset_dataset


def write_file(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class DatasetFilterTests(unittest.TestCase):
    def test_create_label_subset_dataset_keeps_only_requested_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"

            for split in ("train", "val", "test"):
                write_file(source / "images" / split / "sample.bmp")
                write_file(
                    source / "labels" / split / "sample.txt",
                    "\n".join(
                        [
                            "0 0 0 1 0 1 1 0 1",
                            "5 0 0 1 0 1 1 0 1",
                            "6 0 0 1 0 1 1 0 1",
                            "8 0 0 1 0 1 1 0 1",
                        ]
                    )
                    + "\n",
                )

            report = create_label_subset_dataset(
                source=source,
                output=output,
                keep_classes={0, 1, 2, 3, 4, 5},
                names={i: f"label{i + 1}" for i in range(6)},
            )

            self.assertEqual(report["train"]["kept_objects"], 2)
            self.assertEqual(report["train"]["removed_objects"], 2)
            self.assertTrue((output / "images" / "train" / "sample.bmp").exists())
            self.assertEqual(
                (output / "labels" / "train" / "sample.txt").read_text(encoding="utf-8"),
                "0 0 0 1 0 1 1 0 1\n5 0 0 1 0 1 1 0 1\n",
            )
            self.assertIn("path: .", (output / "data.yaml").read_text(encoding="utf-8"))
            self.assertIn("  5: label6", (output / "data.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("  6:", (output / "data.yaml").read_text(encoding="utf-8"))

    def test_create_label_subset_dataset_writes_project_relative_path_under_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "project" / "datasets" / "sample_label_subset"

            for split in ("train", "val", "test"):
                write_file(source / "images" / split / "sample.bmp")
                write_file(source / "labels" / split / "sample.txt", "0 0 0 1 0 1 1 0 1\n")

            create_label_subset_dataset(
                source=source,
                output=output,
                keep_classes={0},
                names={0: "label1"},
            )

            self.assertIn(
                "path: datasets/sample_label_subset",
                (output / "data.yaml").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
