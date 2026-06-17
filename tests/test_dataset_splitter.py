import csv
import tempfile
import unittest
from pathlib import Path

from yolo11_obb.dataset_splitter import create_train_test_dataset, parent_group_key


def write_file(path: Path, content: bytes = b"image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class DatasetSplitterTests(unittest.TestCase):
    def test_parent_group_key_drops_final_crop_index(self) -> None:
        self.assertEqual(
            parent_group_key("CropImage_20260121113025265_F3-I0_NG-7"),
            "CropImage_20260121113025265_F3-I0_NG",
        )

    def test_create_train_test_dataset_groups_samples_and_filters_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            for group in range(4):
                for crop in range(2):
                    stem = f"CropImage_2026010100000000{group}_F3-I0_NG-{crop + 1}"
                    write_file(source / f"{stem}.bmp", content=stem.encode("utf-8"))
                    write_text(
                        source / f"{stem}.txt",
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

            report = create_train_test_dataset(
                source=source,
                output=output,
                train_ratio=0.5,
                seed=1,
                keep_classes={0, 1, 2, 3, 4, 5},
                names={i: f"label{i + 1}" for i in range(6)},
            )

            self.assertEqual(report["train"]["groups"] + report["test"]["groups"], 4)
            self.assertEqual(report["train"]["images"] + report["test"]["images"], 8)
            self.assertEqual(report["train"]["removed_objects"] + report["test"]["removed_objects"], 16)
            self.assertFalse((output / "images" / "val").exists())
            self.assertFalse((output / "labels" / "val").exists())
            self.assertIn("val: images/test", (output / "data.yaml").read_text(encoding="utf-8"))
            self.assertIn("  5: label6", (output / "data.yaml").read_text(encoding="utf-8"))
            self.assertNotIn("  6:", (output / "data.yaml").read_text(encoding="utf-8"))

            split_by_group = {}
            with (output / "split_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    split_by_group.setdefault(row["group"], set()).add(row["split"])
                    label_text = (output / "labels" / row["split"] / row["label"]).read_text(
                        encoding="utf-8"
                    )
                    self.assertEqual(label_text, "0 0 0 1 0 1 1 0 1\n5 0 0 1 0 1 1 0 1\n")

            self.assertTrue(all(len(splits) == 1 for splits in split_by_group.values()))


if __name__ == "__main__":
    unittest.main()
