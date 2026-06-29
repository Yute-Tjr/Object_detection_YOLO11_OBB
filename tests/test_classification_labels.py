import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.classification_labels import (
    LabelSample,
    normalize_status,
    parent_group_key,
    rows_to_label_samples,
    split_samples_by_group,
    write_manifest_csv,
)


class ClassificationLabelsTests(unittest.TestCase):
    def test_normalize_status_accepts_ok_and_ng_only(self) -> None:
        self.assertEqual(normalize_status("OK"), "OK")
        self.assertEqual(normalize_status("ok"), "OK")
        self.assertEqual(normalize_status("NG"), "NG")
        self.assertEqual(normalize_status(" ng "), "NG")
        self.assertIsNone(normalize_status(""))
        self.assertIsNone(normalize_status(None))
        self.assertIsNone(normalize_status("OTHER"))

    def test_parent_group_key_strips_trailing_child_index(self) -> None:
        self.assertEqual(
            parent_group_key("CropImage_20260126092714769_F3-I0_OK-5"),
            "CropImage_20260126092714769_F3-I0_OK",
        )
        self.assertEqual(parent_group_key("sample_without_child"), "sample_without_child")

    def test_rows_to_label_samples_filters_invalid_statuses(self) -> None:
        rows = [
            {"images_name": "parent-2", "tag1": "OK"},
            {"images_name": "parent-3", "tag1": "NG"},
            {"images_name": "parent-4", "tag1": "OTHER"},
            {"images_name": "", "tag1": "OK"},
        ]

        samples = rows_to_label_samples(rows, target_column="tag1")

        self.assertEqual(
            samples,
            [
                LabelSample(image_name="parent-2", class_name="OK", group="parent"),
                LabelSample(image_name="parent-3", class_name="NG", group="parent"),
            ],
        )

    def test_split_samples_by_group_keeps_related_images_together(self) -> None:
        samples = [
            LabelSample("group-a-2", "OK", "group-a"),
            LabelSample("group-a-3", "NG", "group-a"),
            LabelSample("group-b-2", "OK", "group-b"),
            LabelSample("group-c-2", "NG", "group-c"),
            LabelSample("group-d-2", "OK", "group-d"),
        ]

        split = split_samples_by_group(samples, train_ratio=0.5, seed=7)
        train_groups = {sample.group for sample in split["train"]}
        test_groups = {sample.group for sample in split["test"]}

        self.assertTrue(train_groups)
        self.assertTrue(test_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(len(split["train"]) + len(split["test"]), len(samples))

    def test_write_manifest_csv_writes_header_and_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            rows = [
                {
                    "image_name": "sample-2",
                    "class_name": "OK",
                    "group": "sample",
                    "split": "train",
                    "source_json": "sample-2.json",
                    "source_image": "sample-2.bmp",
                    "crop_path": "images/train/OK/sample-2.png",
                }
            ]

            write_manifest_csv(path, rows)

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    "image_name,class_name,group,split,source_json,source_image,crop_path",
                    "sample-2,OK,sample,train,sample-2.json,sample-2.bmp,images/train/OK/sample-2.png",
                ],
            )


if __name__ == "__main__":
    unittest.main()
