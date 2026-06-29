import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.classification_training import (
    classification_metrics,
    confusion_counts,
    discover_classes,
)


class ClassificationTrainingTests(unittest.TestCase):
    def test_discover_classes_sorts_expected_class_names(self) -> None:
        self.assertEqual(discover_classes(["NG", "OK"]), {"NG": 0, "OK": 1})

    def test_confusion_counts_counts_pairs(self) -> None:
        counts = confusion_counts(["OK", "OK", "NG", "NG"], ["OK", "NG", "NG", "OK"], ["NG", "OK"])

        self.assertEqual(
            counts,
            {
                ("NG", "NG"): 1,
                ("NG", "OK"): 1,
                ("OK", "NG"): 1,
                ("OK", "OK"): 1,
            },
        )

    def test_classification_metrics_returns_macro_f1(self) -> None:
        metrics = classification_metrics(["OK", "OK", "NG", "NG"], ["OK", "NG", "NG", "OK"], ["NG", "OK"])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["per_class"]["OK"]["precision"], 0.5)
        self.assertEqual(metrics["per_class"]["NG"]["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
