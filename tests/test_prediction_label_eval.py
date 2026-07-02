from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.prediction_label_eval import evaluate_prediction_labels


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


class PredictionLabelEvalTests(unittest.TestCase):
    def test_evaluate_prediction_labels_reports_custom_iou_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            pred = root / "pred"
            pred.mkdir()
            stem = "sample"
            write_image(dataset / "images" / "test" / f"{stem}.bmp")
            (dataset / "labels" / "test").mkdir(parents=True)
            (dataset / "labels" / "test" / f"{stem}.txt").write_text(
                "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n",
                encoding="utf-8",
            )
            (dataset / "data.yaml").write_text(
                "\n".join(
                    [
                        "path: .",
                        "train: images/test",
                        "val: images/test",
                        "test: images/test",
                        "names:",
                        "  0: label1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (pred / f"{stem}.txt").write_text(
                "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6 0.99\n",
                encoding="utf-8",
            )

            rows = evaluate_prediction_labels(
                data_yaml=dataset / "data.yaml",
                pred_labels=pred,
                split="test",
            )

            self.assertEqual(rows[0]["class"], "all")
            self.assertAlmostEqual(rows[0]["mAP90"], 1.0)
            self.assertAlmostEqual(rows[1]["mAP90"], 1.0)


if __name__ == "__main__":
    unittest.main()
