from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.prediction_label_eval import (
    evaluate_prediction_labels,
    evaluate_prediction_labels_ultralytics,
)


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

    def test_ultralytics_metric_eval_builds_obb_metrics_from_prediction_labels(self) -> None:
        class FakeMetrics:
            def __init__(self, names):
                self.names = names
                self.stats = []
                self.process_args = None

            def update_stats(self, stat):
                self.stats.append(stat)

            def process(self, save_dir, plot=False):
                self.process_args = (save_dir, plot)

        class FakeBackend:
            def __init__(self):
                self.metrics = None
                self.probiou_inputs = None

            def create_metrics(self, names):
                self.metrics = FakeMetrics(names)
                return self.metrics

            def probiou(self, gt_xywhr, pred_xywhr):
                self.probiou_inputs = (gt_xywhr, pred_xywhr)
                return np.ones((len(gt_xywhr), len(pred_xywhr)), dtype=np.float32)

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
                "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6 0.77\n",
                encoding="utf-8",
            )
            backend = FakeBackend()
            save_dir = root / "eval"

            result = evaluate_prediction_labels_ultralytics(
                data_yaml=dataset / "data.yaml",
                pred_labels=pred,
                split="test",
                save_dir=save_dir,
                metric_backend=backend,
            )

            self.assertIs(result, backend.metrics)
            self.assertEqual(result.names, {0: "label1"})
            self.assertEqual(result.process_args, (save_dir, False))
            self.assertEqual(len(result.stats), 1)
            stat = result.stats[0]
            np.testing.assert_array_equal(stat["target_cls"], np.array([0.0], dtype=np.float32))
            np.testing.assert_array_equal(stat["pred_cls"], np.array([0.0], dtype=np.float32))
            np.testing.assert_allclose(stat["conf"], np.array([0.77], dtype=np.float32))
            self.assertEqual(stat["tp"].shape, (1, 10))
            self.assertTrue(stat["tp"].all())
            self.assertEqual(stat["im_name"], f"{stem}.bmp")
            gt_xywhr, pred_xywhr = backend.probiou_inputs
            self.assertEqual(gt_xywhr.shape, (1, 5))
            self.assertEqual(pred_xywhr.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
