from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from yolo11_obb.prediction_label_eval import (
    evaluate_prediction_labels_ultralytics,
    match_ground_truths_ultralytics,
)
from yolo11_obb.obb_geometry import ObbBox


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")


class PredictionLabelEvalTests(unittest.TestCase):
    def test_match_ground_truths_ultralytics_uses_probiou_backend(self) -> None:
        class FakeBackend:
            def __init__(self):
                self.probiou_inputs = None

            def probiou(self, gt_xywhr, pred_xywhr):
                self.probiou_inputs = (gt_xywhr, pred_xywhr)
                return np.array([[0.95, 0.20]], dtype=np.float32)

        gt = ObbBox(
            class_id=0,
            points=((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)),
        )
        same_class = ObbBox(
            class_id=0,
            points=((2.0, 0.0), (22.0, 0.0), (22.0, 10.0), (2.0, 10.0)),
            confidence=0.90,
        )
        wrong_class = ObbBox(
            class_id=1,
            points=((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)),
            confidence=0.99,
        )
        backend = FakeBackend()

        matches = match_ground_truths_ultralytics(
            [gt],
            [same_class, wrong_class],
            iou_threshold=0.90,
            metric_backend=backend,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].prediction, same_class)
        self.assertAlmostEqual(matches[0].iou, 0.95)
        self.assertTrue(matches[0].passed)
        gt_xywhr, pred_xywhr = backend.probiou_inputs
        self.assertEqual(gt_xywhr.shape, (1, 5))
        self.assertEqual(pred_xywhr.shape, (2, 5))

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
