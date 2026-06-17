import tempfile
import unittest
from pathlib import Path

from yolo11_obb.eval_metrics import build_custom_map_rows, write_custom_metrics_csv


class FakeBoxMetrics:
    def __init__(self) -> None:
        self.ap_class_index = [0, 1]
        self.p = [0.91, 0.82]
        self.r = [0.73, 0.64]
        self.all_ap = [
            [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        ]


class FakeMetricsResult:
    def __init__(self) -> None:
        self.names = {0: "label1", 1: "label2"}
        self.box = FakeBoxMetrics()


class EvalMetricsTests(unittest.TestCase):
    def test_build_custom_map_rows_extracts_requested_iou_thresholds(self) -> None:
        rows = build_custom_map_rows(FakeMetricsResult())

        self.assertEqual([row["class"] for row in rows], ["all", "label1", "label2"])
        self.assertAlmostEqual(rows[0]["precision"], 0.865)
        self.assertAlmostEqual(rows[0]["recall"], 0.685)
        self.assertAlmostEqual(rows[0]["mAP50"], 0.45)
        self.assertAlmostEqual(rows[0]["mAP80"], 0.75)
        self.assertAlmostEqual(rows[0]["mAP85"], 0.80)
        self.assertAlmostEqual(rows[0]["mAP90"], 0.85)
        self.assertAlmostEqual(rows[0]["mAP95"], 0.90)
        self.assertAlmostEqual(rows[1]["mAP95"], 0.95)
        self.assertAlmostEqual(rows[2]["mAP80"], 0.70)

    def test_write_custom_metrics_csv_omits_map50_95_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom_metrics.csv"

            written = write_custom_metrics_csv(FakeMetricsResult(), path)

            self.assertEqual(written, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("class,precision,recall,mAP50,mAP80,mAP85,mAP90,mAP95\n", text)
            self.assertNotIn("mAP50-95", text)
            self.assertIn("label1,0.910000,0.730000,0.500000", text)


if __name__ == "__main__":
    unittest.main()
