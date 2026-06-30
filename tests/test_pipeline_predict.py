import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.pipeline_predict import (
    PipelineDetection,
    final_result_from_selected,
    format_detection_row,
    selected_detection_by_label,
    summary_row_for_image,
)


POINTS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


class PipelinePredictTests(unittest.TestCase):
    def test_selected_detection_by_label_chooses_highest_confidence(self) -> None:
        detections = [
            PipelineDetection(Path("a.png"), "a.png", 0, "label3", 0.30, POINTS),
            PipelineDetection(Path("a.png"), "a.png", 1, "label3", 0.90, POINTS),
        ]

        selected, warnings = selected_detection_by_label(detections, ["label3", "label5"])

        self.assertEqual(selected["label3"].det_index, 1)
        self.assertIn("duplicate label3 count=2", warnings)
        self.assertIn("missing label5", warnings)

    def test_final_result_from_selected_marks_ng_ok_and_unknown(self) -> None:
        self.assertEqual(final_result_from_selected({"label3": "OK", "label5": "OK"}), "OK")
        self.assertEqual(final_result_from_selected({"label3": "NG", "label5": "OK"}), "NG")
        self.assertEqual(final_result_from_selected({"label3": "OK"}), "UNKNOWN")

    def test_format_detection_row_includes_empty_classification_fields(self) -> None:
        detection = PipelineDetection(Path("a.png"), "a.png", 0, "label2", 0.75, POINTS)

        row = format_detection_row(detection)

        self.assertEqual(row["det_label"], "label2")
        self.assertEqual(row["classifier"], "")
        self.assertEqual(row["prob_NG"], "")

    def test_summary_row_for_image_uses_selected_predictions(self) -> None:
        label3 = PipelineDetection(Path("a.png"), "a.png", 0, "label3", 0.80, POINTS)
        label3.predicted_label = "OK"
        label3.cls_confidence = 0.90
        label5 = PipelineDetection(Path("a.png"), "a.png", 1, "label5", 0.70, POINTS)
        label5.predicted_label = "NG"
        label5.cls_confidence = 0.95

        row = summary_row_for_image(Path("a.png"), [label3, label5], Path("vis/a.png"))

        self.assertEqual(row["final_result"], "NG")
        self.assertEqual(row["label3_pred"], "OK")
        self.assertEqual(row["label5_pred"], "NG")


if __name__ == "__main__":
    unittest.main()
