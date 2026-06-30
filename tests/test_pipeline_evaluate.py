import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.pipeline_evaluate import (
    PIPELINE_EVAL_FIELDNAMES,
    evaluate_pipeline_summary_rows,
    resolve_split_source,
    truth_from_sheet_rows_by_label,
    write_pipeline_evaluation,
)


class PipelineEvaluateTests(unittest.TestCase):
    def test_truth_from_sheet_rows_by_label_normalizes_ok_ng(self) -> None:
        truth = truth_from_sheet_rows_by_label(
            {
                "label3": [
                    {"images_name": "a-3.bmp", "tag1": "ok"},
                    {"images_name": "b-3.bmp", "tag1": "NG"},
                    {"images_name": "skip-3.bmp", "tag1": "OTHER"},
                ],
                "label5": [
                    {"images_name": "a-5.bmp", "tag1": "NG"},
                ],
            },
            target_column="tag1",
        )

        self.assertEqual(truth["label3"], {"a-3": "OK", "b-3": "NG"})
        self.assertEqual(truth["label5"], {"a-5": "NG"})

    def test_truth_from_sheet_rows_by_label_rejects_conflicting_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting truth labels"):
            truth_from_sheet_rows_by_label(
                {
                    "label3": [
                        {"images_name": "a-3.bmp", "tag1": "OK"},
                        {"images_name": "a-3.bmp", "tag1": "NG"},
                    ],
                },
                target_column="tag1",
            )

    def test_evaluate_pipeline_summary_rows_counts_missing_prediction_as_wrong(self) -> None:
        rows = [
            {
                "image_path": "/tmp/a-3.bmp",
                "image_name": "a-3.bmp",
                "label3_pred": "OK",
                "label5_pred": "NG",
                "final_result": "NG",
                "warnings": "",
                "visualization_path": "visualizations/a.jpg",
            },
            {
                "image_path": "/tmp/b-3.bmp",
                "image_name": "b-3.bmp",
                "label3_pred": "",
                "label5_pred": "OK",
                "final_result": "UNKNOWN",
                "warnings": "missing label3",
                "visualization_path": "visualizations/b.jpg",
            },
        ]
        truth = {
            "label3": {"a-3": "OK", "b-3": "NG"},
            "label5": {"a-3": "NG", "b-3": "OK"},
        }

        result = evaluate_pipeline_summary_rows(rows, truth)

        self.assertEqual(result.metrics["label3"]["total"], 2)
        self.assertEqual(result.metrics["label3"]["correct"], 1)
        self.assertEqual(result.metrics["label3"]["unknown_predictions"], 1)
        self.assertEqual(result.metrics["label5"]["accuracy"], 1.0)
        self.assertEqual(result.metrics["both_labels"]["accuracy"], 0.5)
        self.assertEqual(result.metrics["final_result"]["accuracy"], 0.5)
        self.assertEqual(result.rows[1]["pred_label3"], "UNKNOWN")
        self.assertEqual(result.rows[1]["label3_correct"], "0")

    def test_write_pipeline_evaluation_outputs_metrics_predictions_and_confusion(self) -> None:
        rows = [
            {
                "image_path": "/tmp/a-3.bmp",
                "image_name": "a-3.bmp",
                "label3_pred": "OK",
                "label5_pred": "NG",
                "final_result": "NG",
                "warnings": "",
                "visualization_path": "visualizations/a.jpg",
            },
        ]
        truth = {
            "label3": {"a-3": "OK"},
            "label5": {"a-3": "NG"},
        }
        result = evaluate_pipeline_summary_rows(rows, truth)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_pipeline_evaluation(output, result)

            with (output / "pipeline_eval_predictions.csv").open(encoding="utf-8") as handle:
                prediction_rows = list(csv.DictReader(handle))
            with (output / "pipeline_eval_metrics.csv").open(encoding="utf-8") as handle:
                metric_rows = list(csv.DictReader(handle))

            self.assertEqual(prediction_rows[0]["both_labels_correct"], "1")
            self.assertEqual(metric_rows[0]["target"], "label3")
            self.assertEqual(set(prediction_rows[0]), set(PIPELINE_EVAL_FIELDNAMES))
            self.assertTrue((output / "pipeline_label3_confusion_matrix.csv").exists())
            self.assertTrue((output / "pipeline_eval_report.txt").exists())

    def test_resolve_split_source_reads_test_path_from_data_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            (dataset / "images" / "test").mkdir(parents=True)
            data_yaml = dataset / "data.yaml"
            data_yaml.write_text(
                "\n".join(
                    [
                        "path: .",
                        "train: images/train",
                        "val: images/test",
                        "test: images/test",
                        "names:",
                        "  0: label3",
                        "  1: label5",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(resolve_split_source(data_yaml, "test"), (dataset / "images" / "test").resolve())


if __name__ == "__main__":
    unittest.main()
