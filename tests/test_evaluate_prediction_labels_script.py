import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_obb_prediction_labels.py"


def load_script():
    spec = importlib.util.spec_from_file_location("evaluate_obb_prediction_labels_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluatePredictionLabelsScriptTests(unittest.TestCase):
    def test_main_defaults_to_ultralytics_metrics_csv(self) -> None:
        module = load_script()
        fake_result = object()

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "custom_metrics.csv"
            argv = [
                "evaluate_obb_prediction_labels.py",
                "--data",
                "dataset.yaml",
                "--pred-labels",
                "pred_labels",
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv), \
                patch.object(module, "resolve_from_root", side_effect=lambda path, root: Path(path)), \
                patch.object(module, "evaluate_prediction_labels_ultralytics", return_value=fake_result) as evaluate, \
                patch.object(module, "write_custom_metrics_csv", return_value=output) as write_csv, \
                patch.object(module, "build_custom_map_rows", return_value=[{
                    "class": "all",
                    "precision": 1.0,
                    "recall": 1.0,
                    "mAP50": 1.0,
                    "mAP80": 1.0,
                    "mAP85": 1.0,
                    "mAP90": 1.0,
                    "mAP95": 1.0,
                }]):
                module.main()

            evaluate.assert_called_once()
            write_csv.assert_called_once_with(fake_result, output)


if __name__ == "__main__":
    unittest.main()
