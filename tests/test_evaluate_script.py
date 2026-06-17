import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_yolo11_obb.py"


def load_evaluate_script():
    spec = importlib.util.spec_from_file_location("evaluate_yolo11_obb_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluateScriptTests(unittest.TestCase):
    def test_main_writes_custom_metrics_after_evaluate(self) -> None:
        module = load_evaluate_script()
        fake_result = object()
        fake_dataset = object()

        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "custom_metrics.csv"
            argv = [
                "evaluate_yolo11_obb.py",
                "--model",
                "weights.pt",
                "--data",
                "dataset.yaml",
            ]
            with patch.object(sys, "argv", argv), \
                patch.object(module, "resolve_from_root", side_effect=lambda path, root: Path(path)), \
                patch.object(module, "load_dataset_config", return_value=fake_dataset), \
                patch.object(module, "validate_dataset_layout"), \
                patch.object(module, "evaluate", return_value=fake_result), \
                patch.object(module, "default_custom_metrics_path", return_value=metrics_path, create=True), \
                patch.object(module, "write_custom_metrics_csv", return_value=metrics_path, create=True) as write_csv:
                module.main()

            write_csv.assert_called_once_with(fake_result, metrics_path)


if __name__ == "__main__":
    unittest.main()
