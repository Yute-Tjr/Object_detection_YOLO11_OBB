import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class SmokeScriptTest(unittest.TestCase):
    def test_missing_weight_exits_nonzero_and_names_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / "existing.pt"
            image = root / "image.png"
            existing.write_bytes(b"placeholder")
            image.write_bytes(b"placeholder")
            for missing_flag, expected_name in (
                ("--detector", "detector"),
                ("--label3", "label3"),
                ("--label5", "label5"),
            ):
                paths = {
                    "--detector": existing,
                    "--label3": existing,
                    "--label5": existing,
                }
                paths[missing_flag] = root / "missing.pt"
                command = [
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/smoke_terminal_pipeline.py"),
                    "--detector",
                    str(paths["--detector"]),
                    "--label3",
                    str(paths["--label3"]),
                    "--label5",
                    str(paths["--label5"]),
                    "--image",
                    str(image),
                    "--output",
                    str(root / "output"),
                ]

                result = subprocess.run(command, capture_output=True, text=True)

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_name, result.stderr)
