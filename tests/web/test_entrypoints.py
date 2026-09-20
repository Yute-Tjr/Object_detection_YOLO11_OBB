import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ScriptEntrypointTest(unittest.TestCase):
    def test_api_script_can_import_project_when_run_directly(self):
        result = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "scripts/run_terminal_api.py", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Run the terminal inspection API", result.stdout)

    def test_worker_script_can_import_project_when_run_directly(self):
        result = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-c",
                "import runpy; runpy.run_path('scripts/run_terminal_worker.py', run_name='entrypoint_import_test')",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
