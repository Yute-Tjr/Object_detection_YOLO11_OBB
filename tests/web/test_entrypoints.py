import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_api_script_uses_import_string_when_reload_is_enabled(self):
        script = ROOT / "scripts/run_terminal_api.py"
        with (
            patch.object(sys, "argv", [str(script), "--reload"]),
            patch("uvicorn.run") as run_server,
        ):
            runpy.run_path(str(script), run_name="__main__")

        run_server.assert_called_once_with(
            "terminal_web.api.app:create_app",
            factory=True,
            host="0.0.0.0",
            port=8000,
            reload=True,
        )

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
