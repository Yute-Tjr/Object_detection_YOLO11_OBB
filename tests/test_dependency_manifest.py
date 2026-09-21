import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_requirements() -> list[str]:
    return [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


class DependencyManifestTest(unittest.TestCase):
    def test_project_uses_one_python_requirements_file(self):
        self.assertFalse((ROOT / "requirements-web.txt").exists())

    def test_pyproject_dependencies_match_requirements(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["dependencies"], load_requirements())

    def test_uv_manages_a_python_311_application_environment(self):
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["requires-python"], ">=3.11,<3.12")
        self.assertFalse(pyproject["tool"]["uv"]["package"])

    def test_active_setup_files_do_not_reference_removed_web_requirements(self):
        active_files = (
            ROOT / "README.md",
            ROOT / "deploy" / "api.Dockerfile",
            ROOT / "docs" / "web-deployment.md",
            ROOT / "terminal_web" / "launcher.py",
        )

        for path in active_files:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertNotIn(
                    "requirements-web.txt",
                    path.read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
