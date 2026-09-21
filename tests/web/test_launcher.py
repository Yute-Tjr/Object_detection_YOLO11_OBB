import gc
import io
import importlib
import os
import sys
import tempfile
import time
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

try:
    launcher = importlib.import_module("terminal_web.launcher")
except ModuleNotFoundError:
    launcher = None


class TerminalLauncherTest(unittest.TestCase):
    def test_launcher_module_is_available(self):
        self.assertIsNotNone(launcher, "一键启动器尚未实现")

    @unittest.skipIf(launcher is None, "一键启动器尚未实现")
    def test_run_check_prints_a_chinese_success_message(self):
        output = io.StringIO()

        with redirect_stdout(output):
            detail = launcher.run_check("PostgreSQL 连接", lambda: "连接正常")

        self.assertEqual(detail, "连接正常")
        self.assertIn("[通过] PostgreSQL 连接：连接正常", output.getvalue())

    @unittest.skipIf(launcher is None, "一键启动器尚未实现")
    def test_describe_devices_reports_cpu_for_both_models(self):
        summary = launcher.describe_devices(
            detection_device="cpu",
            classification_device="cpu",
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_names=(),
            mps_available=False,
        )

        self.assertEqual(summary, "检测模型=CPU；分类模型=CPU")

    @unittest.skipIf(launcher is None, "一键启动器尚未实现")
    def test_describe_devices_rejects_cuda_when_cuda_is_unavailable(self):
        with self.assertRaisesRegex(launcher.LauncherError, "CUDA"):
            launcher.describe_devices(
                detection_device="0",
                classification_device="cpu",
                cuda_available=False,
                cuda_device_count=0,
                cuda_device_names=(),
                mps_available=False,
            )

    @unittest.skipIf(launcher is None, "一键启动器尚未实现")
    def test_weight_and_frontend_checks_accept_complete_project_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("DATABASE_URL=test\n", encoding="utf-8")
            frontend = root / "web_frontend"
            (frontend / "node_modules" / ".bin").mkdir(parents=True)
            (frontend / "package.json").write_text("{}", encoding="utf-8")
            (frontend / "node_modules" / ".bin" / "vite").write_text("", encoding="utf-8")
            weight_paths = []
            for relative in (
                "weights/detector/model.pt",
                "weights/classifiers/label3/model.pt",
                "weights/classifiers/label5/model.pt",
            ):
                weight = root / relative
                weight.parent.mkdir(parents=True, exist_ok=True)
                weight.write_bytes(b"weight")
                weight_paths.append(weight)
            settings = SimpleNamespace(
                detector_weights=weight_paths[0],
                label3_classifier_weights=weight_paths[1],
                label5_classifier_weights=weight_paths[2],
            )

            weight_detail = launcher.check_weights(settings)
            frontend_detail = launcher.check_frontend_dependencies(
                root,
                npm_path="/usr/bin/npm",
                node_version="24.20.0",
            )

        self.assertIn("3 个模型权重", weight_detail)
        self.assertIn("Vite 依赖完整", frontend_detail)

    def test_frontend_check_rejects_node_14(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontend = root / "web_frontend"
            (frontend / "node_modules" / ".bin").mkdir(parents=True)
            (frontend / "package.json").write_text("{}", encoding="utf-8")
            (frontend / "node_modules" / ".bin" / "vite").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(launcher.LauncherError, "Node.js 18"):
                launcher.check_frontend_dependencies(
                    root,
                    npm_path="/opt/node/bin/npm",
                    node_version="14.21.3",
                )

    def test_web_service_prefers_node_next_to_selected_npm(self):
        services = launcher.build_service_specs(
            Path("/tmp/project"),
            host="127.0.0.1",
            api_port=8000,
            web_port=5173,
            python_executable=sys.executable,
            npm_path="/opt/node24/bin/npm",
        )

        web_path = services[2].env["PATH"].split(os.pathsep)
        self.assertEqual(web_path[0], "/opt/node24/bin")

    @unittest.skipIf(launcher is None, "一键启动器尚未实现")
    def test_supervisor_prefixes_logs_and_stops_other_services_after_failure(self):
        messages = []
        services = (
            launcher.ServiceSpec(
                name="API",
                prefix="[API]",
                command=(
                    sys.executable,
                    "-c",
                    "import sys; print('启动后失败', flush=True); sys.exit(7)",
                ),
                cwd=Path.cwd(),
            ),
            launcher.ServiceSpec(
                name="WORKER",
                prefix="[WORKER]",
                command=(sys.executable, "-c", "import time; time.sleep(30)"),
                cwd=Path.cwd(),
            ),
        )

        started = time.monotonic()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ResourceWarning)
            exit_code = launcher.supervise_services(
                services, printer=messages.append, poll_interval=0.02
            )
            gc.collect()
        elapsed = time.monotonic() - started

        self.assertEqual(exit_code, 7)
        self.assertLess(elapsed, 5)
        self.assertIn("[API] 启动后失败", messages)
        self.assertTrue(any("API 异常退出" in message for message in messages))
        self.assertTrue(any("正在停止其余服务" in message for message in messages))
        self.assertEqual(
            [warning for warning in captured if warning.category is ResourceWarning],
            [],
        )


if __name__ == "__main__":
    unittest.main()
