import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy.sh"


def run_bash(source: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", source, "test", *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def prepare_fake_project(root: Path) -> None:
    shutil.copy2(ROOT / "compose.yaml", root / "compose.yaml")
    shutil.copy2(ROOT / "compose.gpu.yaml", root / "compose.gpu.yaml")
    shutil.copy2(SCRIPT, root / "deploy.sh")
    for relative in (
        "weights/detector/yolo11l_obb_best.pt",
        "weights/classifiers/label3/resnet18_best.pt",
        "weights/classifiers/label5/resnet18_best.pt",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weight")


def fake_docker_environment(root: Path, *, volume_exists: bool) -> dict[str, str]:
    binary_dir = root / "fake-bin"
    binary_dir.mkdir()
    docker = binary_dir / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${FAKE_DOCKER_LOG}"
if [[ "$1 $2" == "compose version" ]]; then
    echo "Docker Compose version v2.35.1"
    exit 0
fi
if [[ "$1 $2" == "volume inspect" ]]; then
    if [[ "${FAKE_VOLUME_INSPECT_ERROR:-false}" == "true" ]]; then
        echo "permission denied while inspecting volume" >&2
        exit 1
    fi
    if [[ "${FAKE_VOLUME_EXISTS:-false}" == "true" ]]; then
        exit 0
    fi
    echo "Error response from daemon: no such volume" >&2
    exit 1
fi
if [[ "$1" == "inspect" ]]; then
    echo "healthy"
    exit 0
fi
if [[ " $* " == *" ps -q "* ]]; then
    echo "fake-container-id"
    exit 0
fi
if [[ " $* " == *" exec -T postgres pg_dump "* ]]; then
    echo "fake-database-dump"
    exit 0
fi
exit 0
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{binary_dir}{os.pathsep}{env['PATH']}",
            "DEPLOY_PROJECT_ROOT": str(root),
            "DEPLOY_ENV_FILE": str(root / ".env"),
            "DEPLOY_VOLUME_NAME": "test-postgres-data",
            "DEPLOY_LOG_DIR": str(root / "deploy-logs"),
            "FAKE_VOLUME_EXISTS": "true" if volume_exists else "false",
            "FAKE_DOCKER_LOG": str(root / "docker-commands.log"),
            "NO_COLOR": "1",
            "POSTGRES_DB": "terminal_inspection",
            "POSTGRES_USER": "terminal",
            "POSTGRES_PASSWORD": "SafePass_2026",
            "POSTGRES_BIND_ADDRESS": "127.0.0.1",
            "POSTGRES_PORT": "5432",
            "WEB_PORT": "8080",
            "DETECTION_DEVICE": "cpu",
            "CLASSIFICATION_DEVICE": "cpu",
            "DETECTION_IMGSZ": "1280",
            "CLASSIFICATION_IMGSZ": "224",
            "MAX_IMAGES_PER_TASK": "100",
        }
    )
    return env


class DockerDeployScriptTest(unittest.TestCase):
    def test_detects_first_install_configured_install_update_and_orphaned_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            command = 'source "$1"; detect_deployment_mode "$2" "$3"'

            first = run_bash(command, SCRIPT, env_file, "false")
            env_file.write_text("POSTGRES_DB=terminal_inspection\n", encoding="utf-8")
            configured = run_bash(command, SCRIPT, env_file, "false")
            update = run_bash(command, SCRIPT, env_file, "true")
            env_file.unlink()
            orphaned = run_bash(command, SCRIPT, env_file, "true")

        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stdout.strip(), "first_install")
        self.assertEqual(configured.stdout.strip(), "configured_install")
        self.assertEqual(update.stdout.strip(), "update")
        self.assertEqual(orphaned.stdout.strip(), "orphaned_volume")

    def test_database_password_requires_twelve_url_safe_characters(self):
        command = 'source "$1"; validate_db_password "$2"'

        valid = run_bash(command, SCRIPT, "SafePass_2026")
        short = run_bash(command, SCRIPT, "Short_1")
        reserved = run_bash(command, SCRIPT, "Unsafe@Pass_2026")
        placeholder = run_bash(command, SCRIPT, "replace_with_a_strong_password")

        self.assertEqual(valid.returncode, 0, valid.stderr)
        self.assertNotEqual(short.returncode, 0)
        self.assertNotEqual(reserved.returncode, 0)
        self.assertNotEqual(placeholder.returncode, 0)

    def test_configured_first_install_reconfirms_existing_password(self):
        command = """
            source "$1"
            ASSUME_YES=false
            DEPLOYMENT_MODE=configured_install
            POSTGRES_PASSWORD=replace_with_a_strong_password
            prompt_new_password
            printf 'PASSWORD=%s' "$POSTGRES_PASSWORD"
        """

        result = subprocess.run(
            ["bash", "-c", command, "test", str(SCRIPT)],
            input="FreshPass_2026\nFreshPass_2026\n",
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASSWORD=FreshPass_2026", result.stdout)

    def test_update_allows_legacy_short_password_but_first_install_rejects_it(self):
        command = """
            source "$1"
            DEPLOYMENT_MODE="$2"
            POSTGRES_DB=terminal_inspection
            POSTGRES_USER=terminal
            POSTGRES_PASSWORD=terminal
            POSTGRES_BIND_ADDRESS=127.0.0.1
            POSTGRES_PORT=5432
            WEB_PORT=8080
            DETECTION_DEVICE=cpu
            CLASSIFICATION_DEVICE=cpu
            DETECTION_IMGSZ=1280
            CLASSIFICATION_IMGSZ=224
            MAX_IMAGES_PER_TASK=100
            validate_configuration
        """

        update = run_bash(command, SCRIPT, "update")
        first_install = run_bash(command, SCRIPT, "first_install")

        self.assertEqual(update.returncode, 0, update.stderr)
        self.assertIn("旧数据库密码少于 12 位", update.stdout)
        self.assertNotEqual(first_install.returncode, 0)

    def test_writes_private_explicit_docker_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            command = """
                source "$1"
                POSTGRES_DB=terminal_inspection
                POSTGRES_USER=terminal_operator
                POSTGRES_PASSWORD=SafePass_2026
                POSTGRES_BIND_ADDRESS=127.0.0.1
                POSTGRES_PORT=5432
                WEB_PORT=8080
                DETECTION_DEVICE=0
                CLASSIFICATION_DEVICE=0
                DETECTION_IMGSZ=1280
                CLASSIFICATION_IMGSZ=224
                MAX_IMAGES_PER_TASK=100
                write_env_file "$2"
            """

            result = run_bash(command, SCRIPT, env_file)
            contents = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
            mode = stat.S_IMODE(env_file.stat().st_mode) if env_file.exists() else 0

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(mode, 0o600)
        self.assertIn("POSTGRES_USER=terminal_operator", contents)
        self.assertIn("POSTGRES_PASSWORD=SafePass_2026", contents)
        self.assertIn("DETECTION_DEVICE=0", contents)
        self.assertNotIn("DATABASE_URL=", contents)

    def test_environment_update_keeps_native_database_url_in_sync_and_preserves_custom_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "\n".join(
                    (
                        "# Native startup",
                        "DATABASE_URL=postgresql+psycopg://native:pass@localhost/native",
                        "INSPECTION_STORAGE_ROOT=./custom-storage",
                        "POSTGRES_USER=old_user",
                        "CUSTOM_SETTING=keep-me",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            command = """
                source "$1"
                POSTGRES_DB=terminal_inspection
                POSTGRES_USER=new_user
                POSTGRES_PASSWORD=SafePass_2026
                POSTGRES_BIND_ADDRESS=127.0.0.1
                POSTGRES_PORT=5432
                WEB_PORT=8080
                DETECTION_DEVICE=0
                CLASSIFICATION_DEVICE=0
                DETECTION_IMGSZ=1280
                CLASSIFICATION_IMGSZ=224
                MAX_IMAGES_PER_TASK=100
                write_env_file "$2"
            """

            result = run_bash(command, SCRIPT, env_file)
            contents = env_file.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "DATABASE_URL=postgresql+psycopg://new_user:SafePass_2026@127.0.0.1:5432/terminal_inspection",
            contents,
        )
        self.assertNotIn("DATABASE_URL=postgresql+psycopg://native:pass@localhost/native", contents)
        self.assertIn("INSPECTION_STORAGE_ROOT=./custom-storage", contents)
        self.assertIn("CUSTOM_SETTING=keep-me", contents)
        self.assertIn("POSTGRES_USER=new_user", contents)
        self.assertNotIn("POSTGRES_USER=old_user", contents)

    def test_migrates_legacy_database_url_into_explicit_postgres_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "DATABASE_URL=postgresql+psycopg://legacy_user:LegacyPass_2026@127.0.0.1:5544/legacy_db\n",
                encoding="utf-8",
            )
            command = """
                source "$1"
                unset POSTGRES_DB POSTGRES_USER POSTGRES_PASSWORD POSTGRES_PORT
                load_env_file "$2"
                migrate_legacy_database_url
                printf '%s|%s|%s|%s' "$POSTGRES_DB" "$POSTGRES_USER" "$POSTGRES_PASSWORD" "$POSTGRES_PORT"
            """

            result = run_bash(command, SCRIPT, env_file)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            "legacy_db|legacy_user|LegacyPass_2026|5544",
        )

    def test_cpu_compose_is_default_and_gpu_reservation_is_an_overlay(self):
        base = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
        overlay = yaml.safe_load(
            (ROOT / "compose.gpu.yaml").read_text(encoding="utf-8")
        )

        worker = base["services"]["worker"]
        postgres = base["services"]["postgres"]
        gpu_worker = overlay["services"]["worker"]

        self.assertNotIn("deploy", worker)
        self.assertEqual(worker["depends_on"]["api"]["condition"], "service_healthy")
        self.assertEqual(
            postgres["ports"],
            ["${POSTGRES_BIND_ADDRESS:-127.0.0.1}:${POSTGRES_PORT:-5432}:5432"],
        )
        self.assertEqual(
            gpu_worker["deploy"]["resources"]["reservations"]["devices"][0][
                "driver"
            ],
            "nvidia",
        )
        self.assertEqual(
            gpu_worker["deploy"]["resources"]["reservations"]["devices"][0][
                "count"
            ],
            "all",
            "Worker 必须看到全部 GPU，DETECTION_DEVICE=6 等宿主机编号才不会失效",
        )

    def test_worker_image_copies_the_yolo_pipeline_package(self):
        dockerfile = (ROOT / "deploy" / "api.Dockerfile").read_text(encoding="utf-8")

        self.assertIn("COPY yolo11_obb ./yolo11_obb", dockerfile)

    def test_docker_context_excludes_runtime_data_and_local_environments(self):
        dockerignore = ROOT / ".dockerignore"
        self.assertTrue(
            dockerignore.exists(),
            ".dockerignore 必须存在，避免发送超大构建上下文",
        )

        patterns = {
            line.strip().rstrip("/")
            for line in dockerignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        for expected in {
            ".git",
            ".venv",
            "datasets",
            "weights",
            "runs",
            "outputs",
            "var",
            "backups",
            "web_frontend/node_modules",
        }:
            self.assertIn(expected, patterns)

    def test_first_install_dry_run_shows_plan_without_writing_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)

            result = subprocess.run(
                [str(SCRIPT), "--dry-run", "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            env_exists = (project / ".env").exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("首次部署", result.stdout)
        self.assertIn("演练模式", result.stdout)
        self.assertIn("不会写入 .env", result.stdout)
        self.assertNotIn("SafePass_2026", result.stdout + result.stderr)
        self.assertFalse(env_exists)

    def test_update_dry_run_reuses_existing_database_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=True)
            (project / ".env").write_text(
                "\n".join(
                    (
                        "POSTGRES_DB=terminal_inspection",
                        "POSTGRES_USER=terminal",
                        "POSTGRES_PASSWORD=ExistingPass_2026",
                        "POSTGRES_BIND_ADDRESS=127.0.0.1",
                        "POSTGRES_PORT=5432",
                        "WEB_PORT=8080",
                        "DETECTION_DEVICE=cpu",
                        "CLASSIFICATION_DEVICE=cpu",
                        "DETECTION_IMGSZ=1280",
                        "CLASSIFICATION_IMGSZ=224",
                        "MAX_IMAGES_PER_TASK=100",
                        "",
                    )
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [str(SCRIPT), "--dry-run", "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("更新部署", result.stdout)
        self.assertIn("复用现有数据库凭据", result.stdout)
        self.assertNotIn("ExistingPass_2026", result.stdout + result.stderr)

    def test_existing_database_volume_without_env_stops_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=True)

            result = subprocess.run(
                [str(SCRIPT), "--dry-run", "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("数据卷已经存在，但 .env 不存在", result.stdout + result.stderr)

    def test_volume_inspection_error_is_not_treated_as_first_install(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_VOLUME_INSPECT_ERROR"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--dry-run", "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("无法检查 PostgreSQL 数据卷", result.stdout + result.stderr)

    def test_gpu_check_rejects_a_device_index_missing_from_nvidia_smi(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            fake_bin = Path(env["PATH"].split(os.pathsep)[0])
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text(
                """#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu=index"* ]]; then
    printf '0\\n1\\n'
fi
exit 0
""",
                encoding="utf-8",
            )
            nvidia_smi.chmod(0o755)
            command = """
                source "$1"
                DETECTION_DEVICE=6
                CLASSIFICATION_DEVICE=6
                check_gpu
            """

            result = subprocess.run(
                ["bash", "-c", command, "test", str(SCRIPT)],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("GPU 设备 6 不存在", result.stdout + result.stderr)

    def test_gpu_deployment_preflights_cuda_inside_the_worker_container(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["DETECTION_DEVICE"] = "0"
            env["CLASSIFICATION_DEVICE"] = "0"
            fake_bin = Path(env["PATH"].split(os.pathsep)[0])
            nvidia_smi = fake_bin / "nvidia-smi"
            nvidia_smi.write_text(
                """#!/usr/bin/env bash
if [[ "$*" == *"--query-gpu=index"* ]]; then
    printf '0\\n'
fi
exit 0
""",
                encoding="utf-8",
            )
            nvidia_smi.chmod(0o755)

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compose.gpu.yaml", commands)
        self.assertIn("torch.cuda.is_available", commands)
        self.assertLess(
            commands.index("torch.cuda.is_available"),
            commands.index("run --rm api alembic upgrade head"),
        )

    def test_pull_restarts_with_the_newly_pulled_deploy_script(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=True)
            (project / ".env").write_text(
                "\n".join(
                    (
                        "POSTGRES_DB=terminal_inspection",
                        "POSTGRES_USER=terminal",
                        "POSTGRES_PASSWORD=ExistingPass_2026",
                        "POSTGRES_BIND_ADDRESS=127.0.0.1",
                        "POSTGRES_PORT=5432",
                        "WEB_PORT=8080",
                        "DETECTION_DEVICE=cpu",
                        "CLASSIFICATION_DEVICE=cpu",
                        "DETECTION_IMGSZ=1280",
                        "CLASSIFICATION_IMGSZ=224",
                        "MAX_IMAGES_PER_TASK=100",
                        "",
                    )
                ),
                encoding="utf-8",
            )
            fake_bin = Path(env["PATH"].split(os.pathsep)[0])
            fake_git = fake_bin / "git"
            fake_git.write_text(
                """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${FAKE_GIT_LOG}"
if [[ "$*" == *"status --porcelain"* ]]; then
    exit 0
fi
if [[ "$*" == *"rev-parse --is-inside-work-tree"* ]]; then
    echo true
fi
if [[ "$*" == *"rev-parse --abbrev-ref"* ]]; then
    echo origin/main
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_git.chmod(0o755)
            env["FAKE_GIT_LOG"] = str(project / "git-commands.log")

            result = subprocess.run(
                [str(project / "deploy.sh"), "--yes", "--pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            git_commands = (project / "git-commands.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pull --ff-only", git_commands)
        self.assertIn("使用更新后的部署脚本重新检查", result.stdout)

    def test_first_install_runs_build_migration_and_health_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            commands = (project / "docker-commands.log").read_text(encoding="utf-8")
            env_mode = stat.S_IMODE((project / ".env").stat().st_mode)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("部署完成", result.stdout)
        self.assertIn("正在执行，详细输出写入", result.stdout)
        self.assertIn("info\n", commands)
        self.assertIn("config --quiet", commands)
        self.assertIn("build --pull api worker frontend", commands)
        self.assertIn("up -d postgres", commands)
        self.assertIn("run --rm api alembic upgrade head", commands)
        self.assertIn("up -d --remove-orphans api worker frontend", commands)
        self.assertIn("exec -T api python -c", commands)
        self.assertIn("exec -T frontend wget", commands)
        self.assertEqual(env_mode, 0o600)

    def test_update_backs_up_database_before_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=True)
            (project / ".env").write_text(
                "\n".join(
                    (
                        "POSTGRES_DB=terminal_inspection",
                        "POSTGRES_USER=terminal",
                        "POSTGRES_PASSWORD=ExistingPass_2026",
                        "POSTGRES_BIND_ADDRESS=127.0.0.1",
                        "POSTGRES_PORT=5432",
                        "WEB_PORT=8080",
                        "DETECTION_DEVICE=cpu",
                        "CLASSIFICATION_DEVICE=cpu",
                        "DETECTION_IMGSZ=1280",
                        "CLASSIFICATION_IMGSZ=224",
                        "MAX_IMAGES_PER_TASK=100",
                        "",
                    )
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            commands = (project / "docker-commands.log").read_text(encoding="utf-8")
            backups = list((project / "backups").glob("*.dump"))
            env_mode = stat.S_IMODE((project / ".env").stat().st_mode)
            backup_content = (
                backups[0].read_text(encoding="utf-8").strip() if backups else ""
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(backups), 1)
        self.assertEqual(backup_content, "fake-database-dump")
        self.assertRegex(
            backups[0].name,
            r"^terminal_inspection_\d{8}-\d{6}\.[A-Za-z0-9]{6}\.dump$",
        )
        self.assertEqual(env_mode, 0o600)
        self.assertLess(
            commands.index("exec -T postgres pg_dump"),
            commands.index("run --rm api alembic upgrade head"),
        )
        self.assertLess(
            commands.index("stop worker api"),
            commands.index("run --rm api alembic upgrade head"),
        )
        self.assertIn("worker-readiness.json", commands)
        self.assertLess(
            commands.index("worker-readiness.json"),
            commands.index("up -d --remove-orphans api worker frontend"),
        )


if __name__ == "__main__":
    unittest.main()
