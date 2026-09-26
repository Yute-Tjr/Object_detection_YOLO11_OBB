import os
import shutil
import stat
import subprocess
import tempfile
import unittest
import unicodedata
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy.sh"
NGINX_CONFIG = ROOT / "deploy" / "nginx.conf"
API_DOCKERFILE = ROOT / "deploy" / "api.Dockerfile"
WORKER_DOCKERFILE = ROOT / "deploy" / "worker.Dockerfile"


def terminal_display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in text
    )


def run_bash(source: str, *args: Path | str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", source, "test", *(str(arg) for arg in args)],
        cwd=ROOT,
        text=True,
        errors="replace",
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
printf '%s|%s|%s|%s\\n' \
    "${PYTHON_BASE_IMAGE:-}" \
    "${NODE_BASE_IMAGE:-}" \
    "${NGINX_BASE_IMAGE:-}" \
    "${POSTGRES_BASE_IMAGE:-}" >> "${FAKE_DOCKER_IMAGE_LOG}"
printf '%s|%s|%s\\n' \
    "${PYTHON_BASE_IMAGE:-python:3.11-slim}" \
    "${POSTGRES_BASE_IMAGE:-postgres:16-alpine}" \
    "$*" >> "${FAKE_DOCKER_ATTEMPT_LOG}"
if [[ " $* " == *" build --pull api "* && "${FAKE_BUILD_TRANSIENT:-false}" == "true" ]]; then
    if [[ ! -e "${FAKE_BUILD_MARKER}" ]]; then
        : > "${FAKE_BUILD_MARKER}"
        echo 'failed to dial gRPC: header key "x-docker-expose-session-sharedkey" contains value with non-printable ASCII characters' >&2
        exit 1
    fi
fi
if [[ " $* " == *" build --pull api "* \
    && "${FAKE_REGISTRY_TIMEOUT:-false}" == "true" \
    && "${PYTHON_BASE_IMAGE:-}" != *"m.daocloud.io/"* ]]; then
    echo 'failed to authorize: DeadlineExceeded: failed to fetch anonymous token: Get "https://auth.docker.io/token": dial tcp 203.0.113.10:443: i/o timeout' >&2
    exit 1
fi
if [[ " $* " == *" build --pull api "* \
    && "${FAKE_REGISTRY_RESET:-false}" == "true" \
    && "${PYTHON_BASE_IMAGE:-}" != *"m.daocloud.io/"* ]]; then
    echo 'failed to authorize: failed to fetch anonymous token: Get "https://auth.docker.io/token": read tcp 192.168.3.29:57585->172.64.144.78:443: read: connection reset by peer' >&2
    exit 1
fi
if [[ " $* " == *" build --pull api "* \
    && "${FAKE_REGISTRY_REFUSED:-false}" == "true" \
    && "${PYTHON_BASE_IMAGE:-}" != *"m.daocloud.io/"* ]]; then
    echo 'failed to authorize: failed to fetch anonymous token: Get "https://auth.docker.io/token": dial tcp 157.240.21.9:443: connect: connection refused' >&2
    exit 1
fi
if [[ "${FAKE_HUB_AND_MIRROR_UNAVAILABLE:-false}" == "true" \
    && " $* " == *" build --pull api "* ]]; then
    if [[ "${PYTHON_BASE_IMAGE:-}" == *"m.daocloud.io/"* ]]; then
        echo 'failed to resolve source metadata for m.daocloud.io/docker.io/library/python:3.11-slim: i/o timeout' >&2
    else
        echo 'failed to fetch anonymous token from auth.docker.io: connection reset by peer' >&2
    fi
    exit 1
fi
if [[ "${FAKE_POSTGRES_REGISTRY_UNAVAILABLE:-false}" == "true" \
    && " $* " == *" up -d postgres "* \
    && "${POSTGRES_BASE_IMAGE:-}" != *"m.daocloud.io/docker.io/library/postgres:16-alpine" ]]; then
    echo 'failed to fetch anonymous token from auth.docker.io: connection reset by peer' >&2
    exit 1
fi
if [[ "${FAKE_POSTGRES_HUB_AND_MIRROR_UNAVAILABLE:-false}" == "true" \
    && ( " $* " == *" up -d postgres "* || " $* " == *" up -d --pull never postgres "* ) ]]; then
    if [[ " $* " == *" --pull never "* ]]; then
        :
    elif [[ "${POSTGRES_BASE_IMAGE:-}" == *"m.daocloud.io/"* ]]; then
        echo 'failed to resolve source metadata for m.daocloud.io/docker.io/library/postgres:16-alpine: i/o timeout' >&2
        exit 1
    else
        echo 'failed to fetch anonymous token from auth.docker.io: connection reset by peer' >&2
        exit 1
    fi
fi
if [[ " $* " == *" build --pull api "* && "${FAKE_BUILD_ERROR:-false}" == "true" ]]; then
    echo 'Dockerfile parse error: unknown instruction' >&2
    exit 1
fi
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
if [[ " $* " == *" exec -T api python scripts/manage_users.py add "* ]]; then
    cat > "${FAKE_APP_USER_STDIN_LOG}"
    if [[ "${FAKE_APP_USER_FAILURE:-false}" == "true" ]]; then
        echo "错误：用户已存在" >&2
        exit 2
    fi
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
            "FAKE_DOCKER_IMAGE_LOG": str(root / "docker-images.log"),
            "FAKE_DOCKER_ATTEMPT_LOG": str(root / "docker-attempts.log"),
            "FAKE_APP_USER_STDIN_LOG": str(root / "app-user-stdin.log"),
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
    if not volume_exists:
        env.update(
            {
                "INITIAL_APP_USERNAME": "InitialAdmin",
                "INITIAL_APP_PASSWORD": "Secret6",
            }
        )
    return env


class DockerDeployScriptTest(unittest.TestCase):
    def test_nginx_preserves_public_host_port_for_same_origin_checks(self):
        config = NGINX_CONFIG.read_text(encoding="utf-8")

        self.assertIn("proxy_set_header Host $http_host;", config)
        self.assertNotIn("proxy_set_header Host $host;", config)

    def test_banner_lines_have_the_same_terminal_display_width(self):
        for title in (
            "首次部署 · Docker Compose",
            "首次部署 · 复用已有配置",
            "更新部署 · Docker Compose",
            "安全检查未通过",
        ):
            with self.subTest(title=title):
                result = run_bash('source "$1"; banner "$2"', SCRIPT, title)
                banner_lines = [line for line in result.stdout.splitlines() if line]

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(banner_lines), 4)
                self.assertEqual(
                    [terminal_display_width(line) for line in banner_lines],
                    [46, 46, 46, 46],
                )

    def test_failed_logged_step_prints_numeric_exit_code_without_unbound_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            log_file = Path(directory) / "deploy.log"
            command = """
                source "$1"
                LOG_FILE="$2"
                : > "$LOG_FILE"
                run_logged_step "07/10" "测试失败步骤" bash -c 'exit 7'
            """

            result = run_bash(command, SCRIPT, log_file)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("退出码 7", result.stdout + result.stderr)
        self.assertNotIn("unbound variable", result.stdout + result.stderr)

    def test_progress_frame_contains_an_indeterminate_bar_and_elapsed_time(self):
        result = run_bash('source "$1"; progress_frame 7', SCRIPT)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[", result.stdout)
        self.assertIn("]", result.stdout)
        self.assertIn("已用时 7 秒", result.stdout)

    def test_transient_buildkit_shared_key_error_is_retried_once(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_BUILD_TRANSIENT"] = "true"
            env["FAKE_BUILD_MARKER"] = str(project / "build-failed-once")

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(commands.count(" build --pull api\n"), 2)
        self.assertEqual(commands.count(" build --pull worker\n"), 1)
        self.assertEqual(commands.count(" build --pull frontend\n"), 1)
        self.assertIn("BuildKit 会话异常，自动重试一次", result.stdout)

    def test_registry_timeout_falls_back_to_domestic_images_before_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_REGISTRY_TIMEOUT"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")
            deploy_log = next((project / "deploy-logs").glob("deploy-*.log")).read_text(
                encoding="utf-8"
            )
            attempts = (project / "docker-attempts.log").read_text(encoding="utf-8")

        self.assertEqual(
            result.returncode,
            0,
            result.stdout + result.stderr + "\nCOMMANDS:\n" + commands + "\nLOG:\n" + deploy_log,
        )
        self.assertEqual(commands.count(" build --pull api\n"), 2)
        self.assertNotIn(" build api\n", commands)
        self.assertIn(" build --pull worker\n", commands)
        build_api_attempts = [line for line in attempts.splitlines() if " build --pull api" in line]
        self.assertEqual(build_api_attempts[0].split("|", 1)[0], "python:3.11-slim")
        self.assertTrue(build_api_attempts[1].startswith("m.daocloud.io/"))
        self.assertIn("Docker Hub 网络异常，切换到国内基础镜像源", result.stdout)

    def test_non_network_build_error_does_not_fall_back_to_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_BUILD_ERROR"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(commands.count(" build --pull api\n"), 1)
        self.assertNotIn(" build api\n", commands)
        self.assertNotIn("改用本地基础镜像缓存", result.stdout + result.stderr)

    def test_registry_connection_reset_falls_back_to_domestic_images(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_REGISTRY_RESET"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands.count(" build --pull api\n"), 2)
        self.assertNotIn(" build api\n", commands)
        self.assertIn("切换到国内基础镜像源", result.stdout)

    def test_registry_connection_refused_falls_back_to_domestic_images(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_REGISTRY_REFUSED"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands.count(" build --pull api\n"), 2)
        self.assertNotIn(" build api\n", commands)
        self.assertIn("切换到国内基础镜像源", result.stdout)

    def test_hub_and_domestic_failure_falls_back_to_default_local_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_HUB_AND_MIRROR_UNAVAILABLE"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")
            image_environments = (project / "docker-images.log").read_text(
                encoding="utf-8"
            )
            attempts = (project / "docker-attempts.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands.count(" build --pull api\n"), 2)
        self.assertEqual(commands.count(" build api\n"), 1)
        self.assertIn("build worker", commands)
        self.assertIn("build frontend", commands)
        self.assertIn("已切换到国内基础镜像源", result.stdout)
        self.assertIn("国内镜像网络异常，改用本地基础镜像缓存", result.stdout)
        build_api_attempts = [line for line in attempts.splitlines() if " build" in line and " api" in line]
        self.assertEqual(
            [line.split("|", 1)[0] for line in build_api_attempts],
            [
                "python:3.11-slim",
                "m.daocloud.io/docker.io/library/python:3.11-slim",
                "python:3.11-slim",
            ],
        )
        self.assertIn(
            "m.daocloud.io/docker.io/library/python:3.11-slim"
            "|m.daocloud.io/docker.io/library/node:22-alpine"
            "|m.daocloud.io/docker.io/library/nginx:1.27-alpine"
            "|m.daocloud.io/docker.io/library/postgres:16-alpine",
            image_environments,
        )

    def test_postgres_pull_failure_falls_back_to_domestic_image(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_POSTGRES_REGISTRY_UNAVAILABLE"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            commands = (project / "docker-commands.log").read_text(encoding="utf-8")
            image_environments = (project / "docker-images.log").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(commands.count(" up -d postgres\n"), 2)
        self.assertIn("已切换到国内基础镜像源", result.stdout)
        self.assertIn(
            "m.daocloud.io/docker.io/library/postgres:16-alpine",
            image_environments,
        )

    def test_postgres_hub_and_domestic_failure_falls_back_to_local_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env["FAKE_POSTGRES_HUB_AND_MIRROR_UNAVAILABLE"] = "true"

            result = subprocess.run(
                [str(SCRIPT), "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                errors="replace",
                capture_output=True,
                check=False,
            )
            attempts = (project / "docker-attempts.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        postgres_attempts = [
            line for line in attempts.splitlines()
            if " up -d" in line.split("|", 2)[2]
            and line.split("|", 2)[2].endswith("postgres")
        ]
        self.assertEqual(len(postgres_attempts), 3)
        self.assertEqual(postgres_attempts[0].split("|")[1], "postgres:16-alpine")
        self.assertIn("m.daocloud.io/", postgres_attempts[1].split("|")[1])
        self.assertEqual(postgres_attempts[2].split("|")[1], "postgres:16-alpine")
        self.assertIn("--pull never postgres", postgres_attempts[2])

    def test_domestic_mirror_prefix_can_be_overridden(self):
        command = """
            source "$1"
            ASSUME_YES=true
            DOCKER_MIRROR_PREFIX=mirror.example.com/docker.io
            enable_domestic_image_source
            printf '%s|%s|%s|%s' \
                "$PYTHON_BASE_IMAGE" "$NODE_BASE_IMAGE" \
                "$NGINX_BASE_IMAGE" "$POSTGRES_BASE_IMAGE"
        """

        result = run_bash(command, SCRIPT)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "mirror.example.com/docker.io/library/python:3.11-slim"
            "|mirror.example.com/docker.io/library/node:22-alpine"
            "|mirror.example.com/docker.io/library/nginx:1.27-alpine"
            "|mirror.example.com/docker.io/library/postgres:16-alpine",
            result.stdout,
        )

    def test_domestic_fallback_also_switches_debian_package_sources(self):
        command = """
            source "$1"
            ASSUME_YES=true
            enable_domestic_image_source
            printf '%s|%s' "$DEBIAN_MIRROR_URL" "$DEBIAN_SECURITY_MIRROR_URL"
        """

        result = run_bash(command, SCRIPT)
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "https://mirrors.aliyun.com/debian"
            "|https://mirrors.aliyun.com/debian-security",
            result.stdout,
        )
        for service in ("api", "worker"):
            with self.subTest(service=service):
                build_args = compose["services"][service]["build"]["args"]
                self.assertEqual(
                    build_args["DEBIAN_MIRROR_URL"],
                    "${DEBIAN_MIRROR_URL:-}",
                )
                self.assertEqual(
                    build_args["DEBIAN_SECURITY_MIRROR_URL"],
                    "${DEBIAN_SECURITY_MIRROR_URL:-}",
                )
        for dockerfile in (API_DOCKERFILE, WORKER_DOCKERFILE):
            with self.subTest(dockerfile=dockerfile.name):
                contents = dockerfile.read_text(encoding="utf-8")
                self.assertIn("ARG DEBIAN_MIRROR_URL=", contents)
                self.assertIn("ARG DEBIAN_SECURITY_MIRROR_URL=", contents)
                self.assertIn("/etc/apt/sources.list.d/debian.sources", contents)

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

    def test_database_password_requires_six_url_safe_characters(self):
        command = 'source "$1"; validate_db_password "$2"'

        valid = run_bash(command, SCRIPT, "Db_123")
        short = run_bash(command, SCRIPT, "Db_12")
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

    def test_six_character_password_is_valid_for_install_and_update(self):
        command = """
            source "$1"
            DEPLOYMENT_MODE="$2"
            POSTGRES_DB=terminal_inspection
            POSTGRES_USER=terminal
            POSTGRES_PASSWORD=Db_123
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
        self.assertNotIn("旧数据库密码少于", update.stdout)
        self.assertEqual(first_install.returncode, 0, first_install.stderr)

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
        self.assertNotIn("DETECTION_IMGSZ=", contents)
        self.assertNotIn("CLASSIFICATION_IMGSZ=", contents)
        self.assertNotIn("DATABASE_URL=", contents)

    def test_environment_backup_is_stored_outside_the_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            env_file = project / ".env"
            env_file.write_text("ORIGINAL=value\n", encoding="utf-8")
            command = """
                source "$1"
                PROJECT_ROOT="$2"
                ENV_FILE="$3"
                backup_environment_file
            """

            result = run_bash(command, SCRIPT, project, env_file)
            backups = list((project / "backups" / "env").glob(".env.*.backup"))
            backup_content = backups[0].read_text(encoding="utf-8") if backups else ""
            root_backup_exists = (project / ".env.backup").exists()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(backups), 1)
        self.assertEqual(backup_content, "ORIGINAL=value\n")
        self.assertFalse(root_backup_exists)

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
        api = base["services"]["api"]
        postgres = base["services"]["postgres"]
        api_environment = base["services"]["api"]["environment"]
        gpu_worker = overlay["services"]["worker"]

        self.assertEqual(api["build"]["dockerfile"], "deploy/api.Dockerfile")
        self.assertEqual(worker["build"]["dockerfile"], "deploy/worker.Dockerfile")
        self.assertEqual(
            worker["build"]["args"]["PYTORCH_INDEX_URL"],
            "${PYTORCH_INDEX_URL:-}",
        )
        self.assertNotIn("deploy", worker)
        self.assertEqual(worker["depends_on"]["api"]["condition"], "service_healthy")
        self.assertEqual(
            postgres["ports"],
            ["${POSTGRES_BIND_ADDRESS:-127.0.0.1}:${POSTGRES_PORT:-5432}:5432"],
        )
        self.assertEqual(postgres["command"], ["postgres", "-c", "timezone=Asia/Shanghai"])
        self.assertEqual(postgres["environment"]["TZ"], "Asia/Shanghai")
        self.assertEqual(api_environment["TZ"], "Asia/Shanghai")
        self.assertEqual(api_environment["DETECTION_IMGSZ"], "1280")
        self.assertEqual(api_environment["CLASSIFICATION_IMGSZ"], "224")
        self.assertEqual(
            api_environment["SESSION_TTL_HOURS"],
            "${SESSION_TTL_HOURS:-12}",
        )
        self.assertEqual(
            api_environment["SESSION_COOKIE_SECURE"],
            "${SESSION_COOKIE_SECURE:-false}",
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

    def test_api_and_worker_images_install_different_runtime_profiles(self):
        api_dockerfile = (ROOT / "deploy" / "api.Dockerfile").read_text(
            encoding="utf-8"
        )
        worker_dockerfile = (ROOT / "deploy" / "worker.Dockerfile").read_text(
            encoding="utf-8"
        )

        self.assertIn("/tmp/api-constraints.txt", api_dockerfile)
        self.assertIn("--constraint /tmp/api-constraints.txt", api_dockerfile)
        self.assertIn("sed -E", api_dockerfile)
        self.assertNotIn("pip install -r requirements.txt", api_dockerfile)
        self.assertNotIn("COPY yolo11_obb ./yolo11_obb", api_dockerfile)
        self.assertNotIn("COPY obb_detection ./obb_detection", api_dockerfile)
        self.assertIn("pip install -r requirements.txt", worker_dockerfile)
        self.assertIn("ARG PYTORCH_INDEX_URL", worker_dockerfile)
        self.assertIn("COPY yolo11_obb ./yolo11_obb", worker_dockerfile)
        self.assertIn("COPY obb_detection ./obb_detection", worker_dockerfile)

    def test_cpu_uses_cpu_only_pytorch_and_gpu_uses_cuda_distribution(self):
        command = """
            source "$1"
            DETECTION_DEVICE="$2"
            CLASSIFICATION_DEVICE="$3"
            PYTORCH_CPU_INDEX_URL="${4:-}"
            configure_pytorch_distribution
            printf '%s' "$PYTORCH_INDEX_URL"
        """

        cpu = run_bash(
            command,
            SCRIPT,
            "cpu",
            "cpu",
            "",
        )
        gpu = run_bash(
            command,
            SCRIPT,
            "0",
            "0",
            "",
        )
        mixed = run_bash(
            command,
            SCRIPT,
            "cpu",
            "1",
            "",
        )
        custom_cpu = run_bash(
            command,
            SCRIPT,
            "cpu",
            "cpu",
            "https://mirror.example.com/pytorch/cpu",
        )

        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        self.assertEqual(cpu.stdout, "https://download.pytorch.org/whl/cpu")
        self.assertEqual(gpu.returncode, 0, gpu.stderr)
        self.assertEqual(gpu.stdout, "")
        self.assertEqual(mixed.returncode, 0, mixed.stderr)
        self.assertEqual(mixed.stdout, "")
        self.assertEqual(custom_cpu.returncode, 0, custom_cpu.stderr)
        self.assertEqual(
            custom_cpu.stdout,
            "https://mirror.example.com/pytorch/cpu",
        )

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
        self.assertNotIn("Secret6", result.stdout + result.stderr)
        self.assertFalse(env_exists)

    def test_noninteractive_first_install_requires_app_user_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=False)
            env.pop("INITIAL_APP_USERNAME")
            env.pop("INITIAL_APP_PASSWORD")

            result = subprocess.run(
                [str(SCRIPT), "--dry-run", "--yes", "--no-pull"],
                cwd=project,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("INITIAL_APP_USERNAME", result.stdout + result.stderr)
        self.assertIn("INITIAL_APP_PASSWORD", result.stdout + result.stderr)

    def test_first_install_creates_app_user_via_stdin_without_leaking_password(self):
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
            deploy_log = next((project / "deploy-logs").glob("deploy-*.log")).read_text(encoding="utf-8")
            written_env = (project / ".env").read_text(encoding="utf-8")
            password_stdin = (project / "app-user-stdin.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("manage_users.py add InitialAdmin", commands)
        self.assertEqual(password_stdin, "Secret6\nSecret6\n")
        for visible in (result.stdout, result.stderr, commands, deploy_log, written_env):
            self.assertNotIn("Secret6", visible)
        self.assertNotIn("INITIAL_APP_USERNAME", written_env)
        self.assertNotIn("INITIAL_APP_PASSWORD", written_env)
        self.assertIn("登录用户已创建：InitialAdmin", result.stdout)

    def test_update_skips_user_creation_without_explicit_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=True)
            (project / ".env").write_text(
                "POSTGRES_DB=terminal_inspection\nPOSTGRES_USER=terminal\n"
                "POSTGRES_PASSWORD=ExistingPass_2026\nPOSTGRES_BIND_ADDRESS=127.0.0.1\n"
                "POSTGRES_PORT=5432\nWEB_PORT=8080\nDETECTION_DEVICE=cpu\n"
                "CLASSIFICATION_DEVICE=cpu\nMAX_IMAGES_PER_TASK=100\n",
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

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("manage_users.py add", commands)

    def test_update_adds_user_when_noninteractive_credentials_are_provided(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            prepare_fake_project(project)
            env = fake_docker_environment(project, volume_exists=True)
            env["INITIAL_APP_USERNAME"] = "NewOperator"
            env["INITIAL_APP_PASSWORD"] = "NewPass6"
            (project / ".env").write_text(
                "POSTGRES_DB=terminal_inspection\nPOSTGRES_USER=terminal\n"
                "POSTGRES_PASSWORD=ExistingPass_2026\nPOSTGRES_BIND_ADDRESS=127.0.0.1\n"
                "POSTGRES_PORT=5432\nWEB_PORT=8080\nDETECTION_DEVICE=cpu\n"
                "CLASSIFICATION_DEVICE=cpu\nMAX_IMAGES_PER_TASK=100\n",
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
            password_stdin = (project / "app-user-stdin.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("manage_users.py add NewOperator", commands)
        self.assertEqual(password_stdin, "NewPass6\nNewPass6\n")
        self.assertNotIn("NewPass6", result.stdout + result.stderr + commands)

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
            env["INITIAL_APP_USERNAME"] = "PulledOperator"
            env["INITIAL_APP_PASSWORD"] = "PulledPass6"
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
            docker_commands = (project / "docker-commands.log").read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("pull --ff-only", git_commands)
        self.assertIn("使用更新后的部署脚本重新检查", result.stdout)
        self.assertIn("manage_users.py add PulledOperator", docker_commands)
        self.assertNotIn("PulledPass6", result.stdout + result.stderr + docker_commands)

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
            written_env = (project / ".env").read_text(encoding="utf-8")
            env_mode = stat.S_IMODE((project / ".env").stat().st_mode)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("部署完成", result.stdout)
        self.assertIn("http://127.0.0.1:8080/tasks", result.stdout)
        self.assertIn("登录用户已创建：InitialAdmin", result.stdout)
        self.assertIn("SESSION_TTL_HOURS=12", written_env)
        self.assertIn("SESSION_COOKIE_SECURE=false", written_env)
        self.assertNotIn("DEFAULT_PASSWORD", written_env)
        self.assertIn("正在执行，详细输出写入", result.stdout)
        self.assertIn("info\n", commands)
        self.assertIn("config --quiet", commands)
        self.assertIn("build --pull api", commands)
        self.assertIn("build --pull worker", commands)
        self.assertIn("build --pull frontend", commands)
        self.assertNotIn("build --pull api worker frontend", commands)
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
