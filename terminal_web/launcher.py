from __future__ import annotations

import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Iterable, Sequence, TypeVar

from sqlalchemy import text

from terminal_web.config import Settings
from terminal_web.database import build_session_factory


T = TypeVar("T")


class LauncherError(RuntimeError):
    """A user-facing startup validation or supervision failure."""


@dataclass(frozen=True)
class CheckResult(Generic[T]):
    value: T
    detail: str


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    prefix: str
    command: tuple[str, ...]
    cwd: Path
    env: dict[str, str] | None = None


def run_check(label: str, checker: Callable[[], T | CheckResult[T]]) -> T:
    try:
        result = checker()
    except LauncherError as exc:
        print(f"[失败] {label}：{exc}", flush=True)
        raise
    except Exception as exc:
        print(f"[失败] {label}：{exc}", flush=True)
        raise LauncherError(str(exc)) from exc

    if isinstance(result, CheckResult):
        print(f"[通过] {label}：{result.detail}", flush=True)
        return result.value
    print(f"[通过] {label}：{result}", flush=True)
    return result


def load_settings(root: Path) -> CheckResult[Settings]:
    env_file = root / ".env"
    if not env_file.is_file():
        raise LauncherError("未找到 .env，请先执行 cp .env.example .env 并完成配置")
    try:
        settings = Settings(_env_file=env_file)
    except Exception as exc:
        raise LauncherError(f".env 配置无效：{exc}") from exc
    return CheckResult(settings, f"已加载 {env_file.name}")


def check_weights(settings: Settings) -> str:
    weights = (
        ("目标检测", settings.detector_weights),
        ("label3 分类", settings.label3_classifier_weights),
        ("label5 分类", settings.label5_classifier_weights),
    )
    invalid = [f"{name}={path}" for name, path in weights if not path.is_file() or path.stat().st_size == 0]
    if invalid:
        raise LauncherError("模型权重不存在或为空：" + "；".join(invalid))
    return "3 个模型权重均存在且非空"


def check_python_environment() -> str:
    required_modules = (
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "alembic",
        "psycopg",
        "torch",
        "torchvision",
        "ultralytics",
    )
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if missing:
        raise LauncherError(
            "缺少 Python 依赖："
            + ", ".join(missing)
            + "；uv 环境请执行 uv sync，Conda 环境请执行 python -m pip install -r requirements.txt"
        )
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"Python {version}，运行解释器 {sys.executable}，核心依赖完整"


def check_frontend_dependencies(
    root: Path,
    npm_path: str | None = None,
    node_version: str | None = None,
) -> str:
    frontend = root / "web_frontend"
    package_json = frontend / "package.json"
    vite = frontend / "node_modules" / ".bin" / "vite"
    resolved_npm = npm_path or shutil.which("npm")
    if resolved_npm is None:
        raise LauncherError("未找到 npm，请先安装 Node.js 与 npm")
    if not package_json.is_file():
        raise LauncherError(f"未找到前端配置：{package_json}")
    if not vite.exists():
        raise LauncherError("前端依赖未安装，请先执行 cd web_frontend && npm ci")
    if node_version is None:
        adjacent_node = Path(resolved_npm).parent / "node"
        node_path = str(adjacent_node) if adjacent_node.is_file() else shutil.which("node")
        if node_path is None:
            raise LauncherError("未找到 Node.js 可执行文件")
        result = subprocess.run(
            [node_path, "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise LauncherError(f"无法读取 Node.js 版本：{result.stderr.strip()}")
        node_version = result.stdout.strip().lstrip("v")
    match = re.match(r"^(\d+)", node_version)
    if match is None or int(match.group(1)) < 18:
        raise LauncherError(f"需要 Node.js 18 或更高版本，当前为 {node_version}")
    return f"Node.js {node_version}，npm={resolved_npm}，package.json 与 Vite 依赖完整"


def _configured_device(
    requested: str | None,
    *,
    cuda_available: bool,
    cuda_device_count: int,
    cuda_device_names: Sequence[str],
    mps_available: bool,
) -> str:
    value = "" if requested is None else str(requested).strip().lower()
    if not value:
        if cuda_available:
            value = "cuda:0"
        elif mps_available:
            value = "mps"
        else:
            value = "cpu"
    if value == "cpu":
        return "CPU"
    if value == "mps":
        if not mps_available:
            raise LauncherError("配置请求 Apple MPS GPU，但当前 PyTorch 不可用 MPS")
        return "GPU(Apple MPS)"
    if value.isdigit():
        index = int(value)
    else:
        match = re.fullmatch(r"cuda(?::(\d+))?", value)
        if match is None:
            raise LauncherError(f"无法识别设备配置：{requested}")
        index = int(match.group(1) or 0)
    if not cuda_available:
        raise LauncherError(f"配置请求 CUDA:{index}，但当前 PyTorch 检测不到 CUDA GPU")
    if index >= cuda_device_count:
        raise LauncherError(f"配置请求 CUDA:{index}，但当前只有 {cuda_device_count} 块 CUDA GPU")
    name = cuda_device_names[index] if index < len(cuda_device_names) else "未知型号"
    return f"GPU(CUDA:{index} {name})"


def describe_devices(
    *,
    detection_device: str | None,
    classification_device: str | None,
    cuda_available: bool,
    cuda_device_count: int,
    cuda_device_names: Sequence[str],
    mps_available: bool,
) -> str:
    detection = _configured_device(
        detection_device,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        mps_available=mps_available,
    )
    classification = _configured_device(
        classification_device,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_device_names,
        mps_available=mps_available,
    )
    return f"检测模型={detection}；分类模型={classification}"


def check_device_environment(settings: Settings) -> str:
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_names = tuple(torch.cuda.get_device_name(index) for index in range(cuda_device_count))
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    configured = describe_devices(
        detection_device=settings.detection_device,
        classification_device=settings.classification_device,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_names=cuda_names,
        mps_available=mps_available,
    )
    available = []
    if cuda_available:
        available.append(f"CUDA {cuda_device_count} 块")
    if mps_available:
        available.append("Apple MPS")
    hardware = "、".join(available) if available else "无可用 GPU 加速器"
    return f"可用加速器={hardware}；当前配置：{configured}"


def check_postgresql(database_url: str) -> str:
    engine, _ = build_session_factory(database_url)
    try:
        with engine.connect() as connection:
            value = connection.execute(text("SELECT 1")).scalar_one()
        if value != 1:
            raise LauncherError("数据库探测查询未返回预期结果")
    except LauncherError:
        raise
    except Exception as exc:
        raise LauncherError(f"连接失败：{exc}") from exc
    finally:
        engine.dispose()
    return "连接正常，SELECT 1 执行成功"


def run_migrations(root: Path, python_executable: str) -> str:
    result = subprocess.run(
        [python_executable, "-m", "alembic", "upgrade", "head"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "未知错误").strip()
        raise LauncherError(f"数据库迁移失败：{message}")
    return "已升级到最新 head"


def build_service_specs(
    root: Path,
    *,
    host: str,
    api_port: int,
    web_port: int,
    python_executable: str,
    npm_path: str,
) -> tuple[ServiceSpec, ...]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    npm_directory = str(Path(npm_path).parent)
    current_path = environment.get("PATH", "")
    environment["PATH"] = npm_directory + (os.pathsep + current_path if current_path else "")
    return (
        ServiceSpec(
            name="API",
            prefix="[API]",
            command=(
                python_executable,
                "scripts/run_terminal_api.py",
                "--host",
                host,
                "--port",
                str(api_port),
            ),
            cwd=root,
            env=environment,
        ),
        ServiceSpec(
            name="WORKER",
            prefix="[WORKER]",
            command=(python_executable, "scripts/run_terminal_worker.py"),
            cwd=root,
            env=environment,
        ),
        ServiceSpec(
            name="WEB",
            prefix="[WEB]",
            command=(
                npm_path,
                "run",
                "dev",
                "--",
                "--host",
                host,
                "--port",
                str(web_port),
            ),
            cwd=root / "web_frontend",
            env=environment,
        ),
    )


def _read_output(process: subprocess.Popen[str], prefix: str, printer: Callable[[str], None]) -> None:
    if process.stdout is None:
        return
    for line in process.stdout:
        printer(f"{prefix} {line.rstrip()}")


def _terminate_process(process: subprocess.Popen[str], timeout: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=timeout)
    except ProcessLookupError:
        return


def stop_processes(processes: Iterable[subprocess.Popen[str]]) -> None:
    for process in processes:
        _terminate_process(process)


def supervise_services(
    services: Sequence[ServiceSpec],
    *,
    printer: Callable[[str], None] = print,
    poll_interval: float = 0.2,
) -> int:
    processes: list[tuple[ServiceSpec, subprocess.Popen[str]]] = []
    readers: list[threading.Thread] = []
    try:
        for service in services:
            process = subprocess.Popen(
                list(service.command),
                cwd=service.cwd,
                env=service.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=os.name == "posix",
            )
            processes.append((service, process))
            reader = threading.Thread(
                target=_read_output,
                args=(process, service.prefix, printer),
                daemon=True,
            )
            reader.start()
            readers.append(reader)
            printer(f"[通过] {service.name} 进程：已启动，PID={process.pid}")

        while True:
            for service, process in processes:
                return_code = process.poll()
                if return_code is None:
                    continue
                if return_code == 0:
                    printer(f"[停止] {service.name} 已退出，正在停止其余服务")
                    return 0
                printer(
                    f"[失败] {service.name} 异常退出（退出码 {return_code}），正在停止其余服务"
                )
                return return_code
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        printer("\n[停止] 收到 Ctrl+C，正在关闭 API、Worker 和前端")
        return 0
    except OSError as exc:
        printer(f"[失败] 服务进程启动失败：{exc}，正在停止已启动的服务")
        return 1
    finally:
        stop_processes(process for _, process in processes)
        for reader in readers:
            reader.join(timeout=1.0)
        for _, process in processes:
            if process.stdout is not None:
                process.stdout.close()
        if processes:
            printer("[完成] 所有服务进程均已停止")
