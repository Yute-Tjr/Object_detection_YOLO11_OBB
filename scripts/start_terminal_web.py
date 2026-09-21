#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_web.launcher import (
    LauncherError,
    build_service_specs,
    check_device_environment,
    check_frontend_dependencies,
    check_postgresql,
    check_python_environment,
    check_weights,
    load_settings,
    run_check,
    run_migrations,
    supervise_services,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键启动端子检测 API、Worker 和网页前端")
    parser.add_argument("--host", default="127.0.0.1", help="API 与前端监听地址")
    parser.add_argument("--api-port", type=int, default=8000, help="API 端口")
    parser.add_argument("--web-port", type=int, default=5173, help="前端端口")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(ROOT)
    print("=== 端子智能检测一键启动 ===", flush=True)
    try:
        settings = run_check("环境配置", lambda: load_settings(ROOT))
        run_check("模型权重", lambda: check_weights(settings))
        run_check("Python 环境", check_python_environment)
        npm_path = shutil.which("npm")
        run_check("前端依赖", lambda: check_frontend_dependencies(ROOT, npm_path=npm_path))
        run_check("运行设备", lambda: check_device_environment(settings))
        run_check("PostgreSQL 连接", lambda: check_postgresql(settings.database_url))
        run_check("数据库迁移", lambda: run_migrations(ROOT, sys.executable))
    except LauncherError:
        print("[终止] 启动前检查未通过，尚未启动任何服务", flush=True)
        return 1

    if npm_path is None:
        print("[失败] 未找到 npm", flush=True)
        return 1
    services = build_service_specs(
        ROOT,
        host=args.host,
        api_port=args.api_port,
        web_port=args.web_port,
        python_executable=sys.executable,
        npm_path=npm_path,
    )
    print(
        f"[启动] 检查全部通过，正在启动服务；网页地址 http://{args.host}:{args.web_port}/tasks",
        flush=True,
    )
    return supervise_services(services)


if __name__ == "__main__":
    raise SystemExit(main())
