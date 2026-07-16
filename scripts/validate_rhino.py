#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RHINO and save MMRotate prediction samples.")
    parser.add_argument("--rhino-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Prediction pickle written by MMRotate.")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rhino_root = args.rhino_root.expanduser().resolve()
    test_tool = rhino_root / "tools/test.py"
    if not test_tool.exists():
        raise FileNotFoundError(f"RHINO test tool not found: {test_tool}")
    config = args.config.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [args.python, str(test_tool), str(config), str(weights), "--out", str(output), "--work-dir", str(work_dir)]
    (work_dir / "validate_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    subprocess.run(command, cwd=rhino_root, check=True)


if __name__ == "__main__":
    main()
