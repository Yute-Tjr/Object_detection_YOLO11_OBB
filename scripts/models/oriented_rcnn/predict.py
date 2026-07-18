#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMRotate Oriented R-CNN inference and export YOLO OBB labels."
    )
    parser.add_argument(
        "--mmrotate-root",
        "--rhino-root",
        dest="mmrotate_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("datasets/obb_thin_thick/data.yaml"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-conf", type=float, default=0.001)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for MMRotate 1.x.")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    mmrotate_root = args.mmrotate_root.expanduser().resolve()
    test_tool = mmrotate_root / "tools" / "test.py"
    if not test_tool.is_file():
        raise FileNotFoundError(f"MMRotate test tool not found: {test_tool}")
    config = _resolve(args.config)
    weights = _resolve(args.weights)
    data = _resolve(args.data)
    for path, label in ((config, "config"), (weights, "weights"), (data, "dataset yaml")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    run_dir = _resolve(args.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"run directory is not empty: {run_dir}; pass --exist-ok to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = run_dir / "predictions.pkl"
    labels = run_dir / "labels"
    mmrotate_work_dir = run_dir / "mmrotate_eval"

    infer_command = [
        args.python,
        str(test_tool),
        str(config),
        str(weights),
        "--out",
        str(predictions),
        "--work-dir",
        str(mmrotate_work_dir),
        "--cfg-options",
        "test_evaluator.format_only=False",
        "test_evaluator.merge_patches=False",
    ]
    export_command = [
        args.python,
        str(ROOT / "scripts" / "common" / "export_mmrotate_predictions.py"),
        "--data",
        str(data),
        "--predictions",
        str(predictions),
        "--split",
        args.split,
        "--output",
        str(labels),
        "--min-conf",
        str(args.min_conf),
    ]
    (run_dir / "predict_command.txt").write_text(
        shlex.join(infer_command) + "\n" + shlex.join(export_command) + "\n",
        encoding="utf-8",
    )
    print("inference:", shlex.join(infer_command))
    print("export:", shlex.join(export_command))
    if not args.dry_run:
        subprocess.run(infer_command, cwd=mmrotate_root, check=True)
        subprocess.run(export_command, cwd=ROOT, check=True)
        print(f"predictions: {predictions}")
        print(f"labels: {labels}")


if __name__ == "__main__":
    main()
