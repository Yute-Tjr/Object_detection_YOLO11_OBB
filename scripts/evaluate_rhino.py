#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export RHINO predictions and calculate unified Ultralytics OBB metrics.")
    parser.add_argument("--data", type=Path, default=Path("datasets/obb_thin_thick/data.yaml"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--min-conf", type=float, default=0.001)
    parser.add_argument(
        "--metric-python",
        type=Path,
        default=None,
        help="Python interpreter containing ultralytics; defaults to the current interpreter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    run_dir = run_dir.resolve()
    labels = run_dir / "labels"
    metrics = run_dir / "custom_metrics.csv"
    export_command = [
        sys.executable, str(ROOT / "scripts/export_rhino_predictions.py"),
        "--data", str(args.data), "--predictions", str(args.predictions), "--split", args.split,
        "--output", str(labels), "--min-conf", str(args.min_conf),
    ]
    metric_python = str(args.metric_python.expanduser().resolve()) if args.metric_python else sys.executable
    metric_command = [
        metric_python, str(ROOT / "scripts/evaluate_obb_prediction_labels.py"),
        "--data", str(args.data), "--pred-labels", str(labels), "--split", args.split,
        "--output", str(metrics),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "high_iou_eval_command.txt").write_text(
        " ".join(export_command) + "\n" + " ".join(metric_command) + "\n",
        encoding="utf-8",
    )
    subprocess.run(export_command, cwd=ROOT, check=True)
    subprocess.run(metric_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
