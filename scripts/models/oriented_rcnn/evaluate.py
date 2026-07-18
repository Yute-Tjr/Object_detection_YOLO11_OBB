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
        description="Run Oriented R-CNN inference and unified Ultralytics OBB mAP50-95 evaluation."
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
    parser.add_argument("--mmrotate-python", default=sys.executable)
    parser.add_argument(
        "--metric-python",
        default=sys.executable,
        help="Python interpreter containing ultralytics (may differ from MMRotate Python).",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip-predict", action="store_true", help="Reuse <run-dir>/labels.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    run_dir = _resolve(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data = _resolve(args.data)
    labels = run_dir / "labels"
    metrics = run_dir / "custom_metrics.csv"

    predict_command = [
        args.mmrotate_python,
        str(ROOT / "scripts" / "models" / "oriented_rcnn" / "predict.py"),
        "--mmrotate-root",
        str(args.mmrotate_root.expanduser().resolve()),
        "--config",
        str(_resolve(args.config)),
        "--weights",
        str(_resolve(args.weights)),
        "--data",
        str(data),
        "--split",
        args.split,
        "--run-dir",
        str(run_dir),
        "--min-conf",
        str(args.min_conf),
        "--python",
        args.mmrotate_python,
        "--exist-ok",
    ]
    metric_command = [
        args.metric_python,
        str(ROOT / "scripts" / "evaluate_obb_prediction_labels.py"),
        "--data",
        str(data),
        "--pred-labels",
        str(labels),
        "--split",
        args.split,
        "--output",
        str(metrics),
    ]
    if args.plot:
        metric_command.append("--plot")

    recorded = ([] if args.skip_predict else [predict_command]) + [metric_command]
    (run_dir / "evaluate_command.txt").write_text(
        "\n".join(shlex.join(command) for command in recorded) + "\n",
        encoding="utf-8",
    )
    if not args.skip_predict:
        print("predict:", shlex.join(predict_command))
    print("metrics:", shlex.join(metric_command))
    if args.dry_run:
        return
    if not args.skip_predict:
        subprocess.run(predict_command, cwd=ROOT, check=True)
    elif not labels.is_dir():
        raise FileNotFoundError(f"prediction label directory not found: {labels}")
    subprocess.run(metric_command, cwd=ROOT, check=True)
    print(f"custom metrics: {metrics}")


if __name__ == "__main__":
    main()
