#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.eval_metrics import build_custom_map_rows, write_custom_metrics_csv
from yolo11_obb.prediction_label_eval import evaluate_prediction_labels_ultralytics


DEFAULT_DATA = ROOT / "datasets" / "obb_thin_thick" / "data.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "obb" / "prediction_custom_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-OBB prediction label files against a dataset split.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pred-labels", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metric",
        choices=["ultralytics"],
        default="ultralytics",
        help="Only Ultralytics OBB metrics are supported.",
    )
    parser.add_argument("--plot", action="store_true", help="Save Ultralytics PR/F1/P/R curves.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    pred_labels = resolve_from_root(args.pred_labels, ROOT)
    output = resolve_from_root(args.output, ROOT)

    result = evaluate_prediction_labels_ultralytics(
        data_yaml=data,
        pred_labels=pred_labels,
        split=args.split,
        save_dir=output.parent,
        plot=args.plot,
    )
    write_custom_metrics_csv(result, output)
    rows = build_custom_map_rows(result)

    print(f"metrics: {output}")
    for row in rows:
        print(
            f"{row['class']},"
            f"{float(row['precision']):.6f},"
            f"{float(row['recall']):.6f},"
            f"{float(row['mAP50']):.6f},"
            f"{float(row['mAP80']):.6f},"
            f"{float(row['mAP85']):.6f},"
            f"{float(row['mAP90']):.6f},"
            f"{float(row['mAP95']):.6f}"
        )


if __name__ == "__main__":
    main()
