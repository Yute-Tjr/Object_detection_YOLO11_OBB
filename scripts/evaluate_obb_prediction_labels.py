#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.prediction_label_eval import evaluate_prediction_labels, write_prediction_metrics_csv


DEFAULT_DATA = ROOT / "datasets" / "154843_after_20260121210219803_no_index1_label1_thin_thick_train_test" / "data.yaml"
DEFAULT_OUTPUT = ROOT / "runs" / "fusion" / "prediction_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO-OBB prediction label files against a dataset split.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--pred-labels", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    pred_labels = resolve_from_root(args.pred_labels, ROOT)
    output = resolve_from_root(args.output, ROOT)
    rows = evaluate_prediction_labels(data_yaml=data, pred_labels=pred_labels, split=args.split)
    write_prediction_metrics_csv(rows, output)
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
