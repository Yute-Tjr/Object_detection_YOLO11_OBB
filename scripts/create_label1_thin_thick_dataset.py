#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.label1_thin_thick_dataset import (
    LABEL1_THIN_THICK_NAMES,
    create_label1_thin_thick_dataset,
)


DEFAULT_SOURCE = ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_obb_converted"
DEFAULT_OUTPUT = ROOT / "datasets" / "154843_obb_converted_label1_thin_thick_train_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a train/test YOLO-OBB dataset that splits label1 into thin/thick classes.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--top-edge-threshold-px",
        type=float,
        required=True,
        help="label1 objects with top-edge width below this pixel threshold become label1_thin.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_from_root(args.source, ROOT)
    output = resolve_from_root(args.output, ROOT)
    report = create_label1_thin_thick_dataset(
        source=source,
        output=output,
        top_edge_threshold_px=args.top_edge_threshold_px,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print(f"created: {output}")
    print(f"data: {output / 'data.yaml'}")
    print(f"label1 width report: {output / 'label1_top_edge_widths.csv'}")
    for split in ("train", "test"):
        item = report[split]
        print(
            f"{split}: groups={item['groups']} images={item['images']} "
            f"labels={item['labels']} kept={item['kept_objects']} "
            f"removed={item['removed_objects']} empty_labels={item['empty_labels']}"
        )
        class_counts = item["class_counts"]
        if isinstance(class_counts, dict):
            for class_id in sorted(LABEL1_THIN_THICK_NAMES):
                print(f"  {LABEL1_THIN_THICK_NAMES[class_id]}: {class_counts[class_id]}")


if __name__ == "__main__":
    main()
