#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.dataset_splitter import create_train_test_dataset


DEFAULT_SOURCE = ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_obb_converted"
DEFAULT_OUTPUT = ROOT / "datasets" / "154843_obb_converted_label1_6_train_test"
NAMES = {
    0: "label1",
    1: "label2",
    2: "label3",
    3: "label4",
    4: "label5",
    5: "label6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an 80/20 train/test YOLO-OBB dataset for label1-label6.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_from_root(args.source, ROOT)
    output = resolve_from_root(args.output, ROOT)
    report = create_train_test_dataset(
        source=source,
        output=output,
        keep_classes=set(NAMES),
        names=NAMES,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    print(f"created: {output}")
    for split in ("train", "test"):
        item = report[split]
        print(
            f"{split}: groups={item['groups']} images={item['images']} "
            f"labels={item['labels']} kept={item['kept_objects']} "
            f"removed={item['removed_objects']} empty_labels={item['empty_labels']}"
        )
    print(f"data: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
