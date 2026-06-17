#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.dataset_filter import create_label_subset_dataset


DEFAULT_SOURCE = ROOT / "datasets/154843_obb_train_val_test"
DEFAULT_OUTPUT = ROOT / "datasets/154843_obb_label1_6_train_val_test"
NAMES = {
    0: "label1",
    1: "label2",
    2: "label3",
    3: "label4",
    4: "label5",
    5: "label6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a label1-label6 YOLO-OBB dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_from_root(args.source, ROOT)
    output = resolve_from_root(args.output, ROOT)
    report = create_label_subset_dataset(
        source=source,
        output=output,
        keep_classes=set(NAMES),
        names=NAMES,
    )
    print(f"created: {output}")
    for split in ("train", "val", "test"):
        item = report[split]
        print(
            f"{split}: images={item['images']} labels={item['labels']} "
            f"kept={item['kept_objects']} removed={item['removed_objects']} "
            f"empty_labels={item['empty_labels']}"
        )
    print(f"data: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
