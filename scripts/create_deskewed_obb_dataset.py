#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.deskew_dataset import create_deskewed_obb_dataset


DEFAULT_SOURCE_DATA = (
    ROOT / "datasets" / "154843_after_20260121210219803_no_index1_label1_thin_thick_train_test" / "data.yaml"
)
DEFAULT_OUTPUT = ROOT / "datasets" / "154843_after_20260121210219803_no_index1_deskewed_label1_thin_thick_train_test"
DEFAULT_ANGLE_CLASS_IDS = [2, 3, 4, 5, 6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deskewed YOLO-OBB dataset by rotating each image and OBB labels together.",
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--angle-class-id",
        type=int,
        action="append",
        default=DEFAULT_ANGLE_CLASS_IDS.copy(),
        help="Class IDs used to estimate the per-image deskew angle. Defaults to label2-label6 in thin/thick datasets.",
    )
    parser.add_argument(
        "--all-classes-for-angle",
        action="store_true",
        help="Use all classes instead of --angle-class-id values to estimate the deskew angle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    output = resolve_from_root(args.output, ROOT)
    angle_class_ids = None if args.all_classes_for_angle else set(args.angle_class_id)
    report = create_deskewed_obb_dataset(
        source_data=data,
        output=output,
        angle_class_ids=angle_class_ids,
    )

    print(f"source data: {data}")
    print(f"dataset: {output}")
    print(f"data: {output / 'data.yaml'}")
    for split in sorted(report):
        item = report[split]
        print(
            f"{split}: images={item['images']} labels={item['labels']} "
            f"objects={item['objects']} mean_abs_angle={item['mean_abs_angle']:.3f}"
        )


if __name__ == "__main__":
    main()
