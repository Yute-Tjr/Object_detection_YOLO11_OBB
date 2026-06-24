#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.label1_thin_thick_dataset import LABEL1_THIN_THICK_NAMES
from yolo11_obb.recent_anylabeling_dataset import create_recent_label1_thin_thick_dataset


DEFAULT_SOURCE = ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_anylabeling"
DEFAULT_CONVERTED_OUTPUT = (
    ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_after_20260121210219803_obb_converted"
)
DEFAULT_DATASET_OUTPUT = ROOT / "datasets" / "154843_after_20260121210219803_label1_thin_thick_train_test"
DEFAULT_AFTER_STEM = "20260121210219803"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a post-marker AnyLabeling dataset and split label1 into thin/thick OBB classes.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--converted-output", type=Path, default=DEFAULT_CONVERTED_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_OUTPUT)
    parser.add_argument("--after-stem", default=DEFAULT_AFTER_STEM)
    parser.add_argument("--top-edge-threshold-px", type=float, default=164.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-clip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_from_root(args.source, ROOT)
    converted_output = resolve_from_root(args.converted_output, ROOT)
    output = resolve_from_root(args.output, ROOT)
    report = create_recent_label1_thin_thick_dataset(
        source=source,
        converted_output=converted_output,
        dataset_output=output,
        after_stem=args.after_stem,
        top_edge_threshold_px=args.top_edge_threshold_px,
        train_ratio=args.train_ratio,
        seed=args.seed,
        clip=not args.no_clip,
    )

    print(f"converted: {converted_output}")
    print(f"dataset: {output}")
    print(f"data: {output / 'data.yaml'}")
    print(f"selected json files: {report.selected_json_files}")
    print(f"converted objects: {report.converted.objects}")
    print(f"images copied: {report.converted.images_copied}")
    for split in ("train", "test"):
        item = report.dataset[split]
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
