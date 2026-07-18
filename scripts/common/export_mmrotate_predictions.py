#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obb_detection.common.mmrotate_predictions import export_mmrotate_predictions
from yolo11_obb.config import IMAGE_EXTENSIONS, load_dataset_config, resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MMRotate prediction pickle samples to YOLO OBB label files."
    )
    parser.add_argument("--data", type=Path, default=Path("datasets/obb_thin_thick/data.yaml"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-conf", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = resolve_from_root(args.data, ROOT)
    predictions = resolve_from_root(args.predictions, ROOT)
    output = resolve_from_root(args.output, ROOT)
    dataset = load_dataset_config(data_yaml)
    image_paths = [
        path
        for path in dataset.splits[args.split].iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    written = export_mmrotate_predictions(
        predictions_path=predictions,
        image_paths=image_paths,
        output_dir=output,
        min_conf=args.min_conf,
    )
    print(f"labels: {output}")
    print(f"images: {written}")


if __name__ == "__main__":
    main()

