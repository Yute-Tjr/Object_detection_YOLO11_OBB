#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.rhino_dataset import create_rhino_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert the main YOLO-OBB dataset to RHINO DOTA annfiles.")
    parser.add_argument("--data", type=Path, default=Path("datasets/obb_thin_thick/data.yaml"))
    parser.add_argument("--output", type=Path, default=Path("datasets/rhino_obb"))
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of hard-linking them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    output = resolve_from_root(args.output, ROOT)
    report = create_rhino_dataset(data, output, image_mode="copy" if args.copy_images else "link")
    print(f"dataset: {report.output}")
    print(f"image_mode: {report.image_mode}")
    for split in ("train", "test"):
        print(f"{split}: images={report.images_by_split[split]} objects={report.objects_by_split[split]}")


if __name__ == "__main__":
    main()
