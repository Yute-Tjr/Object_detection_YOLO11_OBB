#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.anylabeling_to_obb import convert_anylabeling_directory
from yolo11_obb.config import resolve_from_root


DEFAULT_SOURCE = ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_anylabeling"
DEFAULT_OUTPUT = ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_obb_converted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert AnyLabeling 4-point JSON annotations to YOLO OBB txt files.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Only write txt labels; do not copy matching image files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Do not clip normalized coordinates to the 0-1 range.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = resolve_from_root(args.source, ROOT)
    output = resolve_from_root(args.output, ROOT)
    report = convert_anylabeling_directory(
        source=source,
        output=output,
        copy_images=not args.no_copy_images,
        overwrite=args.overwrite,
        clip=not args.no_clip,
    )
    print(f"created: {output}")
    print(f"json files: {report.json_files}")
    print(f"label files: {report.labels_written}")
    print(f"objects: {report.objects}")
    print(f"images copied: {report.images_copied}")


if __name__ == "__main__":
    main()
