#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.classification_dataset import build_classification_dataset
from yolo11_obb.classification_labels import load_sheet_rows, rows_to_label_samples
from yolo11_obb.config import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cropped classification dataset from AnyLabeling OBB annotations.")
    parser.add_argument("--excel", type=Path, default=Path("outputs/label1_6_description.xlsx"))
    parser.add_argument("--source", type=Path, default=Path("已打标的数据202604/user1_2026-03-16_154843_anylabeling"))
    parser.add_argument("--output", type=Path, default=Path("datasets/classification/label5_ok_ng"))
    parser.add_argument("--label", default="label5")
    parser.add_argument("--target-column", default="tag1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excel = resolve_from_root(args.excel, ROOT)
    source = resolve_from_root(args.source, ROOT)
    output = resolve_from_root(args.output, ROOT)
    rows = load_sheet_rows(excel, args.label)
    samples = rows_to_label_samples(rows, target_column=args.target_column)
    report = build_classification_dataset(
        source=source,
        output=output,
        samples=samples,
        label_name=args.label,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"dataset: {output}")
    print(f"total_samples: {report['total_samples']}")
    print(f"classes: {report['classes']}")
    print(f"train: {report['train']}")
    print(f"test: {report['test']}")


if __name__ == "__main__":
    main()
