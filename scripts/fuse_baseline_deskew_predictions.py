#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.prediction_fusion import fuse_prediction_labels, write_fusion_manifest


DEFAULT_DESKEW_REPORT = (
    ROOT
    / "datasets"
    / "154843_after_20260121210219803_no_index1_deskewed_label1_thin_thick_train_test"
    / "deskew_report.csv"
)
DEFAULT_OUTPUT = ROOT / "runs" / "fusion" / "baseline_deskew_fused" / "labels"


def _class_id_set(values: list[int]) -> set[int]:
    return set(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse baseline predictions with deskew predictions mapped back to the original image coordinates.",
    )
    parser.add_argument("--baseline-labels", type=Path, required=True)
    parser.add_argument("--deskew-labels", type=Path, required=True)
    parser.add_argument("--deskew-report", type=Path, default=DEFAULT_DESKEW_REPORT)
    parser.add_argument("--output-labels", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--baseline-class-id",
        type=int,
        action="append",
        default=[1, 4, 6],
        help="Class IDs kept from the baseline predictions. Defaults: 1,4,6.",
    )
    parser.add_argument(
        "--deskew-class-id",
        type=int,
        action="append",
        default=[0, 2, 3, 5],
        help="Class IDs kept from the deskew predictions. Defaults: 0,2,3,5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_labels = resolve_from_root(args.baseline_labels, ROOT)
    deskew_labels = resolve_from_root(args.deskew_labels, ROOT)
    deskew_report = resolve_from_root(args.deskew_report, ROOT)
    output_labels = resolve_from_root(args.output_labels, ROOT)
    rows = fuse_prediction_labels(
        baseline_labels=baseline_labels,
        deskew_labels=deskew_labels,
        deskew_report=deskew_report,
        output_labels=output_labels,
        baseline_class_ids=_class_id_set(args.baseline_class_id),
        deskew_class_ids=_class_id_set(args.deskew_class_id),
        split=args.split,
    )
    manifest = write_fusion_manifest(rows, output_labels.parent / "fusion_manifest.csv")
    print(f"fused labels: {output_labels}")
    print(f"manifest: {manifest}")
    print(f"images fused: {len(rows)}")
    print(f"baseline class ids: {sorted(_class_id_set(args.baseline_class_id))}")
    print(f"deskew class ids: {sorted(_class_id_set(args.deskew_class_id))}")


if __name__ == "__main__":
    main()
