#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.pipeline_predict import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO11-OBB detection and ResNet18 label3/label5 classification.")
    parser.add_argument("--det-weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label3-weights", type=Path, required=True)
    parser.add_argument("--label5-weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-device", default=None)
    parser.add_argument("--cls-device", default=None)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--project", type=Path, default=Path("runs/pipeline"))
    parser.add_argument("--name", default="yolo11_resnet_pipeline")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = resolve_from_root(args.project, ROOT)
    output = project / args.name
    report = run_pipeline(
        det_weights=resolve_from_root(args.det_weights, ROOT),
        source=resolve_from_root(args.source, ROOT),
        label3_weights=resolve_from_root(args.label3_weights, ROOT),
        label5_weights=resolve_from_root(args.label5_weights, ROOT),
        output=output,
        imgsz=args.imgsz,
        det_conf=args.det_conf,
        det_device=args.det_device,
        cls_device=args.cls_device,
        cls_imgsz=args.cls_imgsz,
        exist_ok=args.exist_ok,
    )
    print(f"output: {report['output']}")
    print(f"images: {report['images']}")
    print(f"detections: {report['detections']}")
    print(f"classified: {report['classified']}")


if __name__ == "__main__":
    main()
