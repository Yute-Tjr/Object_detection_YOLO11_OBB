"""
用于拿训练好的模型预测新图片
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import PredictOptions, resolve_from_root
from yolo11_obb.runner import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO11-OBB prediction.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None, help="Example: 0, cpu, or mps")
    parser.add_argument("--project", type=Path, default=Path("runs/obb"))
    parser.add_argument("--name", default="terminal_obb_predict")
    parser.add_argument("--no-save-txt", action="store_true")
    parser.add_argument("--no-save-conf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project = resolve_from_root(args.project, ROOT)
    options = PredictOptions(
        model=args.model,
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save_txt=not args.no_save_txt,
        save_conf=not args.no_save_conf,
    )
    predict(options)


if __name__ == "__main__":
    main()
