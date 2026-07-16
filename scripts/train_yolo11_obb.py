from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import (
    TrainOptions,
    format_layout_report,
    load_dataset_config,
    resolve_from_root,
    validate_dataset_layout,
)
from yolo11_obb.runner import train


DEFAULT_DATA = ROOT / "datasets/obb_thin_thick/data.yaml"


def parse_args() -> argparse.Namespace:
    """
    定义命令行参数，比如：
      --model
      --epochs
      --imgsz
      --batch
      --device
      --project
      --name
      --dry-run
    :return:
    """
    parser = argparse.ArgumentParser(description="Train YOLO11-OBB.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11l-obb.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: 0, cpu, or mps")
    parser.add_argument("--project", type=Path, default=Path("runs/obb"))
    parser.add_argument("--name", default="terminal_obb_yolo11l")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Maximum image rotation augmentation degrees. Example: 5.0",
    )
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Disable validation during training. Use test split later with evaluate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset and print settings without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    """
    读取命令行参数
    修正 data/project 路径
    读取 data.yaml
    检查数据集图片和标签
    创建 TrainOptions
    如果是 dry-run，只打印信息
    否则调用 train()
    :return:
    """
    args = parse_args()
    args.data = resolve_from_root(args.data, ROOT)
    args.project = resolve_from_root(args.project, ROOT)
    dataset = load_dataset_config(args.data)
    report = validate_dataset_layout(dataset)
    print(format_layout_report(report))

    options = TrainOptions(
        data=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        exist_ok=args.exist_ok,
        validate=not args.no_val,
        degrees=args.degrees,
    )

    if args.dry_run:
        print("dry-run: training was not started")
        print(f"data: {dataset.data_yaml}")
        print(f"model: {options.model}")
        print(f"project: {options.project}")
        print(f"name: {options.name}")
        print(f"imgsz: {options.imgsz}")
        print(f"epochs: {options.epochs}")
        print(f"batch: {options.batch}")
        print(f"val: {options.validate}")
        print(f"degrees: {options.degrees}")
        return

    train(options)


if __name__ == "__main__":
    main()
