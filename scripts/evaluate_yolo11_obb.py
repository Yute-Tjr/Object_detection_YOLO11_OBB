from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import (
    EvalOptions,
    load_dataset_config,
    resolve_from_root,
    validate_dataset_layout,
)
from yolo11_obb.eval_metrics import default_custom_metrics_path, write_custom_metrics_csv
from yolo11_obb.runner import evaluate


DEFAULT_DATA = ROOT / "datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test/data.yaml"


def parse_args() -> argparse.Namespace:
    """
    定义命令行参数
    --model  指定 best.pt 或 last.pt
    --split  指定 train/val/test，默认 test
    :return:
    """
    parser = argparse.ArgumentParser(description="Evaluate YOLO11-OBB.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Example: 0, cpu, or mps")
    parser.add_argument("--project", type=Path, default=Path("runs/obb"))
    parser.add_argument("--name", default="terminal_obb_eval")
    return parser.parse_args()


def main() -> None:

    args = parse_args()
    args.data = resolve_from_root(args.data, ROOT)
    args.project = resolve_from_root(args.project, ROOT)
    dataset = load_dataset_config(args.data)
    validate_dataset_layout(dataset)
    options = EvalOptions(
        data=args.data,
        model=args.model,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )
    result = evaluate(options)
    metrics_path = default_custom_metrics_path(result, options.project / options.name)
    write_custom_metrics_csv(result, metrics_path)
    print(f"custom metrics: {metrics_path}")


if __name__ == "__main__":
    main()
