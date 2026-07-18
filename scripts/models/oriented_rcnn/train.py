#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obb_detection.models.oriented_rcnn import (
    find_base_config,
    load_class_names,
    render_oriented_rcnn_config,
    validate_dota_dataset,
)
from yolo11_obb.config import resolve_from_root


PREFLIGHT_FILES = {"config.py", "train_command.txt"}


def _preflight_only(run_dir: Path) -> bool:
    return run_dir.is_dir() and all(
        path.is_file() and path.name in PREFLIGHT_FILES for path in run_dir.iterdir()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an MMRotate 1.x config and train Oriented R-CNN R50-FPN."
    )
    parser.add_argument(
        "--mmrotate-root",
        "--rhino-root",
        dest="mmrotate_root",
        type=Path,
        required=True,
        help="MMRotate 1.x checkout; --rhino-root is accepted when RHINO contains MMRotate configs.",
    )
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=Path("datasets/rhino_obb"))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--flip-prob", type=float, default=0.75)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--max-keep-ckpts", type=int, default=10)
    parser.add_argument("--project", type=Path, default=Path("runs/oriented_rcnn"))
    parser.add_argument("--name", default="oriented_rcnn_r50_fpn_e50_img1280_b2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mmrotate_root = args.mmrotate_root.expanduser().resolve()
    train_tool = mmrotate_root / "tools" / "train.py"
    if not train_tool.is_file():
        raise FileNotFoundError(f"MMRotate training tool not found: {train_tool}")
    base_config = find_base_config(mmrotate_root, args.base_config)
    data_root = resolve_from_root(args.data, ROOT)
    report = validate_dota_dataset(
        data_root,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    project = resolve_from_root(args.project, ROOT)
    run_dir = project / args.name
    if (
        run_dir.exists()
        and any(run_dir.iterdir())
        and not _preflight_only(run_dir)
        and not args.exist_ok
    ):
        raise FileExistsError(f"run directory is not empty: {run_dir}; pass --exist-ok to reuse it")
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.py"
    config_path.write_text(
        render_oriented_rcnn_config(
            base_config=base_config,
            data_root=data_root,
            class_names=load_class_names(data_root),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            learning_rate=args.lr,
            flip_prob=args.flip_prob,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            checkpoint_interval=args.checkpoint_interval,
            max_keep_ckpts=args.max_keep_ckpts,
        ),
        encoding="utf-8",
    )
    command = [args.python, str(train_tool), str(config_path), "--work-dir", str(run_dir)]
    if args.resume:
        command.append("--resume")
    (run_dir / "train_command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    print(f"dataset: {data_root}")
    print("splits:", ", ".join(f"{name}={counts[0]}" for name, counts in report.items()))
    print(f"base config: {base_config}")
    print(f"generated config: {config_path}")
    print(f"run: {run_dir}")
    print("command:", shlex.join(command))
    if args.val_split == args.test_split == "test":
        print("warning: validation and final evaluation both use test; do not select the final model on this split")
    if not args.dry_run:
        subprocess.run(command, cwd=mmrotate_root, check=True)


if __name__ == "__main__":
    main()
