#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.rhino_config import RHINO_VARIANTS, render_rhino_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a RHINO config and train one OBB variant.")
    parser.add_argument("--rhino-root", type=Path, required=True, help="Path to the SIAnalytics/RHINO checkout.")
    parser.add_argument("--data", type=Path, default=Path("datasets/rhino_obb"))
    parser.add_argument("--variant", choices=RHINO_VARIANTS, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/rhino"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _classes(data: Path) -> list[str]:
    classes_path = data / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"RHINO classes file not found: {classes_path}; run create_rhino_dataset.py first")
    classes = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"no classes found in {classes_path}")
    return classes


def main() -> None:
    args = parse_args()
    rhino_root = args.rhino_root.expanduser().resolve()
    data = (ROOT / args.data).resolve() if not args.data.is_absolute() else args.data.resolve()
    project = (ROOT / args.project).resolve() if not args.project.is_absolute() else args.project.resolve()
    base_config = rhino_root / "configs/rhino/rhino_phc_haus-4scale_r50_2xb2-36e_dotav2.py"
    train_tool = rhino_root / "tools/train.py"
    if not base_config.exists() or not train_tool.exists():
        raise FileNotFoundError("invalid --rhino-root: expected configs/rhino and tools/train.py")
    run_dir = project / args.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.py"
    config_path.write_text(
        render_rhino_config(
            base_config=base_config,
            data_root=data,
            class_names=_classes(data),
            variant=args.variant,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
        ),
        encoding="utf-8",
    )
    command = [args.python, str(train_tool), str(config_path), "--work-dir", str(run_dir)]
    if args.resume:
        command.append("--resume")
    (run_dir / "train_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    print(f"config: {config_path}")
    print(f"run: {run_dir}")
    print("command:", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=rhino_root, check=True)


if __name__ == "__main__":
    main()
