#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from yolo11_obb.classification_inference import (
    ImagePathDataset,
    classification_transform,
    collect_image_paths,
    load_resnet18_checkpoint,
    prediction_rows,
    select_device,
)
from yolo11_obb.classification_training import checkpoint_run_dir
from yolo11_obb.config import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict OK/NG labels with a trained ResNet18 classifier.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the run directory owning --weights.")
    parser.add_argument("--project", type=Path, default=None, help="Legacy override; requires --name.")
    parser.add_argument("--name", default=None, help="Legacy output subdirectory name.")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = resolve_from_root(args.weights, ROOT)
    source = resolve_from_root(args.source, ROOT)
    if args.output_dir is not None:
        run_dir = resolve_from_root(args.output_dir, ROOT)
    elif args.project is not None or args.name is not None:
        if args.project is None or args.name is None:
            raise ValueError("--project and --name must be used together")
        run_dir = resolve_from_root(args.project, ROOT) / args.name
    else:
        run_dir = checkpoint_run_dir(weights)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    image_paths = collect_image_paths(source)
    transform = classification_transform(args.imgsz)
    dataset = ImagePathDataset(image_paths, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model, classes, checkpoint = load_resnet18_checkpoint(weights, device)

    output_paths: list[Path] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, paths in loader:
            outputs = model(images.to(device))
            probabilities.extend(torch.softmax(outputs, dim=1).detach().cpu().tolist())
            output_paths.extend(Path(path) for path in paths)

    rows = prediction_rows(image_paths=output_paths, class_names=classes, probabilities=probabilities)
    fieldnames = ["image_path", "predicted_label", "confidence", *[f"prob_{name}" for name in classes]]
    with (run_dir / "predict_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["predicted_label"] for row in rows)
    args_payload = {
        "weights": str(weights),
        "source": str(source),
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": str(device),
        "workers": args.workers,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_score": checkpoint.get("score"),
        "classes": classes,
        "images": len(rows),
        "predicted_counts": dict(sorted(counts.items())),
    }
    (run_dir / "predict_args.yaml").write_text(yaml.safe_dump(args_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"run: {run_dir}")
    print(f"images: {len(rows)}")
    print(f"predicted_counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
