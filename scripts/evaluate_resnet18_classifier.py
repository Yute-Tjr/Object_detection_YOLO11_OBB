#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from yolo11_obb.classification_inference import (
    ImageFolderWithPaths,
    classification_transform,
    load_resnet18_checkpoint,
    prediction_rows,
    select_device,
)
from yolo11_obb.classification_training import (
    classification_metrics,
    confusion_counts,
    write_confusion_matrix_csv,
)
from yolo11_obb.config import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ResNet18 classifier on a labeled split.")
    parser.add_argument("--data", type=Path, default=Path("datasets/classification/label5_ok_ng"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/classification_eval"))
    parser.add_argument("--name", default="label5_resnet18_eval")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    weights = resolve_from_root(args.weights, ROOT)
    project = resolve_from_root(args.project, ROOT)
    run_dir = project / args.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    transform = classification_transform(args.imgsz)
    dataset = ImageFolderWithPaths(data / "images" / args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    model, classes, checkpoint = load_resnet18_checkpoint(weights, device)
    if list(dataset.classes) != classes:
        raise ValueError(f"dataset classes {dataset.classes} do not match checkpoint classes {classes}")

    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    true_labels: list[str] = []
    pred_labels: list[str] = []
    image_paths: list[Path] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            probs = torch.softmax(outputs, dim=1).detach().cpu().tolist()
            predictions = [max(range(len(row)), key=lambda index: row[index]) for row in probs]
            total_loss += float(loss.item()) * images.size(0)
            true_labels.extend(classes[index] for index in targets.detach().cpu().tolist())
            pred_labels.extend(classes[index] for index in predictions)
            image_paths.extend(Path(path) for path in paths)
            probabilities.extend(probs)

    metrics = classification_metrics(true_labels, pred_labels, classes)
    metrics["loss"] = total_loss / len(dataset)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "loss", "accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerow(
            {
                "split": args.split,
                "loss": f"{metrics['loss']:.6f}",
                "accuracy": f"{metrics['accuracy']:.6f}",
                "macro_f1": f"{metrics['macro_f1']:.6f}",
            }
        )

    rows = prediction_rows(image_paths=image_paths, class_names=classes, probabilities=probabilities)
    fieldnames = ["image_path", "true_label", "predicted_label", "confidence", *[f"prob_{name}" for name in classes]]
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, true_label in zip(rows, true_labels):
            row["true_label"] = true_label
            writer.writerow(row)

    counts = confusion_counts(true_labels, pred_labels, classes)
    write_confusion_matrix_csv(run_dir / "confusion_matrix.csv", counts, classes)
    args_payload = {
        "data": str(data),
        "weights": str(weights),
        "split": args.split,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": str(device),
        "workers": args.workers,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_score": checkpoint.get("score"),
        "classes": classes,
    }
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"run: {run_dir}")
    print(f"split: {args.split}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(f"macro_f1: {metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
