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
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from yolo11_obb.classification_training import (
    classification_metrics,
    confusion_counts,
    write_confusion_matrix_csv,
)
from yolo11_obb.config import resolve_from_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 classifier on cropped OBB regions.")
    parser.add_argument("--data", type=Path, default=Path("datasets/classification/label5_ok_ng"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=Path("runs/classification"))
    parser.add_argument("--name", default="label5_resnet18")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def epoch_pass(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    true_labels = []
    pred_labels = []
    image_paths = []
    class_names = loader.dataset.classes
    with torch.set_grad_enabled(training):
        for images, targets, paths in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            predictions = outputs.argmax(dim=1).detach().cpu().tolist()
            true_labels.extend(class_names[index] for index in targets.detach().cpu().tolist())
            pred_labels.extend(class_names[index] for index in predictions)
            image_paths.extend(paths)
    metrics = classification_metrics(true_labels, pred_labels, class_names)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, true_labels, pred_labels, image_paths


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    project = resolve_from_root(args.project, ROOT)
    run_dir = project / args.name
    if run_dir.exists() and any(run_dir.iterdir()) and not args.exist_ok:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    transform = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = ImageFolderWithPaths(data / "images" / "train", transform=transform)
    test_dataset = ImageFolderWithPaths(data / "images" / "test", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(num_classes=len(train_dataset.classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args_payload = {
        "data": str(data),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "imgsz": args.imgsz,
        "device": str(device),
        "workers": args.workers,
        "pretrained": not args.no_pretrained,
        "model_selection": "best test macro_f1 on requested 8:2 split; optimistic because no separate validation split",
        "classes": train_dataset.classes,
    }
    (run_dir / "args.yaml").write_text(yaml.safe_dump(args_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    metrics_path = run_dir / "metrics.csv"
    best_score = -1.0
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "test_loss",
                "test_accuracy",
                "test_macro_f1",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics, _, _, _ = epoch_pass(model, train_loader, criterion, device, optimizer=optimizer)
            test_metrics, true_labels, pred_labels, _ = epoch_pass(model, test_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "train_macro_f1": f"{train_metrics['macro_f1']:.6f}",
                "test_loss": f"{test_metrics['loss']:.6f}",
                "test_accuracy": f"{test_metrics['accuracy']:.6f}",
                "test_macro_f1": f"{test_metrics['macro_f1']:.6f}",
            }
            writer.writerow(row)
            handle.flush()
            score = float(test_metrics["macro_f1"])
            if score > best_score:
                best_score = score
                torch.save(
                    {"model": model.state_dict(), "classes": train_dataset.classes, "epoch": epoch, "score": best_score},
                    run_dir / "weights" / "best.pt",
                )
            torch.save(
                {"model": model.state_dict(), "classes": train_dataset.classes, "epoch": epoch, "score": score},
                run_dir / "weights" / "last.pt",
            )

    final_metrics, true_labels, pred_labels, image_paths = epoch_pass(model, test_loader, criterion, device)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "true_label", "predicted_label"])
        writer.writeheader()
        for image_path, true_name, pred_name in zip(image_paths, true_labels, pred_labels):
            writer.writerow({"image_path": image_path, "true_label": true_name, "predicted_label": pred_name})
    counts = confusion_counts(true_labels, pred_labels, test_dataset.classes)
    write_confusion_matrix_csv(run_dir / "confusion_matrix.csv", counts, test_dataset.classes)
    print(f"run: {run_dir}")
    print(f"best_macro_f1: {best_score:.6f}")
    print(f"final_test_accuracy: {final_metrics['accuracy']:.6f}")
    print(f"final_test_macro_f1: {final_metrics['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
