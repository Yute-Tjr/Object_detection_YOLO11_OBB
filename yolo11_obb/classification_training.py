from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def checkpoint_run_dir(weights_path: Path) -> Path:
    """Return the owning classifier run directory for <run>/weights/<checkpoint>."""
    path = Path(weights_path)
    if path.parent.name != "weights":
        raise ValueError(f"classifier checkpoint must be stored under a weights directory: {path}")
    return path.parent.parent


def discover_classes(class_names: Iterable[str]) -> Dict[str, int]:
    names = sorted(set(class_names))
    if len(names) < 2:
        raise ValueError("at least two classes are required")
    return {name: index for index, name in enumerate(names)}


def confusion_counts(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> Dict[Tuple[str, str], int]:
    counts = {(true_name, pred_name): 0 for true_name in class_names for pred_name in class_names}
    for true_name, pred_name in zip(true_labels, predicted_labels):
        counts[(true_name, pred_name)] += 1
    return counts


def classification_metrics(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    class_names: Sequence[str],
) -> Dict[str, object]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    total = len(true_labels)
    if total == 0:
        raise ValueError("no predictions to score")
    counts = confusion_counts(true_labels, predicted_labels, class_names)
    correct = sum(counts[(name, name)] for name in class_names)
    per_class: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    for name in class_names:
        tp = counts[(name, name)]
        fp = sum(counts[(other, name)] for other in class_names if other != name)
        fn = sum(counts[(name, other)] for other in class_names if other != name)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
    }


def write_confusion_matrix_csv(
    path: Path,
    counts: Mapping[Tuple[str, str], int],
    class_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for true_name in class_names:
            writer.writerow([true_name, *[counts[(true_name, pred_name)] for pred_name in class_names]])
