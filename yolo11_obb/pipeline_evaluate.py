from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .classification_labels import load_sheet_rows, rows_to_label_samples
from .config import load_dataset_config
from .pipeline_predict import PIPELINE_LABELS, final_result_from_selected


TRUE_CLASSES = ("NG", "OK")
PREDICTION_CLASSES = ("NG", "OK", "UNKNOWN")
PIPELINE_EVAL_FIELDNAMES = [
    "image_path",
    "image_name",
    "true_label3",
    "pred_label3",
    "label3_correct",
    "true_label5",
    "pred_label5",
    "label5_correct",
    "both_labels_correct",
    "true_final",
    "pred_final",
    "final_correct",
    "warnings",
    "visualization_path",
]
PIPELINE_METRIC_FIELDNAMES = [
    "target",
    "total",
    "correct",
    "accuracy",
    "macro_f1",
    "known_predictions",
    "unknown_predictions",
    "truth_missing",
]


MetricMap = Dict[str, int | float]
ConfusionMatrix = Dict[Tuple[str, str], int]


@dataclass(frozen=True)
class PipelineEvaluation:
    rows: List[Dict[str, str]]
    metrics: Dict[str, MetricMap]
    confusion_matrices: Dict[str, ConfusionMatrix]


def _canonical_status(value: object) -> str:
    text = "" if value is None else str(value).strip().upper()
    return text if text in TRUE_CLASSES else "UNKNOWN"


def _image_stem(row: Mapping[str, str]) -> str:
    image_name = str(row.get("image_name") or Path(str(row.get("image_path", ""))).name)
    return Path(image_name).stem


def truth_from_sheet_rows_by_label(
    sheet_rows_by_label: Mapping[str, Iterable[Mapping[str, object]]],
    target_column: str,
) -> Dict[str, Dict[str, str]]:
    truth: Dict[str, Dict[str, str]] = {}
    for label, rows in sheet_rows_by_label.items():
        mapping: Dict[str, str] = {}
        for sample in rows_to_label_samples(rows, target_column=target_column):
            existing = mapping.get(sample.image_name)
            if existing is not None and existing != sample.class_name:
                raise ValueError(
                    f"conflicting truth labels for {label} {sample.image_name}: "
                    f"{existing} vs {sample.class_name}"
                )
            mapping[sample.image_name] = sample.class_name
        truth[label] = mapping
    return truth


def load_truth_from_excel(
    excel_path: Path,
    labels: Sequence[str] = PIPELINE_LABELS,
    target_column: str = "tag1",
) -> Dict[str, Dict[str, str]]:
    sheet_rows = {
        label: load_sheet_rows(Path(excel_path).expanduser().resolve(), label)
        for label in labels
    }
    return truth_from_sheet_rows_by_label(sheet_rows, target_column=target_column)


def resolve_split_source(data_yaml: Path, split: str = "test") -> Path:
    dataset = load_dataset_config(data_yaml)
    if split not in dataset.splits:
        raise ValueError(f"dataset split not found: {split}")
    return dataset.splits[split]


def _empty_confusion_matrix() -> ConfusionMatrix:
    return {(true_label, pred_label): 0 for true_label in TRUE_CLASSES for pred_label in PREDICTION_CLASSES}


def _score_predictions(true_labels: Sequence[str], predicted_labels: Sequence[str]) -> tuple[MetricMap, ConfusionMatrix]:
    if len(true_labels) != len(predicted_labels):
        raise ValueError("true_labels and predicted_labels must have the same length")
    counts = _empty_confusion_matrix()
    for true_label, pred_label in zip(true_labels, predicted_labels):
        counts[(true_label, pred_label)] += 1

    total = len(true_labels)
    correct = sum(1 for true_label, pred_label in zip(true_labels, predicted_labels) if true_label == pred_label)
    unknown_predictions = sum(1 for pred_label in predicted_labels if pred_label == "UNKNOWN")
    known_predictions = total - unknown_predictions
    f1_values: List[float] = []
    for class_name in TRUE_CLASSES:
        tp = counts[(class_name, class_name)]
        fp = sum(counts[(other, class_name)] for other in TRUE_CLASSES if other != class_name)
        fn = sum(counts[(class_name, pred_name)] for pred_name in PREDICTION_CLASSES if pred_name != class_name)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1_values.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))

    metrics: MetricMap = {
        "total": total,
        "correct": correct,
        "accuracy": 0.0 if total == 0 else correct / total,
        "macro_f1": 0.0 if total == 0 else sum(f1_values) / len(f1_values),
        "known_predictions": known_predictions,
        "unknown_predictions": unknown_predictions,
        "truth_missing": 0,
    }
    return metrics, counts


def _metric_without_f1(total: int, correct: int, truth_missing: int = 0) -> MetricMap:
    return {
        "total": total,
        "correct": correct,
        "accuracy": 0.0 if total == 0 else correct / total,
        "macro_f1": 0.0,
        "known_predictions": total,
        "unknown_predictions": 0,
        "truth_missing": truth_missing,
    }


def evaluate_pipeline_summary_rows(
    summary_rows: Iterable[Mapping[str, str]],
    truth_by_label: Mapping[str, Mapping[str, str]],
    labels: Sequence[str] = PIPELINE_LABELS,
) -> PipelineEvaluation:
    evaluation_rows: List[Dict[str, str]] = []
    true_by_label: Dict[str, List[str]] = {label: [] for label in labels}
    pred_by_label: Dict[str, List[str]] = {label: [] for label in labels}
    truth_missing_by_label: Dict[str, int] = {label: 0 for label in labels}
    true_final_labels: List[str] = []
    pred_final_labels: List[str] = []
    both_total = 0
    both_correct = 0
    both_truth_missing = 0

    for summary_row in summary_rows:
        stem = _image_stem(summary_row)
        output_row = {
            "image_path": str(summary_row.get("image_path", "")),
            "image_name": str(summary_row.get("image_name", "")),
            "true_label3": "",
            "pred_label3": "",
            "label3_correct": "",
            "true_label5": "",
            "pred_label5": "",
            "label5_correct": "",
            "both_labels_correct": "",
            "true_final": "",
            "pred_final": _canonical_status(summary_row.get("final_result")),
            "final_correct": "",
            "warnings": str(summary_row.get("warnings", "")),
            "visualization_path": str(summary_row.get("visualization_path", "")),
        }

        selected_truth: Dict[str, str] = {}
        selected_predictions: Dict[str, str] = {}
        for label in labels:
            true_label = truth_by_label.get(label, {}).get(stem)
            pred_label = _canonical_status(summary_row.get(f"{label}_pred"))
            output_row[f"true_{label}"] = true_label or ""
            output_row[f"pred_{label}"] = pred_label
            selected_predictions[label] = pred_label
            if true_label is None:
                truth_missing_by_label[label] += 1
                continue
            selected_truth[label] = true_label
            true_by_label[label].append(true_label)
            pred_by_label[label].append(pred_label)
            output_row[f"{label}_correct"] = "1" if pred_label == true_label else "0"

        if all(label in selected_truth for label in labels):
            both_total += 1
            is_both_correct = all(selected_predictions[label] == selected_truth[label] for label in labels)
            if is_both_correct:
                both_correct += 1
            output_row["both_labels_correct"] = "1" if is_both_correct else "0"
            true_final = final_result_from_selected(selected_truth)
            pred_final = _canonical_status(summary_row.get("final_result"))
            true_final_labels.append(true_final)
            pred_final_labels.append(pred_final)
            output_row["true_final"] = true_final
            output_row["pred_final"] = pred_final
            output_row["final_correct"] = "1" if pred_final == true_final else "0"
        else:
            both_truth_missing += 1

        evaluation_rows.append(output_row)

    metrics: Dict[str, MetricMap] = {}
    confusion_matrices: Dict[str, ConfusionMatrix] = {}
    for label in labels:
        metric, matrix = _score_predictions(true_by_label[label], pred_by_label[label])
        metric["truth_missing"] = truth_missing_by_label[label]
        metrics[label] = metric
        confusion_matrices[label] = matrix

    metrics["both_labels"] = _metric_without_f1(both_total, both_correct, both_truth_missing)
    final_metric, final_matrix = _score_predictions(true_final_labels, pred_final_labels)
    final_metric["truth_missing"] = both_truth_missing
    metrics["final_result"] = final_metric
    confusion_matrices["final_result"] = final_matrix
    return PipelineEvaluation(rows=evaluation_rows, metrics=metrics, confusion_matrices=confusion_matrices)


def evaluate_pipeline_summary(
    summary_csv: Path,
    truth_by_label: Mapping[str, Mapping[str, str]],
    labels: Sequence[str] = PIPELINE_LABELS,
) -> PipelineEvaluation:
    with Path(summary_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return evaluate_pipeline_summary_rows(rows, truth_by_label, labels=labels)


def _format_metric_value(value: int | float) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_matrix(path: Path, matrix: ConfusionMatrix) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *PREDICTION_CLASSES])
        for true_label in TRUE_CLASSES:
            writer.writerow([true_label, *[matrix[(true_label, pred_label)] for pred_label in PREDICTION_CLASSES]])


def write_pipeline_evaluation(output: Path, evaluation: PipelineEvaluation) -> None:
    output = Path(output)
    _write_csv(output / "pipeline_eval_predictions.csv", PIPELINE_EVAL_FIELDNAMES, evaluation.rows)

    metric_rows: List[Dict[str, str]] = []
    for target, values in evaluation.metrics.items():
        row = {"target": target}
        row.update({name: _format_metric_value(values[name]) for name in PIPELINE_METRIC_FIELDNAMES if name != "target"})
        metric_rows.append(row)
    _write_csv(output / "pipeline_eval_metrics.csv", PIPELINE_METRIC_FIELDNAMES, metric_rows)

    for target, matrix in evaluation.confusion_matrices.items():
        _write_confusion_matrix(output / f"pipeline_{target}_confusion_matrix.csv", matrix)

    report_lines = ["pipeline evaluation"]
    for target, values in evaluation.metrics.items():
        report_lines.append(
            f"{target}: total={values['total']} correct={values['correct']} "
            f"accuracy={values['accuracy']:.6f} macro_f1={values['macro_f1']:.6f} "
            f"unknown={values['unknown_predictions']} truth_missing={values['truth_missing']}"
        )
    (output / "pipeline_eval_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
