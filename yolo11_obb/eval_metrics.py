from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


CUSTOM_MAP_COLUMNS = ("mAP50", "mAP80", "mAP85", "mAP90", "mAP95")
IOU_COLUMN_INDEX = {
    "mAP50": 0,
    "mAP80": 6,
    "mAP85": 7,
    "mAP90": 8,
    "mAP95": 9,
}
CSV_COLUMNS = ("class", "precision", "recall") + CUSTOM_MAP_COLUMNS


def _as_float(value: Any) -> float:
    return float(value.item() if hasattr(value, "item") else value)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _sequence_value(values: Any, index: int) -> Optional[float]:
    if values is None:
        return None
    try:
        return _as_float(values[index])
    except (IndexError, TypeError):
        return None


def _class_name(names: Mapping[int, str], class_id: int) -> str:
    return str(names.get(class_id, class_id))


def _metric_value_text(value: Any) -> str:
    return "" if value is None else f"{_as_float(value):.6f}"


def build_custom_map_rows(metrics_result: Any) -> List[Dict[str, Optional[float]]]:
    box = getattr(metrics_result, "box", None)
    if box is None:
        raise ValueError("Evaluation result does not contain box metrics")

    all_ap = getattr(box, "all_ap", None)
    if all_ap is None:
        raise ValueError("Evaluation result does not contain box.all_ap")

    class_ids = list(getattr(box, "ap_class_index", range(len(all_ap))))
    names = getattr(metrics_result, "names", {})
    if not isinstance(names, Mapping):
        names = {}

    precision_values = getattr(box, "p", None)
    recall_values = getattr(box, "r", None)
    class_rows: List[Dict[str, Optional[float]]] = []

    for row_index, class_id in enumerate(class_ids):
        ap_values = all_ap[row_index]
        row: Dict[str, Optional[float]] = {
            "class": _class_name(names, int(class_id)),
            "precision": _sequence_value(precision_values, row_index),
            "recall": _sequence_value(recall_values, row_index),
        }
        for column in CUSTOM_MAP_COLUMNS:
            row[column] = _as_float(ap_values[IOU_COLUMN_INDEX[column]])
        class_rows.append(row)

    all_row: Dict[str, Optional[float]] = {
        "class": "all",
        "precision": _mean([row["precision"] for row in class_rows]),
        "recall": _mean([row["recall"] for row in class_rows]),
    }
    for column in CUSTOM_MAP_COLUMNS:
        all_row[column] = _mean([row[column] for row in class_rows])

    return [all_row] + class_rows


def write_custom_metrics_csv(metrics_result: Any, output_path: Path) -> Path:
    rows = build_custom_map_rows(metrics_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: row[column] if column == "class" else _metric_value_text(row[column])
                 for column in CSV_COLUMNS}
            )
    return output_path


def default_custom_metrics_path(metrics_result: Any, fallback_dir: Path) -> Path:
    save_dir = getattr(metrics_result, "save_dir", None)
    output_dir = Path(save_dir) if save_dir else fallback_dir
    return output_dir / "custom_metrics.csv"
