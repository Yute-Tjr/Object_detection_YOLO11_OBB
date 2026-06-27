#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import resolve_from_root
from yolo11_obb.recent_anylabeling_dataset import sample_order_key


DEFAULT_SOURCE = ROOT / "已打标的数据202604" / "user1_2026-03-16_154843_anylabeling"
DEFAULT_OUTPUT = ROOT / "runs" / "analysis" / "description_distribution_after_20260121210219803"
DEFAULT_AFTER_STEM = "20260121210219803"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize AnyLabeling shape labels by flags.description after a timestamp marker.",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--after-stem", default=DEFAULT_AFTER_STEM)
    parser.add_argument("--exclude-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def selected_json_paths(source: Path, after_stem: str, exclude_indices: Iterable[int]) -> List[Path]:
    after_key = sample_order_key(after_stem)
    excluded = set(exclude_indices)
    paths = sorted(source.glob("*.json"), key=lambda path: (sample_order_key(path.stem), path.stem))
    return [
        path
        for path in paths
        if sample_order_key(path.stem) > after_key and sample_order_key(path.stem)[1] not in excluded
    ]


def raw_description(shape: Dict[str, object]) -> str:
    flags = shape.get("flags")
    if isinstance(flags, dict):
        value = flags.get("description", "")
    else:
        value = ""
    if value is None:
        return ""
    return str(value).strip()


def _compact_description(value: str) -> str:
    value = value.strip()
    for old, new in {
        "：": ":",
        "；": ";",
        "，": ",",
        "、": ",",
        " ": "",
        "\t": "",
    }.items():
        value = value.replace(old, new)
    while value.endswith((",", ";", ":")):
        value = value[:-1]
    return value


def classify_description(description: str) -> str:
    desc = _compact_description(description)
    upper = desc.upper()
    if not desc:
        return "EMPTY"

    if upper == "OK":
        return "OK"
    if upper.startswith("OK"):
        suffix = desc[2:].lstrip(",:;").upper()
        if suffix in {"B", "G", "R", "W"}:
            return f"OK_{suffix}"
        return "OK_OTHER"

    text = desc
    if upper.startswith("NG"):
        text = desc[2:].lstrip(",:;")
    text = text.replace("线芯", "芯线").replace("哪个", "").replace("X", "")

    if "飞出" in text:
        return "NG_铜线飞出"
    if "未露出" in text:
        return "NG_芯线未露出"
    if any(token in text for token in ("露出少", "过少", "太少", "偏少")):
        if "铜线" in text:
            return "NG_铜线露出少"
        return "NG_芯线露出少"
    if any(token in text for token in ("露出多", "过多", "外露", "完全外露", "过度", "露出S", "露出s")):
        if "铜线" in text:
            return "NG_铜线露出多"
        return "NG_芯线露出多"

    if upper.startswith("NG"):
        return "NG_OTHER"
    return "OTHER"


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ordered_categories(categories: Iterable[str]) -> List[str]:
    preferred = [
        "EMPTY",
        "OK",
        "OK_B",
        "OK_G",
        "OK_R",
        "OK_W",
        "OK_OTHER",
        "NG_芯线露出多",
        "NG_芯线露出少",
        "NG_芯线未露出",
        "NG_铜线露出多",
        "NG_铜线露出少",
        "NG_铜线飞出",
        "NG_OTHER",
        "OTHER",
    ]
    category_set = set(categories)
    ordered = [category for category in preferred if category in category_set]
    ordered.extend(sorted(category_set - set(ordered)))
    return ordered


def summarize(source: Path, after_stem: str, exclude_indices: List[int], output: Path) -> None:
    paths = selected_json_paths(source, after_stem, exclude_indices)
    if not paths:
        raise ValueError("no JSON files selected")

    detail_rows: List[Dict[str, object]] = []
    exact_counter: Counter[Tuple[str, str, str]] = Counter()
    category_counter: Counter[Tuple[str, str]] = Counter()
    label_counter: Counter[str] = Counter()
    examples: Dict[Tuple[str, str, str], str] = {}

    for json_path in paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_path = str(data.get("imagePath", ""))
        timestamp, index = sample_order_key(json_path.stem)
        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            continue
        for shape_index, shape in enumerate(shapes):
            if not isinstance(shape, dict):
                continue
            label = str(shape.get("label", ""))
            description = raw_description(shape)
            category = classify_description(description)
            label_counter[label] += 1
            exact_counter[(label, category, description)] += 1
            category_counter[(label, category)] += 1
            examples.setdefault((label, category, description), json_path.name)
            detail_rows.append(
                {
                    "json": json_path.name,
                    "image": image_path,
                    "order_timestamp": timestamp,
                    "order_index": index,
                    "shape_index": shape_index,
                    "label": label,
                    "description": description,
                    "category": category,
                }
            )

    category_rows: List[Dict[str, object]] = []
    for (label, category), count in sorted(category_counter.items()):
        total = label_counter[label]
        category_rows.append(
            {
                "label": label,
                "category": category,
                "count": count,
                "percent_of_label": f"{count / total:.6f}" if total else "0",
            }
        )

    pivot_categories = ordered_categories(category for _, category in category_counter)
    pivot_rows: List[Dict[str, object]] = []
    for label in sorted(label_counter):
        category_sum = sum(category_counter[(label, category)] for category in pivot_categories)
        row: Dict[str, object] = {
            "label": label,
            "total_shapes": label_counter[label],
            "category_sum": category_sum,
            "missing_from_sum": label_counter[label] - category_sum,
        }
        for category in pivot_categories:
            row[category] = category_counter[(label, category)]
        pivot_rows.append(row)

    exact_rows: List[Dict[str, object]] = []
    for (label, category, description), count in sorted(exact_counter.items()):
        total = label_counter[label]
        exact_rows.append(
            {
                "label": label,
                "category": category,
                "description": description,
                "count": count,
                "percent_of_label": f"{count / total:.6f}" if total else "0",
                "example_json": examples[(label, category, description)],
            }
        )

    write_csv(
        output / "description_category_counts.csv",
        category_rows,
        ["label", "category", "count", "percent_of_label"],
    )
    write_csv(
        output / "description_category_pivot_counts.csv",
        pivot_rows,
        ["label", "total_shapes", "category_sum", "missing_from_sum"] + pivot_categories,
    )
    write_csv(
        output / "description_exact_counts.csv",
        exact_rows,
        ["label", "category", "description", "count", "percent_of_label", "example_json"],
    )
    write_csv(
        output / "description_details.csv",
        detail_rows,
        ["json", "image", "order_timestamp", "order_index", "shape_index", "label", "description", "category"],
    )

    empty_rows = [row for row in detail_rows if row["category"] == "EMPTY"]
    empty_counter = Counter(str(row["label"]) for row in empty_rows)
    empty_count_rows = [
        {
            "label": label,
            "empty_count": empty_counter[label],
            "total_count": label_counter[label],
            "percent_of_label": f"{empty_counter[label] / label_counter[label]:.6f}" if label_counter[label] else "0",
        }
        for label in sorted(empty_counter)
    ]
    write_csv(
        output / "empty_description_counts.csv",
        empty_count_rows,
        ["label", "empty_count", "total_count", "percent_of_label"],
    )
    write_csv(
        output / "empty_description_details.csv",
        empty_rows,
        ["json", "image", "order_timestamp", "order_index", "shape_index", "label", "description", "category"],
    )

    lines = [
        f"source: {source}",
        f"after_stem: {after_stem}",
        f"after_key: {sample_order_key(after_stem)}",
        f"exclude_indices: {','.join(str(index) for index in exclude_indices)}",
        f"selected_json_files: {len(paths)}",
        f"selected_shapes: {len(detail_rows)}",
        "",
        "| label | shapes | categories |",
        "| --- | ---: | ---: |",
    ]
    for label in sorted(label_counter):
        categories = sum(1 for item_label, _ in category_counter if item_label == label)
        lines.append(f"| {label} | {label_counter[label]} | {categories} |")
    lines.extend(
        [
            "",
            "Category pivot counts:",
            "",
            "| label | total_shapes | category_sum | missing_from_sum | "
            + " | ".join(pivot_categories)
            + " |",
            "| --- | ---: | ---: | ---: | "
            + " | ".join("---:" for _ in pivot_categories)
            + " |",
        ]
    )
    for row in pivot_rows:
        lines.append(
            f"| {row['label']} | {row['total_shapes']} | {row['category_sum']} | {row['missing_from_sum']} | "
            + " | ".join(str(row[category]) for category in pivot_categories)
            + " |"
        )
    lines.extend(
        [
            "",
            "| label | empty descriptions |",
            "| --- | ---: |",
        ]
    )
    for label in sorted(label_counter):
        lines.append(f"| {label} | {empty_counter[label]} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"selected json files: {len(paths)}")
    print(f"selected shapes: {len(detail_rows)}")
    print(f"output: {output}")
    for label in sorted(label_counter):
        print(f"{label}: {label_counter[label]}")


def main() -> None:
    args = parse_args()
    summarize(
        source=resolve_from_root(args.source, ROOT),
        after_stem=args.after_stem,
        exclude_indices=args.exclude_index,
        output=resolve_from_root(args.output, ROOT),
    )


if __name__ == "__main__":
    main()
