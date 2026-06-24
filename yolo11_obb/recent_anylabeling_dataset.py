from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .anylabeling_to_obb import (
    ConversionReport,
    annotation_to_obb_lines,
    load_annotation,
    resolve_image_path,
)
from .label1_thin_thick_dataset import create_label1_thin_thick_dataset


OrderKey = Tuple[str, int]


@dataclass(frozen=True)
class RecentLabel1ThinThickReport:
    selected_json_files: int
    converted: ConversionReport
    dataset: Dict[str, Dict[str, object]]


def sample_order_key(stem: str) -> OrderKey:
    clean_stem = Path(stem).stem
    timestamp_match = re.search(r"(20\d{15})", clean_stem)
    index_match = re.search(r"-(\d+)$", clean_stem)
    if not timestamp_match:
        raise ValueError(f"cannot parse sample order key from stem: {stem}")
    index = int(index_match.group(1)) if index_match else 999
    return timestamp_match.group(1), index


def _selected_json_paths(source: Path, after_stem: str) -> List[Path]:
    after_key = sample_order_key(after_stem)
    json_paths = sorted(
        source.glob("*.json"),
        key=lambda path: (sample_order_key(path.stem), path.stem),
    )
    selected = [path for path in json_paths if sample_order_key(path.stem) > after_key]
    if not selected:
        raise ValueError(f"no JSON annotations found after marker: {after_stem}")
    return selected


def _convert_selected_anylabeling(
    source: Path,
    output: Path,
    json_paths: List[Path],
    clip: bool,
) -> ConversionReport:
    if output.exists():
        raise FileExistsError(f"converted output already exists: {output}")
    output.mkdir(parents=True)

    objects = 0
    images_copied = 0
    manifest_rows = []

    for json_path in json_paths:
        annotation = load_annotation(json_path)
        lines = annotation_to_obb_lines(annotation, clip=clip)
        (output / f"{json_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        objects += len(lines)

        image_source = resolve_image_path(source, json_path, annotation)
        shutil.copy2(image_source, output / image_source.name)
        images_copied += 1
        manifest_rows.append(
            {
                "json": json_path.name,
                "image": image_source.name,
                "order_timestamp": sample_order_key(json_path.stem)[0],
                "order_index": sample_order_key(json_path.stem)[1],
            }
        )

    with (output / "selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["json", "image", "order_timestamp", "order_index"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return ConversionReport(
        json_files=len(json_paths),
        labels_written=len(json_paths),
        objects=objects,
        images_copied=images_copied,
    )


def create_recent_label1_thin_thick_dataset(
    source: Union[str, Path],
    converted_output: Union[str, Path],
    dataset_output: Union[str, Path],
    after_stem: str,
    top_edge_threshold_px: float,
    train_ratio: float = 0.8,
    seed: int = 42,
    clip: bool = True,
) -> RecentLabel1ThinThickReport:
    source = Path(source).expanduser().resolve()
    converted_output = Path(converted_output).expanduser().resolve()
    dataset_output = Path(dataset_output).expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source}")

    selected_json_paths = _selected_json_paths(source, after_stem)
    conversion_report = _convert_selected_anylabeling(
        source=source,
        output=converted_output,
        json_paths=selected_json_paths,
        clip=clip,
    )
    dataset_report = create_label1_thin_thick_dataset(
        source=converted_output,
        output=dataset_output,
        top_edge_threshold_px=top_edge_threshold_px,
        train_ratio=train_ratio,
        seed=seed,
    )

    lines = [
        f"source: {source}",
        f"converted_output: {converted_output}",
        f"dataset_output: {dataset_output}",
        f"after_stem: {after_stem}",
        f"after_key: {sample_order_key(after_stem)}",
        f"selected_json_files: {len(selected_json_paths)}",
        f"top_edge_threshold_px: {top_edge_threshold_px}",
        f"train_ratio: {train_ratio}",
        f"seed: {seed}",
    ]
    (dataset_output / "source_filter_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return RecentLabel1ThinThickReport(
        selected_json_files=len(selected_json_paths),
        converted=conversion_report,
        dataset=dataset_report,
    )
