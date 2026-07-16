#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import load_dataset_config, resolve_from_root
from yolo11_obb.rhino_prediction import rboxes_to_yolo_lines


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _numpy(value: Any) -> np.ndarray:
    value = _field(value, "tensor", value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _sample_path(sample: Any) -> Path:
    metainfo = _field(sample, "metainfo", {})
    image_path = _field(metainfo, "img_path")
    if image_path is None:
        image_path = _field(sample, "img_path")
    if image_path is None:
        raise ValueError("RHINO prediction sample has no img_path")
    return Path(str(image_path))


def _prediction_samples(payload: Any) -> list[Any]:
    """Handle both a direct sample list and common MMEngine --out wrappers."""
    if isinstance(payload, Mapping):
        for key in ("predictions", "results"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        raise ValueError("RHINO prediction pickle must contain a list of prediction samples")
    return list(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RHINO/MMRotate pickle predictions to YOLO-OBB labels.")
    parser.add_argument("--data", type=Path, default=Path("datasets/obb_thin_thick/data.yaml"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-conf", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = resolve_from_root(args.data, ROOT)
    predictions_path = resolve_from_root(args.predictions, ROOT)
    output = resolve_from_root(args.output, ROOT)
    dataset = load_dataset_config(data_yaml)
    image_paths = {path.name: path for path in dataset.splits[args.split].iterdir() if path.is_file()}
    with predictions_path.open("rb") as handle:
        samples = _prediction_samples(pickle.load(handle))
    output.mkdir(parents=True, exist_ok=True)
    written = 0
    for sample in samples:
        image_name = _sample_path(sample).name
        image_path = image_paths.get(image_name)
        if image_path is None:
            raise ValueError(f"prediction image is outside dataset split {args.split}: {image_name}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"failed to read image: {image_path}")
        height, width = image.shape[:2]
        instances = _field(sample, "pred_instances")
        if instances is None:
            raise ValueError(f"RHINO prediction sample has no pred_instances: {image_name}")
        rboxes = _numpy(_field(instances, "bboxes"))
        labels = _numpy(_field(instances, "labels")).astype(int)
        scores = _numpy(_field(instances, "scores")).astype(float)
        keep = scores >= args.min_conf
        lines = rboxes_to_yolo_lines(rboxes[keep], labels[keep], scores[keep], width, height)
        (output / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    print(f"labels: {output}")
    print(f"images: {written}")


if __name__ == "__main__":
    main()
