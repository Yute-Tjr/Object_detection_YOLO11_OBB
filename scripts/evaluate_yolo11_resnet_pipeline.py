#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo11_obb.config import load_dataset_config, resolve_from_root, validate_dataset_layout
from yolo11_obb.pipeline_evaluate import (
    evaluate_pipeline_summary,
    load_truth_from_excel,
    write_pipeline_evaluation,
)
from yolo11_obb.pipeline_predict import run_pipeline


DEFAULT_DATA = ROOT / "datasets/obb_thin_thick/data.yaml"
DEFAULT_EXCEL = ROOT / "outputs/label1_6_description.xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11-OBB + ResNet18 label3/label5 pipeline on a YOLO dataset split."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--target-column", default="tag1")
    parser.add_argument("--det-weights", type=Path, required=True)
    parser.add_argument("--label3-weights", type=Path, required=True)
    parser.add_argument("--label5-weights", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-device", default=None)
    parser.add_argument("--cls-device", default=None)
    parser.add_argument("--cls-imgsz", type=int, default=224)
    parser.add_argument("--project", type=Path, default=Path("runs/pipeline_eval"))
    parser.add_argument("--name", default="yolo11_resnet_test_eval")
    parser.add_argument("--exist-ok", action="store_true")
    return parser.parse_args()


def _print_metric(name: str, values: dict[str, int | float]) -> None:
    print(
        f"{name}: total={values['total']} correct={values['correct']} "
        f"accuracy={values['accuracy']:.6f} macro_f1={values['macro_f1']:.6f} "
        f"unknown={values['unknown_predictions']} truth_missing={values['truth_missing']}"
    )


def main() -> None:
    args = parse_args()
    data = resolve_from_root(args.data, ROOT)
    excel = resolve_from_root(args.excel, ROOT)
    project = resolve_from_root(args.project, ROOT)
    output = project / args.name

    dataset = load_dataset_config(data)
    validate_dataset_layout(dataset)
    source = dataset.splits[args.split]

    report = run_pipeline(
        det_weights=resolve_from_root(args.det_weights, ROOT),
        source=source,
        label3_weights=resolve_from_root(args.label3_weights, ROOT),
        label5_weights=resolve_from_root(args.label5_weights, ROOT),
        output=output,
        imgsz=args.imgsz,
        det_conf=args.det_conf,
        det_device=args.det_device,
        cls_device=args.cls_device,
        cls_imgsz=args.cls_imgsz,
        exist_ok=args.exist_ok,
    )

    truth = load_truth_from_excel(excel, target_column=args.target_column)
    evaluation = evaluate_pipeline_summary(output / "summary.csv", truth)
    write_pipeline_evaluation(output, evaluation)

    print(f"source: {source}")
    print(f"output: {report['output']}")
    print(f"images: {report['images']}")
    print(f"detections: {report['detections']}")
    print(f"classified: {report['classified']}")
    for target in ["label3", "label5", "both_labels", "final_result"]:
        _print_metric(target, evaluation.metrics[target])
    print(f"metrics: {output / 'pipeline_eval_metrics.csv'}")
    print(f"predictions: {output / 'pipeline_eval_predictions.csv'}")


if __name__ == "__main__":
    main()
