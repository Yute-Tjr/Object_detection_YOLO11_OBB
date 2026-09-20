#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_web.domain import ImageStage
from terminal_web.inference.pipeline import InferenceArtifacts, LoadedInferencePipeline
from terminal_web.readiness import sha256_file


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the three terminal inspection checkpoints"
    )
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--label3", type=Path, required=True)
    parser.add_argument("--label5", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-device", default="cpu")
    parser.add_argument("--cls-device", default="cpu")
    return parser.parse_args(argv)


def validate_inputs(args: argparse.Namespace) -> None:
    for name, path in (
        ("detector", args.detector),
        ("label3", args.label3),
        ("label5", args.label5),
    ):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"missing {name} weight: {path}")
    if not args.image.expanduser().is_file():
        raise FileNotFoundError(f"missing input image: {args.image}")


def _classification_json(prediction):
    if prediction is None:
        return None
    return {
        "label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        validate_inputs(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    detector = args.detector.expanduser().resolve()
    label3 = args.label3.expanduser().resolve()
    label5 = args.label5.expanduser().resolve()
    image = args.image.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    load_started = time.perf_counter()
    pipeline = LoadedInferencePipeline(
        detector_weights=detector,
        label3_weights=label3,
        label5_weights=label5,
        detection_device=args.det_device,
        classification_device=args.cls_device,
    )
    load_seconds = time.perf_counter() - load_started

    stage_times: dict[ImageStage, float] = {}

    def record_stage(stage: ImageStage) -> None:
        stage_times[stage] = time.perf_counter()

    result_path = output / "result.jpg"
    prediction = pipeline.predict_image(
        image,
        InferenceArtifacts(crop_dir=output / "crops", result_path=result_path),
        record_stage,
    )
    detection_seconds = (
        stage_times[ImageStage.anomaly_classification]
        - stage_times[ImageStage.object_detection]
    )
    classification_seconds = (
        stage_times[ImageStage.rendering]
        - stage_times[ImageStage.anomaly_classification]
    )
    render_seconds = (
        stage_times[ImageStage.complete] - stage_times[ImageStage.rendering]
    )
    timings = {
        "modelLoadSeconds": load_seconds,
        "detectionSeconds": detection_seconds,
        "classificationSeconds": classification_seconds,
        "renderSeconds": render_seconds,
    }
    payload = {
        "image": image.name,
        "resultImage": result_path.name,
        "overallResult": prediction.overall_result.value,
        "warnings": list(prediction.warnings),
        "models": {
            "detector": {"name": "YOLO11l-OBB", "sha256": sha256_file(detector)},
            "label3": {"name": "ResNet18-label3", "sha256": sha256_file(label3)},
            "label5": {"name": "ResNet18-label5", "sha256": sha256_file(label5)},
        },
        "timings": timings,
        "detections": [
            {
                "regionLabel": region.region_label,
                "detectionConfidence": region.detection_confidence,
                "points": [list(point) for point in region.points],
                "selectedForClassification": region.selected_for_classification,
                "anomaly": _classification_json(region.anomaly),
                "color": _classification_json(region.color),
                "error": region.error,
            }
            for region in prediction.regions
        ],
    }
    (output / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for name, seconds in timings.items():
        print(f"{name}: {seconds:.3f}")
    print(f"result: {output / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
