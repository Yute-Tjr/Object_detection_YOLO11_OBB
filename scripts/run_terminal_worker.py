#!/usr/bin/env python3
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from terminal_web.config import get_settings
from terminal_web.database import build_session_factory
from terminal_web.inference.pipeline import LoadedInferencePipeline
from terminal_web.readiness import ModelSpec, ReadinessStore
from terminal_web.storage import ArtifactStorage
from terminal_web.worker import InspectionWorker


def main() -> None:
    settings = get_settings()
    _, session_factory = build_session_factory(settings.database_url)
    storage = ArtifactStorage(settings.storage_root)
    readiness = ReadinessStore(
        settings.storage_root / "worker-readiness.json",
        timeout_seconds=settings.worker_heartbeat_timeout_seconds,
    )

    def load_pipeline() -> LoadedInferencePipeline:
        return LoadedInferencePipeline(
            detector_weights=settings.detector_weights,
            label3_weights=settings.label3_classifier_weights,
            label5_weights=settings.label5_classifier_weights,
            detection_imgsz=settings.detection_imgsz,
            detection_confidence=settings.detection_confidence,
            detection_device=settings.detection_device,
            classification_imgsz=settings.classification_imgsz,
            classification_device=settings.classification_device,
        )

    specs = (
        ModelSpec(
            "detector",
            "YOLO11l-OBB",
            "baseline",
            settings.detector_weights,
            {
                "imgsz": settings.detection_imgsz,
                "confidence": settings.detection_confidence,
            },
        ),
        ModelSpec(
            "anomaly",
            "ResNet18-label3",
            "best",
            settings.label3_classifier_weights,
            {"imgsz": settings.classification_imgsz, "region": "label3"},
        ),
        ModelSpec(
            "anomaly",
            "ResNet18-label5",
            "best",
            settings.label5_classifier_weights,
            {"imgsz": settings.classification_imgsz, "region": "label5"},
        ),
    )
    worker = InspectionWorker(
        session_factory=session_factory,
        storage=storage,
        pipeline_loader=load_pipeline,
        readiness=readiness,
        worker_id=f"{socket.gethostname()}-{os.getpid()}",
        heartbeat_timeout_seconds=settings.worker_heartbeat_timeout_seconds,
        max_attempts=settings.task_max_attempts,
        model_specs=specs,
    )
    worker.run_forever()


if __name__ == "__main__":
    main()
