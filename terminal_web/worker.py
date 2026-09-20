from __future__ import annotations

import signal
import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from terminal_web.domain import ImageStage, ImageStatus, TaskStatus
from terminal_web.inference.pipeline import InferenceArtifacts
from terminal_web.models import (
    ClassificationResult,
    Detection,
    InspectionImage,
    InspectionTask,
    ModelRecord,
)
from terminal_web.queue import claim_next_task, recover_stale_tasks
from terminal_web.readiness import (
    ModelSpec,
    ReadinessStore,
    register_models,
    validate_pipeline_models,
)
from terminal_web.repositories import TaskRepository
from terminal_web.storage import ArtifactStorage


class InspectionWorker:
    def __init__(
        self,
        *,
        session_factory,
        storage: ArtifactStorage,
        pipeline_loader: Callable[[], object],
        readiness: ReadinessStore,
        worker_id: str,
        clock: Callable[[], datetime] | None = None,
        heartbeat_timeout_seconds: int = 120,
        max_attempts: int = 2,
        model_specs: Sequence[ModelSpec] = (),
        idle_sleep_seconds: float = 1.0,
    ):
        self.session_factory = session_factory
        self.storage = storage
        self.pipeline_loader = pipeline_loader
        self.readiness = readiness
        self.worker_id = worker_id
        self.clock = clock or (lambda: datetime.now(UTC))
        self.heartbeat_timeout_seconds = heartbeat_timeout_seconds
        self.max_attempts = max_attempts
        self.model_specs = tuple(model_specs)
        self.idle_sleep_seconds = idle_sleep_seconds
        self.pipeline = None
        self.model_ids: dict[tuple[str, str], uuid.UUID] = {}
        self._stop = threading.Event()

    def _load_pipeline(self) -> bool:
        if self.pipeline is not None:
            return True
        try:
            pipeline = self.pipeline_loader()
            validate_pipeline_models(pipeline)
            with self.session_factory() as session:
                models = register_models(session, self.model_specs)
                session.commit()
                if self.model_specs:
                    records = list(session.scalars(select(ModelRecord)))
                    self.model_ids = {
                        (record.model_type, record.name): record.id for record in records
                    }
            self.pipeline = pipeline
            self.readiness.mark_ready(models, self.worker_id, self.clock())
            return True
        except Exception as exc:
            self.pipeline = None
            self.readiness.mark_failed(
                f"model loading failed: {exc.__class__.__name__}"
            )
            return False

    def run_once(self) -> bool:
        if not self._load_pipeline():
            return False

        now = self.clock()
        with self.session_factory() as session:
            recover_stale_tasks(
                session,
                now,
                timeout_seconds=self.heartbeat_timeout_seconds,
                max_attempts=self.max_attempts,
            )
            session.commit()
            task = claim_next_task(session, self.worker_id, now)
            if task is None:
                self.readiness.heartbeat(self.worker_id, now)
                session.commit()
                return False
            detector_id = self.model_ids.get(("detector", "YOLO11l-OBB"))
            if detector_id is not None:
                task.detector_model_id = detector_id
            task_id = task.id
            session.commit()

            task = session.get(InspectionTask, task_id)
            queued_images = sorted(
                (
                    image
                    for image in task.images
                    if image.status == ImageStatus.queued.value
                ),
                key=lambda image: image.sequence_no,
            )
            for image in queued_images:
                self._process_image(session, task, image)

            TaskRepository(session).recompute_progress(task.id)
            session.commit()
        return True

    def _process_image(self, session, task: InspectionTask, image: InspectionImage) -> None:
        image.status = ImageStatus.running
        image.stage = ImageStage.object_detection
        image.started_at = self.clock()
        image.error_code = None
        image.error_message = None
        task.current_stage = ImageStage.object_detection
        task.heartbeat_at = self.clock()
        self.readiness.heartbeat(self.worker_id, task.heartbeat_at)
        session.commit()

        def stage_callback(stage: ImageStage) -> None:
            image.stage = stage
            task.current_stage = stage
            task.heartbeat_at = self.clock()
            self.readiness.heartbeat(self.worker_id, task.heartbeat_at)
            session.commit()

        try:
            original_path = self.storage.resolve(image.original_path)
            result_path = self.storage.result_path(task.id, image.id)
            crop_dir = self.storage.crop_path(
                task.id, image.id, "region-placeholder.png"
            ).parent
            prediction = self.pipeline.predict_image(
                original_path,
                InferenceArtifacts(crop_dir=crop_dir, result_path=result_path),
                stage_callback,
            )
            if not result_path.is_file():
                raise RuntimeError("inference did not produce a result image")

            image.detections.clear()
            for region in prediction.regions:
                detection = Detection(
                    region_label=region.region_label,
                    confidence=region.detection_confidence,
                    points=[list(point) for point in region.points],
                    selected_for_classification=region.selected_for_classification,
                    crop_path=(
                        self.storage.relative(Path(region.crop_path))
                        if region.crop_path
                        else None
                    ),
                )
                if region.anomaly is not None:
                    detection.classifications.append(
                        ClassificationResult(
                            classifier_type="anomaly",
                            predicted_label=region.anomaly.label,
                            confidence=region.anomaly.confidence,
                            probabilities=dict(region.anomaly.probabilities),
                            model_id=self.model_ids.get(
                                ("anomaly", f"ResNet18-{region.region_label}")
                            ),
                        )
                    )
                if region.color is not None:
                    detection.classifications.append(
                        ClassificationResult(
                            classifier_type="color",
                            predicted_label=region.color.label,
                            confidence=region.color.confidence,
                            probabilities=dict(region.color.probabilities),
                        )
                    )
                image.detections.append(detection)

            image.result_path = self.storage.relative(result_path)
            image.overall_result = prediction.overall_result
            image.status = ImageStatus.succeeded
            image.stage = ImageStage.complete
            image.finished_at = self.clock()
            task.heartbeat_at = self.clock()
            TaskRepository(session).recompute_progress(task.id)
            session.commit()
        except Exception as exc:
            session.rollback()
            task = session.get(InspectionTask, task.id)
            image = session.get(InspectionImage, image.id)
            image.status = ImageStatus.failed
            image.stage = ImageStage.complete
            image.error_code = exc.__class__.__name__
            image.error_message = str(exc).replace(
                str(self.storage.root), "<storage>"
            )[:500]
            image.finished_at = self.clock()
            task.heartbeat_at = self.clock()
            self.readiness.heartbeat(self.worker_id, task.heartbeat_at)
            TaskRepository(session).recompute_progress(task.id)
            session.commit()

    def run_forever(self) -> None:
        def request_stop(signum, frame):
            self._stop.set()

        try:
            signal.signal(signal.SIGTERM, request_stop)
            signal.signal(signal.SIGINT, request_stop)
        except ValueError:
            pass
        while not self._stop.is_set():
            worked = self.run_once()
            if not worked:
                self._stop.wait(self.idle_sleep_seconds)

    def stop(self) -> None:
        self._stop.set()
