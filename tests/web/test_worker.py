import io
import tempfile
import unittest
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from terminal_web.database import Base
from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus
from terminal_web.inference.types import ImagePrediction, RegionPrediction
from terminal_web.models import InspectionImage, InspectionTask
from terminal_web.readiness import ReadinessStore
from terminal_web.storage import ArtifactStorage, InvalidImageError
from terminal_web.worker import InspectionWorker


def png_bytes() -> bytes:
    handle = io.BytesIO()
    Image.new("RGB", (40, 80), (230, 230, 230)).save(handle, format="PNG")
    return handle.getvalue()


class MutableClock:
    def __init__(self):
        self.value = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)

    def __call__(self):
        return self.value

    def advance(self, seconds: int):
        self.value += timedelta(seconds=seconds)


class FakePipeline:
    def __init__(self, *, fail_calls=None, clock=None):
        self.fail_calls = set(fail_calls or [])
        self.clock = clock
        self.attempted = []
        self.stages = []

    def predict_image(self, image_path, artifacts, stage_callback):
        call_index = len(self.attempted)
        self.attempted.append(Path(image_path).name)
        if call_index in self.fail_calls:
            raise InvalidImageError("corrupt image")
        for stage in (
            ImageStage.object_detection,
            ImageStage.anomaly_classification,
            ImageStage.rendering,
        ):
            if self.clock:
                self.clock.advance(5)
            stage_callback(stage)
            self.stages.append(stage)
        artifacts.result_path.parent.mkdir(parents=True, exist_ok=True)
        source = cv2.imread(str(image_path))
        cv2.imwrite(str(artifacts.result_path), source)
        if self.clock:
            self.clock.advance(5)
        stage_callback(ImageStage.complete)
        self.stages.append(ImageStage.complete)
        return ImagePrediction(
            regions=(
                RegionPrediction(
                    region_label="label2",
                    detection_confidence=0.9,
                    points=((1, 1), (20, 1), (20, 20), (1, 20)),
                    selected_for_classification=False,
                    anomaly=None,
                    color=None,
                    crop_path=None,
                    error=None,
                ),
            ),
            overall_result=OverallResult.unknown,
            warnings=(),
        )


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.storage = ArtifactStorage(self.root / "storage")
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.clock = MutableClock()
        self.readiness = ReadinessStore(
            self.root / "readiness.json", timeout_seconds=120, clock=self.clock
        )

    def tearDown(self):
        self.engine.dispose()
        self.temp.cleanup()

    def make_task(self, image_count: int) -> uuid.UUID:
        task_id = uuid.uuid4()
        task = InspectionTask(
            id=task_id,
            display_id=f"T-{task_id.hex[:8]}",
            status=TaskStatus.queued,
            total_images=image_count,
        )
        for index in range(image_count):
            image_id = uuid.uuid4()
            stored = self.storage.save_upload(
                task_id, image_id, f"image-{index}.png", png_bytes()
            )
            task.images.append(
                InspectionImage(
                    id=image_id,
                    sequence_no=index,
                    original_filename=stored.original_filename,
                    stored_filename=stored.stored_filename,
                    original_path=stored.relative_path,
                    width=stored.width,
                    height=stored.height,
                    size_bytes=stored.size_bytes,
                )
            )
        with self.session_factory() as session:
            session.add(task)
            session.commit()
        return task_id

    def worker(self, pipeline):
        return InspectionWorker(
            session_factory=self.session_factory,
            storage=self.storage,
            pipeline_loader=lambda: pipeline,
            readiness=self.readiness,
            worker_id="test-worker",
            clock=self.clock,
            heartbeat_timeout_seconds=120,
            max_attempts=2,
        )

    def test_worker_updates_stage_and_progress_for_each_image(self):
        task_id = self.make_task(1)
        pipeline = FakePipeline()

        worked = self.worker(pipeline).run_once()

        self.assertTrue(worked)
        self.assertEqual(
            pipeline.stages,
            [
                ImageStage.object_detection,
                ImageStage.anomaly_classification,
                ImageStage.rendering,
                ImageStage.complete,
            ],
        )
        with self.session_factory() as session:
            task = session.get(InspectionTask, task_id)
            self.assertEqual(task.completed_images, 1)
            self.assertEqual(task.status, TaskStatus.succeeded)

    def test_one_corrupt_image_does_not_abort_remaining_images(self):
        task_id = self.make_task(2)
        pipeline = FakePipeline(fail_calls={0})

        self.worker(pipeline).run_once()

        self.assertEqual(len(pipeline.attempted), 2)
        with self.session_factory() as session:
            task = session.get(InspectionTask, task_id)
            self.assertEqual([image.status for image in task.images], ["failed", "succeeded"])

    def test_mixed_outcomes_finish_partial_failed(self):
        task_id = self.make_task(2)
        self.worker(FakePipeline(fail_calls={0})).run_once()
        with self.session_factory() as session:
            task = session.get(InspectionTask, task_id)
            self.assertEqual(task.succeeded_images, 1)
            self.assertEqual(task.failed_images, 1)
            self.assertEqual(task.status, TaskStatus.partial_failed)

    def test_model_load_failure_marks_readiness_false_and_claims_nothing(self):
        task_id = self.make_task(1)
        worker = InspectionWorker(
            session_factory=self.session_factory,
            storage=self.storage,
            pipeline_loader=lambda: (_ for _ in ()).throw(RuntimeError("bad weights")),
            readiness=self.readiness,
            worker_id="test-worker",
            clock=self.clock,
        )

        self.assertFalse(worker.run_once())

        self.assertFalse(self.readiness.snapshot().models_ready)
        with self.session_factory() as session:
            task = session.get(InspectionTask, task_id)
            self.assertEqual(task.status, TaskStatus.queued)
            self.assertEqual(task.attempt_count, 0)

    def test_worker_heartbeat_is_refreshed_while_processing(self):
        task_id = self.make_task(1)
        started = self.clock()
        self.worker(FakePipeline(clock=self.clock)).run_once()
        with self.session_factory() as session:
            task = session.get(InspectionTask, task_id)
            self.assertGreater(task.heartbeat_at.replace(tzinfo=UTC), started)
