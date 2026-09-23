import io
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from terminal_web.api.app import create_app
from terminal_web.database import Base
from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus
from terminal_web.models import (
    ClassificationResult,
    Detection,
    InspectionImage,
    InspectionTask,
    ModelRecord,
)
from terminal_web.schemas import HealthResponse
from terminal_web.storage import ArtifactStorage
from tests.web.auth_helpers import create_and_login


def png_bytes(color=(10, 20, 30)) -> bytes:
    handle = io.BytesIO()
    Image.new("RGB", (20, 30), color).save(handle, format="PNG")
    return handle.getvalue()


class FakeReadiness:
    def __init__(self):
        self.api_ready = True
        self.database_ready = True
        self.worker_ready = True
        self.models_ready = True

    def snapshot(self) -> HealthResponse:
        return HealthResponse(
            api_ready=self.api_ready,
            database_ready=self.database_ready,
            worker_ready=self.worker_ready,
            models_ready=self.models_ready,
        )


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage_root = Path(self.temp.name) / "storage"
        self.storage = ArtifactStorage(self.storage_root)
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.readiness = FakeReadiness()
        settings = SimpleNamespace(
            max_images_per_task=100,
            session_ttl_hours=12,
            session_cookie_secure=False,
        )
        app = create_app(
            settings=settings,
            session_factory=self.session_factory,
            storage=self.storage,
            readiness=self.readiness,
        )
        self.client = TestClient(app)
        create_and_login(self.client, self.session_factory)

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.temp.cleanup()

    @staticmethod
    def valid_files(count: int):
        return [
            ("files", (f"terminal-{index}.png", png_bytes(), "image/png"))
            for index in range(count)
        ]

    def count_tasks(self) -> int:
        with self.session_factory() as session:
            return session.scalar(select(func.count()).select_from(InspectionTask)) or 0

    def test_create_task_accepts_one_valid_image(self):
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
        )
        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        self.assertEqual(payload["totalImages"], 1)
        self.assertTrue(
            {"operator", "name", "note", "detectorModel"}.isdisjoint(payload)
        )
        self.assertEqual(self.count_tasks(), 1)

    def test_create_task_ignores_legacy_metadata_form_fields(self):
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
            data={"operator": " 张三 ", "name": "早班", "note": "首件"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        self.assertTrue({"operator", "name", "note"}.isdisjoint(payload))

    def test_create_task_rejects_101_images_without_writing_files(self):
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(101),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.count_tasks(), 0)
        self.assertEqual(list(self.storage_root.iterdir()), [])

    def test_create_task_rejects_when_models_not_ready_without_creating_task(self):
        self.readiness.models_ready = False
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.count_tasks(), 0)

    def test_create_task_with_one_corrupt_file_is_rejected_atomically(self):
        files = self.valid_files(1) + [
            ("files", ("bad.png", b"broken", "image/png"))
        ]
        response = self.client.post("/api/v1/tasks", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.count_tasks(), 0)
        self.assertEqual(list(self.storage_root.iterdir()), [])

    def test_failed_filter_includes_partial_failed(self):
        with self.session_factory() as session:
            for status in (
                TaskStatus.failed,
                TaskStatus.partial_failed,
                TaskStatus.succeeded,
            ):
                session.add(
                    InspectionTask(
                        id=uuid.uuid4(),
                        display_id=f"T-{uuid.uuid4().hex[:8]}",
                        status=status,
                    )
                )
            session.commit()

        response = self.client.get("/api/v1/tasks?status=failed")
        self.assertEqual(response.status_code, 200, response.text)
        statuses = {item["status"] for item in response.json()["items"]}
        self.assertEqual(statuses, {"failed", "partial_failed"})

    def test_task_search_accepts_original_image_filename(self):
        created = self.client.post(
            "/api/v1/tasks",
            files=[("files", ("unique-terminal-name.png", png_bytes(), "image/png"))],
        )
        self.assertEqual(created.status_code, 202, created.text)

        response = self.client.get("/api/v1/tasks?query=unique-terminal-name")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(response.json()["items"][0]["id"], created.json()["id"])

    def test_health_response_does_not_expose_model_identity(self):
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(
            set(response.json()),
            {"apiReady", "databaseReady", "workerReady", "modelsReady"},
        )

    def test_image_detail_does_not_expose_classifier_identity(self):
        image_id = uuid.uuid4()
        with self.session_factory() as session:
            model = ModelRecord(
                model_type="anomaly",
                name="private-classifier-name",
                version="private-version",
                weights_path="private.pt",
                sha256="a" * 64,
                load_config={},
            )
            task = InspectionTask(display_id="T-classification", total_images=1)
            image = InspectionImage(
                id=image_id,
                sequence_no=0,
                original_filename="terminal.png",
                stored_filename="terminal.png",
                original_path="tasks/terminal.png",
                width=20,
                height=30,
                size_bytes=100,
            )
            detection = Detection(
                region_label="label3",
                confidence=0.99,
                points=[[1, 1], [10, 1], [10, 10], [1, 10]],
                selected_for_classification=True,
            )
            detection.classifications.append(
                ClassificationResult(
                    classifier_type="anomaly",
                    predicted_label="OK",
                    confidence=0.98,
                    model=model,
                )
            )
            image.detections.append(detection)
            task.images.append(image)
            session.add(task)
            session.commit()

        response = self.client.get(f"/api/v1/images/{image_id}")

        self.assertEqual(response.status_code, 200, response.text)
        anomaly = response.json()["detections"][0]["anomaly"]
        self.assertTrue({"modelName", "modelVersion"}.isdisjoint(anomaly))

    def test_delete_completed_task_removes_database_record_and_artifacts(self):
        created = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
        )
        self.assertEqual(created.status_code, 202, created.text)
        task_id = created.json()["id"]
        task_dir = self.storage_root / "tasks" / task_id
        with self.session_factory() as session:
            task = session.get(InspectionTask, uuid.UUID(task_id))
            task.status = TaskStatus.succeeded
            task.current_stage = ImageStage.complete
            session.commit()

        response = self.client.delete(f"/api/v1/tasks/{task_id}")

        self.assertEqual(response.status_code, 204, response.text)
        self.assertEqual(self.count_tasks(), 0)
        self.assertFalse(task_dir.exists())

    def test_delete_running_task_is_rejected_without_removing_data(self):
        created = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
        )
        self.assertEqual(created.status_code, 202, created.text)
        task_id = created.json()["id"]
        task_dir = self.storage_root / "tasks" / task_id

        response = self.client.delete(f"/api/v1/tasks/{task_id}")

        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(self.count_tasks(), 1)
        self.assertTrue(task_dir.exists())

    def test_retry_only_accepts_failed_image(self):
        image_id = uuid.uuid4()
        with self.session_factory() as session:
            task = InspectionTask(
                id=uuid.uuid4(),
                display_id="T-retry",
                status=TaskStatus.succeeded,
                total_images=1,
                completed_images=1,
                succeeded_images=1,
            )
            task.images.append(
                InspectionImage(
                    id=image_id,
                    sequence_no=0,
                    original_filename="done.png",
                    stored_filename="done.png",
                    original_path="tasks/done.png",
                    status=ImageStatus.succeeded,
                    stage=ImageStage.complete,
                    overall_result=OverallResult.ok,
                    width=20,
                    height=30,
                    size_bytes=100,
                )
            )
            session.add(task)
            session.commit()

        response = self.client.post(f"/api/v1/images/{image_id}/retry")
        self.assertEqual(response.status_code, 409)

    def test_repeated_reads_return_queue_connections(self):
        engine = create_engine(
            f"sqlite+pysqlite:///{Path(self.temp.name) / 'pool-test.sqlite'}",
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=2,
            max_overflow=0,
        )
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine, expire_on_commit=False)
        app = create_app(
            settings=SimpleNamespace(
                max_images_per_task=100,
                session_ttl_hours=12,
                session_cookie_secure=False,
            ),
            session_factory=session_factory,
            storage=self.storage,
            readiness=self.readiness,
        )

        try:
            with TestClient(app) as client:
                create_and_login(client, session_factory, username="pool-tester")
                for _ in range(30):
                    response = client.get("/api/v1/tasks?status=all")
                    self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(engine.pool.checkedout(), 0)
        finally:
            engine.dispose()
