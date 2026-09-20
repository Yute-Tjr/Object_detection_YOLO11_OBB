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
from terminal_web.models import InspectionImage, InspectionTask
from terminal_web.schemas import HealthResponse, ModelHealth
from terminal_web.storage import ArtifactStorage


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
            models=[
                ModelHealth(
                    model_type="detector",
                    name="YOLO11l-OBB",
                    version="baseline",
                    ready=self.models_ready,
                )
            ],
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
        settings = SimpleNamespace(max_images_per_task=100)
        app = create_app(
            settings=settings,
            session_factory=self.session_factory,
            storage=self.storage,
            readiness=self.readiness,
        )
        self.client = TestClient(app)

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
            data={"operator": "张三"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["totalImages"], 1)
        self.assertEqual(response.json()["operator"], "张三")
        self.assertEqual(self.count_tasks(), 1)

    def test_create_task_requires_non_blank_operator(self):
        for data in ({}, {"operator": "   "}):
            response = self.client.post(
                "/api/v1/tasks", files=self.valid_files(1), data=data
            )
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.count_tasks(), 0)

    def test_create_task_persists_operator_name_and_note(self):
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
            data={"operator": " 张三 ", "name": "早班", "note": "首件"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        payload = response.json()
        self.assertEqual(payload["operator"], "张三")
        self.assertEqual(payload["name"], "早班")
        self.assertEqual(payload["note"], "首件")

    def test_legacy_task_without_operator_is_readable(self):
        with self.session_factory() as session:
            task = InspectionTask(display_id="T-legacy", operator=None)
            session.add(task)
            session.commit()
            task_id = task.id

        response = self.client.get(f"/api/v1/tasks/{task_id}")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["operator"])

    def test_create_task_rejects_101_images_without_writing_files(self):
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(101),
            data={"operator": "张三"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.count_tasks(), 0)
        self.assertEqual(list(self.storage_root.iterdir()), [])

    def test_create_task_rejects_when_models_not_ready_without_creating_task(self):
        self.readiness.models_ready = False
        response = self.client.post(
            "/api/v1/tasks",
            files=self.valid_files(1),
            data={"operator": "张三"},
        )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.count_tasks(), 0)

    def test_create_task_with_one_corrupt_file_is_rejected_atomically(self):
        files = self.valid_files(1) + [
            ("files", ("bad.png", b"broken", "image/png"))
        ]
        response = self.client.post(
            "/api/v1/tasks", files=files, data={"operator": "张三"}
        )
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
            settings=SimpleNamespace(max_images_per_task=100),
            session_factory=session_factory,
            storage=self.storage,
            readiness=self.readiness,
        )

        try:
            with TestClient(app) as client:
                for _ in range(30):
                    response = client.get("/api/v1/tasks?status=all")
                    self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(engine.pool.checkedout(), 0)
        finally:
            engine.dispose()
