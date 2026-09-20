import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from terminal_web.api.app import create_app
from terminal_web.database import Base
from terminal_web.domain import ImageStage, OverallResult
from terminal_web.inference.types import ImagePrediction, RegionPrediction
from terminal_web.readiness import ReadinessStore
from terminal_web.schemas import HealthResponse, ModelHealth
from terminal_web.storage import ArtifactStorage
from terminal_web.worker import InspectionWorker


def png_bytes(color: tuple[int, int, int]) -> bytes:
    handle = io.BytesIO()
    Image.new("RGB", (36, 72), color).save(handle, format="PNG")
    return handle.getvalue()


class ReadyForUploads:
    def snapshot(self):
        return HealthResponse(
            api_ready=True,
            database_ready=True,
            worker_ready=True,
            models_ready=True,
            models=[
                ModelHealth(
                    model_type="detector",
                    name="YOLO11l-OBB",
                    version="baseline",
                    ready=True,
                )
            ],
        )


class OneSuccessOneFailurePipeline:
    def __init__(self, storage_root: Path):
        self.calls = 0
        self.storage_root = storage_root

    def predict_image(self, image_path, artifacts, stage_callback):
        call = self.calls
        self.calls += 1
        if call == 1:
            raise RuntimeError(f"private failure at {self.storage_root}/secret/model.bin")
        for stage in (
            ImageStage.object_detection,
            ImageStage.anomaly_classification,
            ImageStage.rendering,
        ):
            stage_callback(stage)
        artifacts.result_path.parent.mkdir(parents=True, exist_ok=True)
        source = cv2.imread(str(image_path))
        cv2.imwrite(str(artifacts.result_path), source)
        stage_callback(ImageStage.complete)
        return ImagePrediction(
            regions=(
                RegionPrediction(
                    region_label="label2",
                    detection_confidence=0.99,
                    points=((1, 1), (20, 1), (20, 30), (1, 30)),
                    selected_for_classification=False,
                    anomaly=None,
                    color=None,
                    crop_path=None,
                    error=None,
                ),
            ),
            overall_result=OverallResult.ok,
            warnings=(),
        )


class EndToEndWorkflowTest(unittest.TestCase):
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
        self.sessions = sessionmaker(bind=self.engine, expire_on_commit=False)
        app = create_app(
            settings=SimpleNamespace(max_images_per_task=100),
            session_factory=self.sessions,
            storage=self.storage,
            readiness=ReadyForUploads(),
        )
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.temp.cleanup()

    def test_two_image_api_worker_flow_persists_partial_failure_and_artifacts(self):
        response = self.client.post(
            "/api/v1/tasks",
            files=[
                ("files", ("ok.png", png_bytes((20, 120, 20)), "image/png")),
                ("files", ("failed.png", png_bytes((120, 20, 20)), "image/png")),
            ],
            data={"operator": "测试员", "name": "端到端测试"},
        )
        self.assertEqual(response.status_code, 202, response.text)
        task_id = response.json()["id"]

        worker = InspectionWorker(
            session_factory=self.sessions,
            storage=self.storage,
            pipeline_loader=lambda: OneSuccessOneFailurePipeline(self.storage.root),
            readiness=ReadinessStore(self.root / "worker-readiness.json", timeout_seconds=120),
            worker_id="e2e-worker",
        )
        self.assertTrue(worker.run_once())

        task_response = self.client.get(f"/api/v1/tasks/{task_id}")
        self.assertEqual(task_response.status_code, 200, task_response.text)
        task = task_response.json()
        self.assertEqual(task["status"], "partial_failed")
        self.assertEqual(task["completedImages"], 2)
        self.assertEqual(task["succeededImages"], 1)
        self.assertEqual(task["failedImages"], 1)

        succeeded = next(image for image in task["images"] if image["status"] == "succeeded")
        failed = next(image for image in task["images"] if image["status"] == "failed")
        self.assertEqual(self.client.get(succeeded["originalUrl"]).status_code, 200)
        self.assertEqual(self.client.get(succeeded["resultUrl"]).status_code, 200)
        self.assertIn("<storage>", failed["errorMessage"])
        self.assertNotIn(str(self.storage.root), failed["errorMessage"])
