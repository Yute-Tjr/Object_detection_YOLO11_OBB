import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from terminal_web.api.app import create_app
from terminal_web.database import Base
from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus
from terminal_web.feedback import logical_region
from terminal_web.models import (
    ClassificationResult,
    Detection,
    ImageFeedback,
    ImageFeedbackItem,
    InspectionImage,
    InspectionTask,
)
from terminal_web.schemas import HealthResponse
from terminal_web.storage import ArtifactStorage
from tests.web.auth_helpers import create_and_login


class ReadyServices:
    def snapshot(self):
        return HealthResponse(
            api_ready=True,
            database_ready=True,
            worker_ready=True,
            models_ready=True,
        )


class FeedbackApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.app = create_app(
            settings=SimpleNamespace(
                max_images_per_task=100,
                session_ttl_hours=12,
                session_cookie_secure=False,
            ),
            session_factory=self.session_factory,
            storage=ArtifactStorage(Path(self.temp.name) / "storage"),
            readiness=ReadyServices(),
        )
        self.client = TestClient(self.app)
        create_and_login(self.client, self.session_factory, username="reviewer")
        self.image_id, self.detection_ids = self.seed_image(
            "T-feedback",
            ("label1_thin", "label3", "label5"),
        )

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.temp.cleanup()

    def seed_image(
        self,
        display_id,
        labels,
        *,
        image_status=ImageStatus.succeeded,
        image_stage=ImageStage.complete,
    ):
        image_id = uuid.uuid4()
        detections = []
        with self.session_factory() as session:
            task = InspectionTask(
                display_id=display_id,
                status=TaskStatus.succeeded,
                current_stage=ImageStage.complete,
                total_images=1,
                completed_images=1,
                succeeded_images=1,
            )
            image = InspectionImage(
                id=image_id,
                sequence_no=0,
                original_filename=f"{display_id}.png",
                stored_filename=f"{image_id}.png",
                original_path=f"tasks/{image_id}/original.png",
                result_path=f"tasks/{image_id}/result.png",
                status=image_status,
                stage=image_stage,
                overall_result=OverallResult.ok,
                width=100,
                height=300,
                size_bytes=1234,
            )
            for index, label in enumerate(labels):
                detection = Detection(
                    region_label=label,
                    confidence=0.9 - index * 0.01,
                    points=[[1, 1], [9, 1], [9, 9], [1, 9]],
                    selected_for_classification=label in {"label3", "label5"},
                )
                if label == "label3":
                    detection.classifications.append(
                        ClassificationResult(
                            classifier_type="anomaly",
                            predicted_label="OK",
                            confidence=0.97,
                        )
                    )
                image.detections.append(detection)
                detections.append(detection)
            task.images.append(image)
            session.add(task)
            session.commit()
            detection_ids = {item.region_label: str(item.id) for item in detections}
        return str(image_id), detection_ids

    def valid_payload(self, *, verdict="OK"):
        return {
            "items": [
                {
                    "detectionId": self.detection_ids["label1_thin"],
                    "verdict": verdict,
                    "color": "B",
                },
                {
                    "detectionId": self.detection_ids["label3"],
                    "verdict": verdict,
                },
                {
                    "detectionId": self.detection_ids["label5"],
                    "verdict": verdict,
                },
            ],
            "missedRegions": ["label6"],
        }

    def put(self, image_id, payload):
        return self.client.put(f"/api/v1/images/{image_id}/feedback", json=payload)

    def test_logical_region_unifies_label1_variants_only(self):
        self.assertEqual(logical_region("label1_thin"), "label1")
        self.assertEqual(logical_region("label1_thick"), "label1")
        self.assertEqual(logical_region("label5"), "label5")

    def test_feedback_view_maps_label1_and_lists_only_missing_regions(self):
        response = self.client.get(f"/api/v1/images/{self.image_id}/feedback")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["imageId"], self.image_id)
        label1 = next(
            item for item in payload["detections"] if item["regionLabel"] == "label1_thin"
        )
        label3 = next(
            item for item in payload["detections"] if item["regionLabel"] == "label3"
        )
        self.assertEqual(label1["logicalRegion"], "label1")
        self.assertEqual(label3["anomaly"]["predictedLabel"], "OK")
        self.assertNotIn("label1", payload["missedRegionCandidates"])
        self.assertNotIn("label3", payload["missedRegionCandidates"])
        self.assertIn("label6", payload["missedRegionCandidates"])
        self.assertIsNone(payload["feedback"])

    def test_repeated_put_replaces_children_without_duplicate_parent(self):
        first = self.put(self.image_id, self.valid_payload(verdict="OK"))
        updated = self.valid_payload(verdict="NG")
        updated["items"][0]["color"] = "R"
        updated["missedRegions"] = ["label2", "label6"]
        second = self.put(self.image_id, updated)

        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(second.json()["feedback"]["items"][0]["verdict"], "NG")
        with self.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(ImageFeedback)),
                1,
            )
            label1_item = session.scalar(
                select(ImageFeedbackItem).where(
                    ImageFeedbackItem.detection_id
                    == uuid.UUID(self.detection_ids["label1_thin"])
                )
            )
            self.assertEqual(label1_item.verdict, "NG")
            self.assertEqual(label1_item.color, "R")

    def test_different_users_have_independent_feedback(self):
        self.assertEqual(
            self.put(self.image_id, self.valid_payload()).status_code,
            200,
        )
        second_client = TestClient(self.app)
        try:
            create_and_login(second_client, self.session_factory, username="reviewer-two")
            before = second_client.get(f"/api/v1/images/{self.image_id}/feedback")
            self.assertIsNone(before.json()["feedback"])
            second_payload = self.valid_payload(verdict="NG")
            saved = second_client.put(
                f"/api/v1/images/{self.image_id}/feedback",
                json=second_payload,
            )
            self.assertEqual(saved.status_code, 200, saved.text)
        finally:
            second_client.close()

        with self.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(ImageFeedback)),
                2,
            )

    def test_detection_ids_must_be_unique_and_match_current_image_exactly(self):
        missing = self.valid_payload()
        missing["items"].pop()
        duplicate = self.valid_payload()
        duplicate["items"].append(dict(duplicate["items"][0]))
        _, foreign_ids = self.seed_image("T-foreign", ("label2",))
        cross_image = self.valid_payload()
        cross_image["items"][0]["detectionId"] = foreign_ids["label2"]

        self.assertEqual(self.put(self.image_id, missing).status_code, 409)
        self.assertEqual(self.put(self.image_id, duplicate).status_code, 422)
        self.assertEqual(self.put(self.image_id, cross_image).status_code, 409)

        stale = self.valid_payload()
        with self.session_factory() as session:
            image = session.get(InspectionImage, uuid.UUID(self.image_id))
            image.detections.append(
                Detection(
                    region_label="label4",
                    confidence=0.8,
                    points=[[1, 1], [9, 1], [9, 9], [1, 9]],
                )
            )
            session.commit()
        self.assertEqual(self.put(self.image_id, stale).status_code, 409)

    def test_color_and_verdict_validation_follows_region_semantics(self):
        invalid_verdict = self.valid_payload()
        invalid_verdict["items"][1]["verdict"] = "MAYBE"
        missing_color = self.valid_payload()
        missing_color["items"][0].pop("color")
        invalid_color = self.valid_payload()
        invalid_color["items"][0]["color"] = "Y"
        unsupported_color = self.valid_payload()
        unsupported_color["items"][1]["color"] = "G"

        self.assertEqual(self.put(self.image_id, invalid_verdict).status_code, 422)
        self.assertEqual(self.put(self.image_id, missing_color).status_code, 422)
        self.assertEqual(self.put(self.image_id, invalid_color).status_code, 422)
        self.assertEqual(self.put(self.image_id, unsupported_color).status_code, 422)

    def test_missed_regions_must_be_unique_and_currently_undetected(self):
        detected_region = self.valid_payload()
        detected_region["missedRegions"] = ["label1"]
        duplicate = self.valid_payload()
        duplicate["missedRegions"] = ["label2", "label2"]
        unknown = self.valid_payload()
        unknown["missedRegions"] = ["label7"]

        self.assertEqual(self.put(self.image_id, detected_region).status_code, 422)
        self.assertEqual(self.put(self.image_id, duplicate).status_code, 422)
        self.assertEqual(self.put(self.image_id, unknown).status_code, 422)

    def test_empty_detection_image_requires_at_least_one_missed_region(self):
        empty_id, _ = self.seed_image("T-empty", ())

        empty = self.put(empty_id, {"items": [], "missedRegions": []})
        missed = self.put(empty_id, {"items": [], "missedRegions": ["label1"]})

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(missed.status_code, 200, missed.text)

    def test_unfinished_and_failed_images_reject_feedback(self):
        pending_id, _ = self.seed_image(
            "T-pending",
            (),
            image_status=ImageStatus.running,
            image_stage=ImageStage.object_detection,
        )
        failed_id, _ = self.seed_image(
            "T-failed",
            (),
            image_status=ImageStatus.failed,
            image_stage=ImageStage.complete,
        )

        pending = self.put(pending_id, {"items": [], "missedRegions": ["label1"]})
        failed = self.put(failed_id, {"items": [], "missedRegions": ["label1"]})

        self.assertEqual(pending.status_code, 409)
        self.assertEqual(failed.status_code, 409)


if __name__ == "__main__":
    unittest.main()
