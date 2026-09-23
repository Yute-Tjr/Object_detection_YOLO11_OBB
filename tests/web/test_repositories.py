import unittest
import uuid
from inspect import signature

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from terminal_web.database import Base
from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus
from terminal_web.auth import hash_password
from terminal_web.models import ImageFeedback, InspectionImage, InspectionTask, User
from terminal_web.repositories import TaskRepository


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine, expire_on_commit=False)
        self.repo = TaskRepository(self.session)

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def make_task_with_images(self, statuses: list[ImageStatus]) -> InspectionTask:
        task = InspectionTask(
            id=uuid.uuid4(),
            display_id=f"T-{uuid.uuid4().hex[:8]}",
            status=TaskStatus.running,
            total_images=len(statuses),
        )
        for index, status in enumerate(statuses):
            task.images.append(
                InspectionImage(
                    id=uuid.uuid4(),
                    sequence_no=index,
                    original_filename=f"image-{index}.png",
                    stored_filename=f"{index}.png",
                    original_path=f"originals/{index}.png",
                    status=status,
                    stage=ImageStage.complete,
                    overall_result=OverallResult.unknown,
                    width=20,
                    height=30,
                    size_bytes=100,
                )
            )
        self.session.add(task)
        self.session.commit()
        return task

    def test_recompute_task_progress_counts_success_and_failure(self):
        task = self.make_task_with_images(
            [ImageStatus.succeeded, ImageStatus.failed]
        )
        self.repo.recompute_progress(task.id)
        self.assertEqual(task.completed_images, 2)
        self.assertEqual(task.succeeded_images, 1)
        self.assertEqual(task.failed_images, 1)
        self.assertEqual(task.status, TaskStatus.partial_failed)

    def test_create_task_persists_required_task_fields(self):
        self.assertEqual(
            tuple(signature(self.repo.create_task).parameters),
            ("task_id", "display_id", "images"),
        )
        task = self.repo.create_task(uuid.uuid4(), "T-create", [])

        self.assertEqual(task.display_id, "T-create")
        self.assertEqual(task.total_images, 0)
        self.assertNotIn("operator", InspectionTask.__table__.columns)
        self.assertNotIn("name", InspectionTask.__table__.columns)
        self.assertNotIn("note", InspectionTask.__table__.columns)

    def test_failed_filter_includes_failed_and_partial_failed(self):
        for status in (TaskStatus.failed, TaskStatus.partial_failed, TaskStatus.succeeded):
            self.session.add(
                InspectionTask(
                    id=uuid.uuid4(),
                    display_id=f"T-{uuid.uuid4().hex[:8]}",
                    status=status,
                )
            )
        self.session.commit()

        tasks, total = self.repo.list_tasks([TaskStatus.failed], None, 0, 20)

        self.assertEqual(total, 2)
        self.assertEqual(
            {task.status for task in tasks},
            {TaskStatus.failed, TaskStatus.partial_failed},
        )

    def test_feedback_queries_identify_only_tasks_with_feedback(self):
        with_feedback = self.make_task_with_images([ImageStatus.succeeded])
        without_feedback = self.make_task_with_images([ImageStatus.succeeded])
        user = User(
            username="reviewer",
            password_hash=hash_password("Password-reviewer-2026"),
        )
        self.session.add(user)
        self.session.flush()
        self.session.add(
            ImageFeedback(
                user_id=user.id,
                image_id=with_feedback.images[0].id,
            )
        )
        self.session.commit()

        self.assertTrue(self.repo.task_has_feedback(with_feedback.id))
        self.assertFalse(self.repo.task_has_feedback(without_feedback.id))
        self.assertEqual(
            self.repo.feedback_task_ids([with_feedback.id, without_feedback.id]),
            {with_feedback.id},
        )
