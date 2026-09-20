import unittest
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from terminal_web.database import Base
from terminal_web.domain import ImageStage, ImageStatus, OverallResult, TaskStatus
from terminal_web.models import InspectionImage, InspectionTask
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
