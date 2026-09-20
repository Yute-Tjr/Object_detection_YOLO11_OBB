import unittest
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from terminal_web.database import Base
from terminal_web.domain import TaskStatus
from terminal_web.models import InspectionTask
from terminal_web.queue import claim_next_task, recover_stale_tasks


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine, expire_on_commit=False)
        self.now = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def make_running_task(
        self, attempt_count: int, heartbeat_age_seconds: int
    ) -> InspectionTask:
        task = InspectionTask(
            id=uuid.uuid4(),
            display_id=f"T-{uuid.uuid4().hex[:8]}",
            status=TaskStatus.running,
            attempt_count=attempt_count,
            worker_id="worker-old",
            heartbeat_at=self.now - timedelta(seconds=heartbeat_age_seconds),
        )
        self.session.add(task)
        self.session.commit()
        return task

    def test_claim_next_task_marks_it_running_and_increments_attempt(self):
        task = InspectionTask(
            id=uuid.uuid4(),
            display_id="T-claim",
            status=TaskStatus.queued,
        )
        self.session.add(task)
        self.session.commit()

        claimed = claim_next_task(self.session, "worker-1", self.now)

        self.assertEqual(claimed.id, task.id)
        self.assertEqual(task.status, TaskStatus.running)
        self.assertEqual(task.attempt_count, 1)
        self.assertEqual(task.worker_id, "worker-1")

    def test_recover_stale_task_requeues_below_attempt_limit(self):
        task = self.make_running_task(attempt_count=1, heartbeat_age_seconds=300)
        changed = recover_stale_tasks(
            self.session, self.now, timeout_seconds=120, max_attempts=2
        )
        self.assertEqual(changed, [task.id])
        self.assertEqual(task.status, TaskStatus.queued)
        self.assertIsNone(task.worker_id)

    def test_recover_stale_task_fails_at_attempt_limit(self):
        task = self.make_running_task(attempt_count=2, heartbeat_age_seconds=300)
        recover_stale_tasks(
            self.session, self.now, timeout_seconds=120, max_attempts=2
        )
        self.assertEqual(task.status, TaskStatus.failed)

    def test_fresh_running_task_is_untouched(self):
        task = self.make_running_task(attempt_count=1, heartbeat_age_seconds=30)
        changed = recover_stale_tasks(
            self.session, self.now, timeout_seconds=120, max_attempts=2
        )
        self.assertEqual(changed, [])
        self.assertEqual(task.status, TaskStatus.running)
