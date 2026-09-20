import os
import unittest
import uuid
from datetime import UTC, datetime

from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import sessionmaker

from terminal_web.database import Base
from terminal_web.domain import TaskStatus
from terminal_web.models import InspectionTask
from terminal_web.queue import claim_next_task


@unittest.skipUnless(
    os.getenv("TEST_DATABASE_URL"),
    "TEST_DATABASE_URL is required for PostgreSQL queue-lock integration",
)
class PostgreSQLQueueIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema_name = f"terminal_test_{uuid.uuid4().hex}"
        cls.admin_engine = create_engine(os.environ["TEST_DATABASE_URL"])
        with cls.admin_engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA "{cls.schema_name}"'))
        cls.engine = create_engine(
            os.environ["TEST_DATABASE_URL"],
            connect_args={"options": f"-csearch_path={cls.schema_name}"},
        )
        Base.metadata.create_all(cls.engine)
        cls.sessions = sessionmaker(bind=cls.engine, expire_on_commit=False)

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()
        with cls.admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{cls.schema_name}" CASCADE'))
        cls.admin_engine.dispose()

    def test_skip_locked_allows_only_one_worker_to_claim_a_task(self):
        task_id = uuid.uuid4()
        with self.sessions() as setup:
            setup.add(
                InspectionTask(
                    id=task_id,
                    display_id=f"T-PG-{task_id.hex[:12]}",
                    status=TaskStatus.queued,
                    total_images=0,
                )
            )
            setup.commit()

        first = self.sessions()
        second = self.sessions()
        try:
            claimed_first = claim_next_task(first, "worker-one", datetime.now(UTC))
            claimed_second = claim_next_task(second, "worker-two", datetime.now(UTC))

            self.assertIsNotNone(claimed_first)
            self.assertEqual(claimed_first.id, task_id)
            self.assertIsNone(claimed_second)
        finally:
            first.rollback()
            second.rollback()
            first.close()
            second.close()
            with self.sessions() as cleanup:
                cleanup.execute(delete(InspectionTask).where(InspectionTask.id == task_id))
                cleanup.commit()

    def test_database_objects_are_isolated_from_public_schema(self):
        with self.engine.connect() as connection:
            self.assertNotEqual(connection.scalar(text("SELECT current_schema()")), "public")
