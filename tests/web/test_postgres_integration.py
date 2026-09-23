import os
import unittest
import uuid
from datetime import UTC, datetime

from sqlalchemy import create_engine, delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from terminal_web.database import Base
from terminal_web.domain import TaskStatus
from terminal_web.models import ImageFeedback, InspectionImage, InspectionTask, User
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

    def test_username_constraints_are_case_sensitive_and_exact_values_are_unique(self):
        first_id = uuid.uuid4()
        second_id = uuid.uuid4()
        with self.sessions() as session:
            session.add_all(
                [
                    User(id=first_id, username="Admin", password_hash="hash-one"),
                    User(id=second_id, username="admin", password_hash="hash-two"),
                ]
            )
            session.commit()

        with self.sessions() as session:
            session.add(User(username="Admin", password_hash="hash-three"))
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

        with self.sessions() as cleanup:
            cleanup.execute(delete(User).where(User.id.in_([first_id, second_id])))
            cleanup.commit()

    def test_feedback_uniqueness_image_cascade_and_user_restrict(self):
        task_id = uuid.uuid4()
        image_id = uuid.uuid4()
        user_id = uuid.uuid4()
        feedback_id = uuid.uuid4()
        with self.sessions() as session:
            session.add(
                InspectionTask(
                    id=task_id,
                    display_id=f"T-FEEDBACK-{task_id.hex[:12]}",
                    status=TaskStatus.succeeded,
                    total_images=1,
                    completed_images=1,
                    succeeded_images=1,
                )
            )
            session.add(
                InspectionImage(
                    id=image_id,
                    task_id=task_id,
                    sequence_no=0,
                    original_filename="terminal.png",
                    stored_filename="terminal.png",
                    original_path="tasks/test/original/terminal.png",
                    status="succeeded",
                    stage="complete",
                    overall_result="OK",
                    width=380,
                    height=1606,
                    size_bytes=1024,
                )
            )
            session.add(User(id=user_id, username="feedback-user", password_hash="hash"))
            session.flush()
            session.add(ImageFeedback(id=feedback_id, image_id=image_id, user_id=user_id))
            session.commit()

        with self.sessions() as session:
            session.add(ImageFeedback(image_id=image_id, user_id=user_id))
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

        with self.sessions() as session:
            with self.assertRaises(IntegrityError):
                session.execute(delete(User).where(User.id == user_id))
                session.commit()
            session.rollback()

        with self.sessions() as session:
            session.execute(delete(InspectionImage).where(InspectionImage.id == image_id))
            session.commit()
            remaining = session.scalar(
                select(func.count()).select_from(ImageFeedback).where(
                    ImageFeedback.id == feedback_id
                )
            )
            self.assertEqual(remaining, 0)
            session.execute(delete(User).where(User.id == user_id))
            session.execute(delete(InspectionTask).where(InspectionTask.id == task_id))
            session.commit()
