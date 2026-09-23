import importlib
import importlib.util
import unittest
import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect

from terminal_web.models import ImageFeedbackItem


MIGRATION_MODULE = "migrations.versions._20260923_04_add_auth_and_feedback"
SOURCE_MIGRATION_MODULE = (
    "migrations.versions._20260923_05_add_feedback_item_source"
)
NEW_TABLES = {
    "users",
    "user_sessions",
    "image_feedbacks",
    "image_feedback_items",
    "image_feedback_misses",
}


class AuthFeedbackMigrationTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")

    def tearDown(self):
        self.engine.dispose()

    def load_migration(self):
        self.assertIsNotNone(
            importlib.util.find_spec(MIGRATION_MODULE),
            "authentication and feedback migration is missing",
        )
        return importlib.import_module(MIGRATION_MODULE)

    @staticmethod
    def create_parent_tables(connection) -> None:
        metadata = sa.MetaData()
        sa.Table(
            "inspection_images",
            metadata,
            sa.Column("id", sa.Uuid(), primary_key=True),
        )
        sa.Table(
            "detections",
            metadata,
            sa.Column("id", sa.Uuid(), primary_key=True),
        )
        metadata.create_all(connection)

    def test_upgrade_creates_auth_and_feedback_tables_with_constraints(self):
        migration = self.load_migration()
        with self.engine.begin() as connection:
            self.create_parent_tables(connection)
            operations = Operations(MigrationContext.configure(connection))

            with patch.object(migration, "op", operations):
                migration.upgrade()

            inspector = inspect(connection)
            self.assertTrue(NEW_TABLES.issubset(set(inspector.get_table_names())))

            user_uniques = inspector.get_unique_constraints("users")
            self.assertIn(
                ["username"],
                [constraint["column_names"] for constraint in user_uniques],
            )
            session_uniques = inspector.get_unique_constraints("user_sessions")
            self.assertIn(
                ["token_hash"],
                [constraint["column_names"] for constraint in session_uniques],
            )
            feedback_uniques = inspector.get_unique_constraints("image_feedbacks")
            self.assertIn(
                ["user_id", "image_id"],
                [constraint["column_names"] for constraint in feedback_uniques],
            )

            feedback_foreign_keys = {
                tuple(foreign_key["constrained_columns"]): foreign_key
                for foreign_key in inspector.get_foreign_keys("image_feedbacks")
            }
            self.assertEqual(
                feedback_foreign_keys[("image_id",)]["options"].get("ondelete"),
                "CASCADE",
            )
            self.assertEqual(
                feedback_foreign_keys[("user_id",)]["options"].get("ondelete"),
                "RESTRICT",
            )

    def test_downgrade_removes_only_auth_and_feedback_tables(self):
        migration = self.load_migration()
        with self.engine.begin() as connection:
            self.create_parent_tables(connection)
            operations = Operations(MigrationContext.configure(connection))
            with patch.object(migration, "op", operations):
                migration.upgrade()
                migration.downgrade()

            remaining = set(inspect(connection).get_table_names())
            self.assertTrue(NEW_TABLES.isdisjoint(remaining))
            self.assertEqual(remaining, {"detections", "inspection_images"})

    def test_source_migration_marks_existing_items_manual_and_allows_null_verdict(self):
        base_migration = self.load_migration()
        self.assertIsNotNone(
            importlib.util.find_spec(SOURCE_MIGRATION_MODULE),
            "feedback item source migration is missing",
        )
        source_migration = importlib.import_module(SOURCE_MIGRATION_MODULE)

        with self.engine.begin() as connection:
            self.create_parent_tables(connection)
            operations = Operations(MigrationContext.configure(connection))
            with patch.object(base_migration, "op", operations):
                base_migration.upgrade()

            metadata = sa.MetaData()
            metadata.reflect(connection)
            now = datetime.now(UTC)
            user_id = uuid.uuid4().hex
            image_id = uuid.uuid4().hex
            detection_id = uuid.uuid4().hex
            feedback_id = uuid.uuid4().hex
            connection.execute(
                metadata.tables["inspection_images"].insert(), {"id": image_id}
            )
            connection.execute(
                metadata.tables["detections"].insert(), {"id": detection_id}
            )
            connection.execute(
                metadata.tables["users"].insert(),
                {
                    "id": user_id,
                    "username": "operator",
                    "password_hash": "hash",
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            connection.execute(
                metadata.tables["image_feedbacks"].insert(),
                {
                    "id": feedback_id,
                    "image_id": image_id,
                    "user_id": user_id,
                    "created_at": now,
                    "updated_at": now,
                },
            )
            connection.execute(
                metadata.tables["image_feedback_items"].insert(),
                {
                    "id": uuid.uuid4().hex,
                    "feedback_id": feedback_id,
                    "detection_id": detection_id,
                    "region_label": "label3",
                    "verdict": "NG",
                    "color": None,
                    "created_at": now,
                },
            )

            with patch.object(source_migration, "op", operations):
                source_migration.upgrade()

            columns = {
                item["name"]: item
                for item in inspect(connection).get_columns("image_feedback_items")
            }
            migrated = connection.execute(
                sa.text("SELECT source, verdict FROM image_feedback_items")
            ).mappings().one()

            self.assertFalse(columns["source"]["nullable"])
            self.assertTrue(columns["verdict"]["nullable"])
            self.assertEqual(migrated, {"source": "manual", "verdict": "NG"})

        self.assertIn(
            "source", {column.name for column in ImageFeedbackItem.__table__.columns}
        )


if __name__ == "__main__":
    unittest.main()
