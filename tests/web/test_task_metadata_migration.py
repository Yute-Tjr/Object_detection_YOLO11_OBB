import importlib
import importlib.util
import unittest
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, inspect, text


MIGRATION_MODULE = "migrations.versions._20260922_03_remove_task_metadata"


class TaskMetadataMigrationTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")

    def tearDown(self):
        self.engine.dispose()

    def load_migration(self):
        self.assertIsNotNone(
            importlib.util.find_spec(MIGRATION_MODULE),
            "task metadata removal migration is missing",
        )
        return importlib.import_module(MIGRATION_MODULE)

    @staticmethod
    def create_legacy_table(connection) -> None:
        connection.execute(text(
            "CREATE TABLE inspection_tasks ("
            "id VARCHAR PRIMARY KEY, display_id VARCHAR NOT NULL, "
            "name VARCHAR, operator VARCHAR, note TEXT)"
        ))
        connection.execute(text(
            "INSERT INTO inspection_tasks VALUES "
            "('1', 'T-1', '早班', '张三', '首件')"
        ))

    def test_upgrade_drops_metadata_without_deleting_tasks(self):
        migration = self.load_migration()
        with self.engine.begin() as connection:
            self.create_legacy_table(connection)
            operations = Operations(MigrationContext.configure(connection))

            with patch.object(migration, "op", operations):
                migration.upgrade()

            columns = {
                column["name"]
                for column in inspect(connection).get_columns("inspection_tasks")
            }
            self.assertEqual(columns, {"id", "display_id"})
            self.assertEqual(
                connection.scalar(text("SELECT count(*) FROM inspection_tasks")),
                1,
            )

    def test_downgrade_restores_only_empty_nullable_metadata_columns(self):
        migration = self.load_migration()
        with self.engine.begin() as connection:
            self.create_legacy_table(connection)
            operations = Operations(MigrationContext.configure(connection))
            with patch.object(migration, "op", operations):
                migration.upgrade()
                migration.downgrade()

            columns = {
                column["name"]: column
                for column in inspect(connection).get_columns("inspection_tasks")
            }
            self.assertTrue(columns["operator"]["nullable"])
            self.assertTrue(columns["name"]["nullable"])
            self.assertTrue(columns["note"]["nullable"])
            row = connection.execute(text(
                "SELECT display_id, operator, name, note FROM inspection_tasks"
            )).one()
            self.assertEqual(row, ("T-1", None, None, None))


if __name__ == "__main__":
    unittest.main()
