import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from terminal_web.config import Settings


class SettingsTest(unittest.TestCase):
    def test_resolves_storage_and_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "DATABASE_URL": "postgresql+psycopg://app:app@localhost/app",
                "INSPECTION_STORAGE_ROOT": str(root / "storage"),
                "DETECTOR_WEIGHTS": str(root / "det.pt"),
                "LABEL3_CLASSIFIER_WEIGHTS": str(root / "label3.pt"),
                "LABEL5_CLASSIFIER_WEIGHTS": str(root / "label5.pt"),
            }
            with patch.dict(os.environ, env, clear=True):
                settings = Settings()

            self.assertEqual(settings.max_images_per_task, 100)
            self.assertEqual(settings.session_ttl_hours, 12)
            self.assertFalse(settings.session_cookie_secure)
            self.assertEqual(settings.storage_root, (root / "storage").resolve())
            self.assertEqual(settings.detector_weights, (root / "det.pt").resolve())

    def test_rejects_session_ttl_outside_supported_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "DATABASE_URL": "postgresql+psycopg://app:app@localhost/app",
                "INSPECTION_STORAGE_ROOT": str(root / "storage"),
                "DETECTOR_WEIGHTS": str(root / "det.pt"),
                "LABEL3_CLASSIFIER_WEIGHTS": str(root / "label3.pt"),
                "LABEL5_CLASSIFIER_WEIGHTS": str(root / "label5.pt"),
                "SESSION_TTL_HOURS": "0",
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "SESSION_TTL_HOURS"):
                    Settings()

    def test_rejects_non_postgresql_database_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "DATABASE_URL": "sqlite:///bad.db",
                "INSPECTION_STORAGE_ROOT": str(root / "storage"),
                "DETECTOR_WEIGHTS": str(root / "det.pt"),
                "LABEL3_CLASSIFIER_WEIGHTS": str(root / "label3.pt"),
                "LABEL5_CLASSIFIER_WEIGHTS": str(root / "label5.pt"),
            }
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "PostgreSQL"):
                    Settings()


if __name__ == "__main__":
    unittest.main()
