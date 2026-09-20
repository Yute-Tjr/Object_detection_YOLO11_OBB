import io
import tempfile
import unittest
import uuid
from pathlib import Path

from PIL import Image

from terminal_web.storage import ArtifactStorage, InvalidImageError


def png_bytes(color=(10, 20, 30)) -> bytes:
    handle = io.BytesIO()
    Image.new("RGB", (20, 30), color).save(handle, format="PNG")
    return handle.getvalue()


class StorageTest(unittest.TestCase):
    def test_duplicate_original_names_get_unique_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = ArtifactStorage(Path(tmp))
            task_id = uuid.uuid4()
            first = storage.save_upload(task_id, uuid.uuid4(), "same.png", png_bytes())
            second = storage.save_upload(task_id, uuid.uuid4(), "same.png", png_bytes())
            self.assertNotEqual(first.relative_path, second.relative_path)

    def test_rejects_decoding_failure_even_with_image_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = ArtifactStorage(Path(tmp))
            with self.assertRaises(InvalidImageError):
                storage.save_upload(uuid.uuid4(), uuid.uuid4(), "fake.png", b"not-image")

    def test_resolver_blocks_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            storage = ArtifactStorage(Path(tmp))
            with self.assertRaises(ValueError):
                storage.resolve("../../etc/passwd")
