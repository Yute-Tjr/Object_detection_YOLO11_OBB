from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_FORMATS_BY_EXTENSION = {
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".png": {"PNG"},
    ".bmp": {"BMP"},
}


class InvalidImageError(ValueError):
    pass


@dataclass(frozen=True)
class StoredUpload:
    relative_path: str
    original_filename: str
    stored_filename: str
    width: int
    height: int
    size_bytes: int


class ArtifactStorage:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        task_id: uuid.UUID,
        image_id: uuid.UUID,
        original_filename: str,
        content: bytes,
    ) -> StoredUpload:
        display_name = original_filename.replace("\\", "/").rsplit("/", 1)[-1]
        extension = Path(display_name).suffix.lower()
        if not display_name or extension not in ALLOWED_IMAGE_EXTENSIONS:
            raise InvalidImageError("unsupported image filename or extension")

        width, height, image_format = self._decode_metadata(content)
        if image_format not in _FORMATS_BY_EXTENSION[extension]:
            raise InvalidImageError("image content does not match its extension")

        stored_filename = f"{image_id.hex}{extension}"
        destination = self._inside(
            self.root / "tasks" / str(task_id) / "originals" / stored_filename
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
        try:
            with temporary.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        return StoredUpload(
            relative_path=self.relative(destination),
            original_filename=display_name,
            stored_filename=stored_filename,
            width=width,
            height=height,
            size_bytes=len(content),
        )

    def crop_path(self, task_id: uuid.UUID, image_id: uuid.UUID, name: str) -> Path:
        safe_name = name.replace("\\", "/").rsplit("/", 1)[-1]
        if not safe_name or safe_name in {".", ".."}:
            raise ValueError("invalid crop name")
        path = self.root / "tasks" / str(task_id) / "crops" / f"{image_id.hex}_{safe_name}"
        path = self._inside(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def result_path(self, task_id: uuid.UUID, image_id: uuid.UUID) -> Path:
        path = self._inside(
            self.root / "tasks" / str(task_id) / "results" / f"{image_id.hex}.jpg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, absolute: Path) -> str:
        resolved = absolute.expanduser().resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("artifact path is outside storage root")
        return resolved.relative_to(self.root).as_posix()

    def resolve(self, relative: str) -> Path:
        supplied = Path(relative)
        if supplied.is_absolute():
            raise ValueError("artifact path must be relative")
        return self._inside(self.root / supplied)

    def _inside(self, path: Path) -> Path:
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("artifact path is outside storage root")
        return resolved

    @staticmethod
    def _decode_metadata(content: bytes) -> tuple[int, int, str]:
        if not content:
            raise InvalidImageError("empty image")
        try:
            with Image.open(io.BytesIO(content)) as image:
                image_format = image.format
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("image cannot be decoded") from exc

        decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.ndim != 3:
            raise InvalidImageError("image cannot be decoded")
        height, width = decoded.shape[:2]
        return width, height, image_format or ""
