from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from terminal_web.models import ModelRecord
from terminal_web.schemas import HealthResponse, ModelHealth


EXPECTED_DETECTOR_LABELS = {
    "label1_thin",
    "label1_thick",
    "label2",
    "label3",
    "label4",
    "label5",
    "label6",
}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    name: str
    version: str
    path: Path
    load_config: dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_models(session: Session, specs: Sequence[ModelSpec]) -> list[ModelHealth]:
    health: list[ModelHealth] = []
    for spec in specs:
        resolved = spec.path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"weight file not found for {spec.name}")
        checksum = sha256_file(resolved)
        statement = select(ModelRecord).where(
            ModelRecord.model_type == spec.model_type,
            ModelRecord.name == spec.name,
            ModelRecord.version == spec.version,
        )
        record = session.scalar(statement)
        if record is None:
            record = ModelRecord(
                id=uuid.uuid4(),
                model_type=spec.model_type,
                name=spec.name,
                version=spec.version,
                weights_path=resolved.name,
                sha256=checksum,
                active=True,
                load_config=dict(spec.load_config),
            )
            session.add(record)
        else:
            record.weights_path = resolved.name
            record.sha256 = checksum
            record.active = True
            record.load_config = dict(spec.load_config)
        health.append(
            ModelHealth(
                model_type=spec.model_type,
                name=spec.name,
                version=spec.version,
                ready=True,
                sha256=checksum,
            )
        )
    session.flush()
    return health


def validate_pipeline_models(pipeline) -> None:
    detector_model = getattr(getattr(pipeline, "detector", None), "model", None)
    names = getattr(detector_model, "names", None)
    if names is not None:
        values = set(names.values() if isinstance(names, dict) else names)
        if values != EXPECTED_DETECTOR_LABELS:
            raise ValueError("detector class names do not match the terminal dataset")

    for label, classifier in getattr(pipeline, "classifiers", {}).items():
        loaded = getattr(classifier, "classifier", None)
        classes = getattr(loaded, "classes", None)
        if classes is not None and set(classes) != {"NG", "OK"}:
            raise ValueError(f"{label} classifier classes must be NG and OK")


class ReadinessStore:
    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: int,
        clock: Callable[[], datetime] | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.clock = clock or (lambda: datetime.now(UTC))

    def mark_ready(
        self,
        models: Sequence[ModelHealth],
        worker_id: str,
        now: datetime | None = None,
    ) -> None:
        moment = now or self.clock()
        self._write(
            {
                "models_ready": True,
                "worker_id": worker_id,
                "heartbeat_at": moment.isoformat(),
                "models": [model.model_dump() for model in models],
                "error": None,
            }
        )

    def mark_failed(
        self,
        error: str,
        models: Sequence[ModelHealth] = (),
    ) -> None:
        self._write(
            {
                "models_ready": False,
                "worker_id": None,
                "heartbeat_at": None,
                "models": [model.model_dump() for model in models],
                "error": error,
            }
        )

    def heartbeat(self, worker_id: str, now: datetime | None = None) -> None:
        state = self._read()
        state["worker_id"] = worker_id
        state["heartbeat_at"] = (now or self.clock()).isoformat()
        self._write(state)

    def snapshot(self) -> HealthResponse:
        state = self._read()
        models = [ModelHealth.model_validate(item) for item in state.get("models", [])]
        models_ready = bool(state.get("models_ready", False))
        heartbeat_text = state.get("heartbeat_at")
        worker_ready = False
        if models_ready and heartbeat_text:
            try:
                heartbeat = datetime.fromisoformat(heartbeat_text)
                if heartbeat.tzinfo is None:
                    heartbeat = heartbeat.replace(tzinfo=UTC)
                worker_ready = self.clock() - heartbeat <= timedelta(
                    seconds=self.timeout_seconds
                )
            except (TypeError, ValueError):
                worker_ready = False
        if not models and state.get("error"):
            models = [
                ModelHealth(
                    model_type="pipeline",
                    name="terminal-inspection",
                    version="configured",
                    ready=False,
                    error=str(state["error"]),
                )
            ]
        return HealthResponse(
            api_ready=True,
            database_ready=True,
            worker_ready=worker_ready,
            models_ready=models_ready,
            models=models,
        )

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _write(self, state: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.part")
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
