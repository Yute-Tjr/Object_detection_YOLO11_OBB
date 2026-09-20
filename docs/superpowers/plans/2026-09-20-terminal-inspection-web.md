# Terminal Inspection Web System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-oriented web application that accepts batches of 1-100 terminal images, runs YOLO11l-OBB detection plus label3/label5 ResNet18 anomaly classification, persists task history in PostgreSQL, and presents the approved upload/progress/comparison/task-table interface.

**Architecture:** A React/Vite frontend talks to a FastAPI API. PostgreSQL stores the persistent task queue and structured results, a configurable filesystem stores image artifacts, and one independent GPU worker claims queued jobs with PostgreSQL row locks, loads all three models once, and processes images sequentially. The first milestone proves the backend flow with fake inference, the second integrates the real checkpoints, and the third implements and verifies the selected UI.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x, Alembic, psycopg 3, PostgreSQL 16, Pydantic Settings, Ultralytics, PyTorch/Torchvision, OpenCV, React 19, TypeScript 5, Vite 7, React Router 7, TanStack Query 5, Vitest, Testing Library, Docker Compose, Nginx.

**Spec:** `docs/superpowers/specs/2026-09-19-terminal-inspection-web-design.md`

## Global Constraints

- Accept `JPG`, `JPEG`, `PNG`, and `BMP`; each task contains 1-100 decodable images.
- Use `weights/detector/yolo11l_obb_best.pt`, `weights/classifiers/label3/resnet18_best.pt`, and `weights/classifiers/label5/resnet18_best.pt` through environment variables; never commit weights.
- Use PostgreSQL for tasks and results; store original, crop, and rendered images under `INSPECTION_STORAGE_ROOT` and only relative paths in the database.
- Run exactly one inference worker in the first deployment; never run model inference inside an API request.
- Poll running task state once per second while the page is visible; stop polling at a terminal task state.
- `label3`/`label5` OK boxes are green, NG boxes are red, unsupported anomaly-classification regions are neutral gray, and incomplete classification yields `UNKNOWN`, never `OK`.
- Color classification is optional. A missing color model returns `null` and never fails anomaly classification or the task.
- A single image failure does not abort the batch. A task with mixed image outcomes is `partial_failed` and appears under the failed filter.
- First release has no authentication, WebSocket, online model management, automatic history deletion, multi-worker scheduling, or editable boxes.
- Preserve existing CLI training/evaluation behavior and existing tests.
- Use the approved fusion mock as the visual target: upload and live progress first, before/after comparison second, expandable task table third.

## Review Focus

- A batch containing one corrupted image among valid images must finish `partial_failed`, preserve the valid results, and expose the corrupt image error without a traceback.
- A Worker crash after claiming a task must not leave it permanently `running`; an expired heartbeat must requeue it once and eventually fail after the configured maximum attempts.
- Duplicate original filenames in the same or different tasks must never overwrite files; UUID storage names and database ordering must preserve both uploads.
- Missing or incompatible checkpoint files must make `/api/v1/health` report `models_ready=false` and make task creation return `503` without creating a task directory.
- A missing label3 or label5 detection, classifier exception, or absent color plugin must never be turned into a false `OK`; the image result must be `UNKNOWN` except that an explicit NG still makes the image `NG`.

---

## Milestone 1: Persistent API and Task Queue

### Task 1: Add web dependencies and typed configuration

**Files:**
- Create: `requirements-web.txt`
- Create: `.env.example`
- Create: `terminal_web/__init__.py`
- Create: `terminal_web/config.py`
- Test: `tests/web/test_config.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: process environment and repository root.
- Produces: `Settings`, `get_settings()`, `ModelPaths`, and normalized filesystem paths used by API and Worker tasks.

- [ ] **Step 1: Write configuration tests**

```python
# tests/web/test_config.py
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
            self.assertEqual(settings.storage_root, (root / "storage").resolve())
            self.assertEqual(settings.detector_weights, (root / "det.pt").resolve())

    def test_rejects_non_postgresql_database_url(self):
        with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///bad.db"}, clear=True):
            with self.assertRaisesRegex(ValueError, "PostgreSQL"):
                Settings()
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `.venv/bin/python -m unittest tests.web.test_config -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'terminal_web'`.

- [ ] **Step 3: Add pinned web dependencies**

```text
# requirements-web.txt
-r requirements.txt
fastapi>=0.116,<1
uvicorn[standard]>=0.35,<1
sqlalchemy>=2.0,<3
alembic>=1.16,<2
psycopg[binary]>=3.2,<4
pydantic-settings>=2.10,<3
python-multipart>=0.0.20,<1
httpx>=0.28,<1
```

- [ ] **Step 4: Implement `Settings` and cached access**

```python
# terminal_web/config.py
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    storage_root: Path = Field(validation_alias="INSPECTION_STORAGE_ROOT")
    detector_weights: Path
    label3_classifier_weights: Path
    label5_classifier_weights: Path
    detection_device: str = "cpu"
    classification_device: str = "cpu"
    detection_imgsz: int = 1280
    classification_imgsz: int = 224
    detection_confidence: float = 0.25
    max_images_per_task: int = 100
    task_max_attempts: int = 2
    worker_heartbeat_timeout_seconds: int = 120

    @field_validator("database_url")
    @classmethod
    def require_postgresql(cls, value: str) -> str:
        if not value.startswith("postgresql+"):
            raise ValueError("DATABASE_URL must use PostgreSQL")
        return value

    @field_validator(
        "storage_root",
        "detector_weights",
        "label3_classifier_weights",
        "label5_classifier_weights",
    )
    @classmethod
    def resolve_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 5: Install the web dependencies**

Run: `.venv/bin/python -m pip install -r requirements-web.txt`

Expected: installation exits 0 and `.venv/bin/python -c "import fastapi, sqlalchemy, psycopg, alembic"` exits 0.

- [ ] **Step 6: Add environment examples and ignore runtime state**

```text
# .env.example
DATABASE_URL=postgresql+psycopg://terminal:terminal@localhost:5432/terminal_inspection
INSPECTION_STORAGE_ROOT=./var/terminal-inspection
DETECTOR_WEIGHTS=./weights/detector/yolo11l_obb_best.pt
LABEL3_CLASSIFIER_WEIGHTS=./weights/classifiers/label3/resnet18_best.pt
LABEL5_CLASSIFIER_WEIGHTS=./weights/classifiers/label5/resnet18_best.pt
DETECTION_DEVICE=cpu
CLASSIFICATION_DEVICE=cpu
```

Append `/var/` and `.env` to `.gitignore` without changing the existing `/weights/` rule.

- [ ] **Step 7: Run configuration tests**

Run: `.venv/bin/python -m unittest tests.web.test_config -v`

Expected: 2 tests pass.

- [ ] **Step 8: Commit the configuration foundation**

```bash
git add requirements-web.txt .env.example .gitignore terminal_web tests/web/test_config.py
git commit -m "feat: add terminal web configuration"
```

### Task 2: Define persistent domain models and migrations

**Files:**
- Create: `terminal_web/domain.py`
- Create: `terminal_web/database.py`
- Create: `terminal_web/models.py`
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/versions/_20260920_01_initial_schema.py`
- Test: `tests/web/test_domain.py`

**Interfaces:**
- Consumes: `Settings.database_url`.
- Produces: `TaskStatus`, `ImageStatus`, `ImageStage`, `OverallResult`, SQLAlchemy `Base`, `SessionFactory`, and ORM models `InspectionTask`, `InspectionImage`, `Detection`, `ClassificationResult`, `ModelRecord`.

- [ ] **Step 1: Write state transition and aggregation tests**

```python
# tests/web/test_domain.py
import unittest

from terminal_web.domain import (
    ImageOutcome,
    OverallResult,
    TaskStatus,
    aggregate_image_result,
    aggregate_task_status,
    ensure_task_transition,
)


class DomainTest(unittest.TestCase):
    def test_partial_failure_is_not_success(self):
        status = aggregate_task_status([ImageOutcome.succeeded, ImageOutcome.failed])
        self.assertEqual(status, TaskStatus.partial_failed)

    def test_missing_classifier_result_is_unknown(self):
        self.assertEqual(
            aggregate_image_result({"label3": "OK", "label5": None}),
            OverallResult.unknown,
        )

    def test_explicit_ng_dominates_missing_region(self):
        self.assertEqual(
            aggregate_image_result({"label3": "NG", "label5": None}),
            OverallResult.ng,
        )

    def test_terminal_task_cannot_return_to_running(self):
        with self.assertRaisesRegex(ValueError, "invalid task transition"):
            ensure_task_transition(TaskStatus.succeeded, TaskStatus.running)
```

- [ ] **Step 2: Run domain tests and verify failure**

Run: `.venv/bin/python -m unittest tests.web.test_domain -v`

Expected: FAIL because `terminal_web.domain` does not exist.

- [ ] **Step 3: Implement enums, transition rules, and aggregation**

```python
# terminal_web/domain.py
from enum import StrEnum


class TaskStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    partial_failed = "partial_failed"
    failed = "failed"


class ImageStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class ImageStage(StrEnum):
    pending = "pending"
    object_detection = "object_detection"
    anomaly_classification = "anomaly_classification"
    rendering = "rendering"
    complete = "complete"


class OverallResult(StrEnum):
    ok = "OK"
    ng = "NG"
    unknown = "UNKNOWN"


class ImageOutcome(StrEnum):
    succeeded = "succeeded"
    failed = "failed"


TERMINAL_TASK_STATUSES = {
    TaskStatus.succeeded,
    TaskStatus.partial_failed,
    TaskStatus.failed,
}

_TASK_TRANSITIONS = {
    TaskStatus.queued: {TaskStatus.running, TaskStatus.failed},
    TaskStatus.running: {
        TaskStatus.queued,
        TaskStatus.succeeded,
        TaskStatus.partial_failed,
        TaskStatus.failed,
    },
}


def ensure_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in _TASK_TRANSITIONS.get(current, set()):
        raise ValueError(f"invalid task transition: {current} -> {target}")


def aggregate_task_status(outcomes: list[ImageOutcome]) -> TaskStatus:
    successes = outcomes.count(ImageOutcome.succeeded)
    failures = outcomes.count(ImageOutcome.failed)
    if successes and failures:
        return TaskStatus.partial_failed
    return TaskStatus.succeeded if successes else TaskStatus.failed


def aggregate_image_result(labels: dict[str, str | None]) -> OverallResult:
    values = {value for value in labels.values() if value}
    if "NG" in values:
        return OverallResult.ng
    if labels.get("label3") == "OK" and labels.get("label5") == "OK":
        return OverallResult.ok
    return OverallResult.unknown
```

- [ ] **Step 4: Define ORM models with UUID keys and timestamp mixins**

Implement `terminal_web/models.py` with PostgreSQL-compatible SQLAlchemy 2 declarative models. Store OBB points and probability maps as `JSON`, use string enums for portable tests, add indexes on `(status, created_at)`, `task_id`, and `original_filename`, and enforce `sequence_no` uniqueness per task.

The minimum public relationships must be:

```python
InspectionTask.images: Mapped[list[InspectionImage]]
InspectionImage.detections: Mapped[list[Detection]]
Detection.classifications: Mapped[list[ClassificationResult]]
```

- [ ] **Step 5: Add database engine/session factories**

```python
# terminal_web/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def build_session_factory(database_url: str):
    engine = create_engine(database_url, pool_pre_ping=True)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
```

- [ ] **Step 6: Add an explicit Alembic initial migration**

Create all five tables from the spec: `inspection_tasks`, `inspection_images`, `detections`, `classification_results`, and `model_registry`. The migration must include foreign-key cascade behavior only from task to child records; it must not delete files from disk.

- [ ] **Step 7: Run domain tests and migration import check**

Run:

```bash
.venv/bin/python -m unittest tests.web.test_domain -v
.venv/bin/python -c "from migrations.versions import _20260920_01_initial_schema"
```

Expected: all domain tests pass and the migration imports without syntax errors.

- [ ] **Step 8: Commit the persistent model layer**

```bash
git add terminal_web/domain.py terminal_web/database.py terminal_web/models.py alembic.ini migrations tests/web/test_domain.py
git commit -m "feat: add inspection persistence schema"
```

### Task 3: Implement safe artifact storage and upload validation

**Files:**
- Create: `terminal_web/storage.py`
- Test: `tests/web/test_storage.py`

**Interfaces:**
- Consumes: raw upload streams, task/image UUIDs, `Settings.storage_root`.
- Produces: `StoredUpload(relative_path, original_filename, stored_filename, width, height, size_bytes)` and safe resolvers used by API and Worker.

- [ ] **Step 1: Write storage boundary tests**

```python
# tests/web/test_storage.py
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
```

- [ ] **Step 2: Run storage tests and verify failure**

Run: `.venv/bin/python -m unittest tests.web.test_storage -v`

Expected: FAIL because storage interfaces are missing.

- [ ] **Step 3: Implement validated, atomic storage**

`ArtifactStorage` must expose these exact methods:

| method | return |
| --- | --- |
| `save_upload(task_id: UUID, image_id: UUID, original_filename: str, content: bytes)` | `StoredUpload` |
| `crop_path(task_id: UUID, image_id: UUID, name: str)` | absolute `Path` under the task crop directory |
| `result_path(task_id: UUID, image_id: UUID)` | absolute `Path` under the task result directory |
| `relative(absolute: Path)` | POSIX relative path string |
| `resolve(relative: str)` | validated absolute `Path` inside the storage root |

Use Pillow verification plus OpenCV decoding, UUID filenames, temporary `*.part` files, `Path.replace()` for atomic completion, and a strict root containment check using `resolved.is_relative_to(self.root)`.

- [ ] **Step 4: Run storage tests**

Run: `.venv/bin/python -m unittest tests.web.test_storage -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit safe storage**

```bash
git add terminal_web/storage.py tests/web/test_storage.py
git commit -m "feat: add inspection artifact storage"
```

### Task 4: Build repositories and PostgreSQL queue semantics

**Files:**
- Create: `terminal_web/repositories.py`
- Create: `terminal_web/queue.py`
- Test: `tests/web/test_repositories.py`
- Test: `tests/web/test_queue.py`

**Interfaces:**
- Consumes: SQLAlchemy `Session`, domain enums, ORM models.
- Produces: `TaskRepository`, `claim_next_task(session, worker_id, now)`, `recover_stale_tasks(session, now, timeout, max_attempts)`, and progress aggregation.

- [ ] **Step 1: Write repository progress tests using a temporary SQLite session**

Use SQLite only for deterministic repository unit tests; PostgreSQL locking is covered by the Compose integration test in Task 12.

```python
def test_recompute_task_progress_counts_success_and_failure(self):
    task = make_task_with_images([ImageStatus.succeeded, ImageStatus.failed])
    self.repo.recompute_progress(task.id)
    self.assertEqual(task.completed_images, 2)
    self.assertEqual(task.succeeded_images, 1)
    self.assertEqual(task.failed_images, 1)
    self.assertEqual(task.status, TaskStatus.partial_failed)
```

- [ ] **Step 2: Write stale heartbeat recovery tests**

```python
def test_recover_stale_task_requeues_below_attempt_limit(self):
    task = self.make_running_task(attempt_count=1, heartbeat_age_seconds=300)
    changed = recover_stale_tasks(self.session, self.now, timeout_seconds=120, max_attempts=2)
    self.assertEqual(changed, [task.id])
    self.assertEqual(task.status, TaskStatus.queued)
    self.assertIsNone(task.worker_id)

def test_recover_stale_task_fails_at_attempt_limit(self):
    task = self.make_running_task(attempt_count=2, heartbeat_age_seconds=300)
    recover_stale_tasks(self.session, self.now, timeout_seconds=120, max_attempts=2)
    self.assertEqual(task.status, TaskStatus.failed)

def test_fresh_running_task_is_untouched(self):
    task = self.make_running_task(attempt_count=1, heartbeat_age_seconds=30)
    changed = recover_stale_tasks(self.session, self.now, timeout_seconds=120, max_attempts=2)
    self.assertEqual(changed, [])
    self.assertEqual(task.status, TaskStatus.running)
```

- [ ] **Step 3: Run tests and verify missing implementation failures**

Run: `.venv/bin/python -m unittest tests.web.test_repositories tests.web.test_queue -v`

Expected: FAIL on missing repository and queue modules.

- [ ] **Step 4: Implement repository CRUD and progress aggregation**

`TaskRepository` must provide:

```python
create_task(task_id, display_id, images) -> InspectionTask
get_task(task_id) -> InspectionTask | None
list_tasks(statuses, query, offset, limit) -> tuple[list[InspectionTask], int]
get_image(image_id) -> InspectionImage | None
recompute_progress(task_id) -> InspectionTask
reset_failed_image(image_id) -> InspectionImage
```

The failed filter maps to `{TaskStatus.failed, TaskStatus.partial_failed}`.

- [ ] **Step 5: Implement PostgreSQL task claiming and recovery**

Use this PostgreSQL locking statement; claim and status update occur in one transaction:

```python
statement = (
    select(InspectionTask)
    .where(InspectionTask.status == TaskStatus.queued)
    .order_by(InspectionTask.created_at)
    .with_for_update(skip_locked=True)
    .limit(1)
)
task = session.scalar(statement)
```

Recovery compares `heartbeat_at` with `now - timeout`, clears `worker_id`, and either requeues or fails based on `attempt_count`.

- [ ] **Step 6: Run repository and queue tests**

Run: `.venv/bin/python -m unittest tests.web.test_repositories tests.web.test_queue -v`

Expected: all tests pass.

- [ ] **Step 7: Commit repository and queue behavior**

```bash
git add terminal_web/repositories.py terminal_web/queue.py tests/web/test_repositories.py tests/web/test_queue.py
git commit -m "feat: add persistent inspection task queue"
```

### Task 5: Implement FastAPI task and artifact endpoints with fake readiness

**Files:**
- Create: `terminal_web/schemas.py`
- Create: `terminal_web/api/__init__.py`
- Create: `terminal_web/api/dependencies.py`
- Create: `terminal_web/api/health.py`
- Create: `terminal_web/api/tasks.py`
- Create: `terminal_web/api/images.py`
- Create: `terminal_web/api/app.py`
- Create: `scripts/run_terminal_api.py`
- Test: `tests/web/test_api.py`

**Interfaces:**
- Consumes: `Settings`, `TaskRepository`, `ArtifactStorage`, a `ReadinessProvider` protocol.
- Produces: `/api/v1/health`, task CRUD/upload endpoints, image metadata/artifact endpoints, and a runnable ASGI app.

- [ ] **Step 1: Write API tests with dependency overrides**

Cover these cases explicitly:

```python
def test_create_task_accepts_one_valid_image(self):
    response = self.client.post("/api/v1/tasks", files=self.valid_files(1))
    self.assertEqual(response.status_code, 202)
    self.assertEqual(response.json()["totalImages"], 1)

def test_create_task_rejects_101_images_without_writing_files(self):
    response = self.client.post("/api/v1/tasks", files=self.valid_files(101))
    self.assertEqual(response.status_code, 400)
    self.assertEqual(self.repository.count_tasks(), 0)
    self.assertEqual(list(self.storage_root.iterdir()), [])

def test_create_task_rejects_when_models_not_ready_without_creating_task(self):
    self.readiness.models_ready = False
    response = self.client.post("/api/v1/tasks", files=self.valid_files(1))
    self.assertEqual(response.status_code, 503)
    self.assertEqual(self.repository.count_tasks(), 0)

def test_create_task_with_one_corrupt_file_is_rejected_atomically(self):
    files = self.valid_files(1) + [("files", ("bad.png", b"broken", "image/png"))]
    response = self.client.post("/api/v1/tasks", files=files)
    self.assertEqual(response.status_code, 400)
    self.assertEqual(self.repository.count_tasks(), 0)

def test_failed_filter_includes_partial_failed(self):
    response = self.client.get("/api/v1/tasks?status=failed")
    statuses = {item["status"] for item in response.json()["items"]}
    self.assertEqual(statuses, {"failed", "partial_failed"})

def test_retry_only_accepts_failed_image(self):
    image = self.repository.add_image(status="succeeded")
    response = self.client.post(f"/api/v1/images/{image.id}/retry")
    self.assertEqual(response.status_code, 409)
```

Use `fastapi.testclient.TestClient` and in-memory repository/storage doubles. The corrupt mixed batch test is the Review Focus test for atomic task creation.

- [ ] **Step 2: Run API tests and verify failure**

Run: `.venv/bin/python -m unittest tests.web.test_api -v`

Expected: FAIL because the app is not implemented.

- [ ] **Step 3: Define stable API schemas**

At minimum define:

```python
class HealthResponse(BaseModel):
    api_ready: bool
    database_ready: bool
    worker_ready: bool
    models_ready: bool
    models: list[ModelHealth]

class TaskSummary(BaseModel):
    id: UUID
    display_id: str
    status: TaskStatus
    current_stage: ImageStage
    total_images: int
    completed_images: int
    succeeded_images: int
    failed_images: int
    detector_model: str
    created_at: datetime

class TaskDetail(TaskSummary):
    images: list[ImageSummary]

class ImageDetail(ImageSummary):
    detections: list[DetectionResponse]
```

All timestamps are timezone-aware ISO 8601 UTC. All enums serialize to the values in Task 2.

- [ ] **Step 4: Implement atomic upload and task creation**

Read and validate all files first into task-scoped temporary storage. Only after every file validates should the API insert task/image rows and atomically rename files into `originals/`. On any validation or database failure, remove the temporary task directory.

Return `202 Accepted` with `TaskDetail`. Return `400` for count/format/decode errors, `503` for missing model readiness or database/worker unavailability.

- [ ] **Step 5: Implement task listing, detail, artifacts, and retry**

The image result endpoint returns `409` when a result does not exist yet. Retry resets only a failed image; if its task is terminal, set the task back to `queued` and clear terminal timestamps.

- [ ] **Step 6: Run API tests**

Run: `.venv/bin/python -m unittest tests.web.test_api -v`

Expected: all tests pass.

- [ ] **Step 7: Commit the API milestone**

```bash
git add terminal_web/api terminal_web/schemas.py scripts/run_terminal_api.py tests/web/test_api.py
git commit -m "feat: add terminal inspection task API"
```

## Milestone 2: Real Detection and Classification Worker

### Task 6: Refactor the existing pipeline into preloadable single-image inference

**Files:**
- Create: `terminal_web/inference/__init__.py`
- Create: `terminal_web/inference/types.py`
- Create: `terminal_web/inference/color.py`
- Create: `terminal_web/inference/pipeline.py`
- Modify: `yolo11_obb/pipeline_predict.py`
- Test: `tests/web/test_inference_pipeline.py`
- Test: `tests/test_pipeline_predict.py`

**Interfaces:**
- Consumes: existing OBB extraction/crop functions, three weight paths, one image path, optional `ColorClassifier`.
- Produces: `LoadedInferencePipeline.predict_image(image_path, artifacts, stage_callback) -> ImagePrediction` without reloading weights per image.

- [ ] **Step 1: Define immutable inference result types**

```python
# terminal_web/inference/types.py
from collections.abc import Sequence


@dataclass(frozen=True)
class ClassificationPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class RegionPrediction:
    region_label: str
    detection_confidence: float
    points: Sequence[tuple[float, float]]
    selected_for_classification: bool
    anomaly: ClassificationPrediction | None
    color: ClassificationPrediction | None
    crop_path: str | None
    error: str | None


@dataclass(frozen=True)
class ImagePrediction:
    regions: Sequence[RegionPrediction]
    overall_result: OverallResult
    warnings: Sequence[str]
```

- [ ] **Step 2: Write fake-model pipeline tests**

Tests must prove:

- Stage callback order is `object_detection`, `anomaly_classification`, `rendering`, `complete`.
- All non-label3/label5 detections remain in results with no anomaly classification.
- Duplicate label3 detections classify only the highest-confidence detection and preserve the others.
- Missing label5 plus label3 OK yields `UNKNOWN`.
- label3 NG plus missing label5 yields `NG`.
- A color classifier returning `None` leaves `color=None` and does not change overall result.
- A classifier exception is captured as a region error and yields `UNKNOWN`, not false OK.

- [ ] **Step 3: Run inference tests and verify failure**

Run: `.venv/bin/python -m unittest tests.web.test_inference_pipeline -v`

Expected: FAIL because `LoadedInferencePipeline` is missing.

- [ ] **Step 4: Implement model adapters and optional color protocol**

```python
class ColorClassifier(Protocol):
    def predict(self, crop: np.ndarray, region_label: str) -> ClassificationPrediction | None:
        raise NotImplementedError


class NullColorClassifier:
    def predict(self, crop, region_label):
        return None
```

`LoadedInferencePipeline.__init__` loads the detector and both classifiers once. Add injectable detector/classifier adapters so tests never need real weights.

- [ ] **Step 5: Implement `predict_image` using existing helpers**

Reuse `detections_from_yolo_result`, `selected_detection_by_label`, and `rectify_obb_crop`. Move batch CLI behavior onto this new single-image primitive while preserving existing `run_pipeline` output fields and CLI tests.

- [ ] **Step 6: Run new and existing pipeline tests**

Run:

```bash
.venv/bin/python -m unittest tests.web.test_inference_pipeline -v
.venv/bin/python -m unittest tests.test_pipeline_predict tests.test_pipeline_evaluate -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the reusable inference pipeline**

```bash
git add terminal_web/inference yolo11_obb/pipeline_predict.py tests/web/test_inference_pipeline.py tests/test_pipeline_predict.py
git commit -m "refactor: expose preloadable terminal inference pipeline"
```

### Task 7: Render production result images with exact color semantics

**Files:**
- Create: `terminal_web/inference/rendering.py`
- Test: `tests/web/test_rendering.py`
- Modify: `yolo11_obb/pipeline_predict.py`

**Interfaces:**
- Consumes: source image and `ImagePrediction`.
- Produces: a result image with BGR green/red/gray OBBs and readable labels; existing CLI visualization delegates to it.

- [ ] **Step 1: Write pixel-level rendering tests**

Create a synthetic white image and fixed square OBBs. Verify sampled border pixels equal:

```python
OK_COLOR = (0, 180, 0)
NG_COLOR = (0, 0, 255)
UNSUPPORTED_COLOR = (128, 128, 128)
```

Also assert the formatted labels:

```text
label3 · OK
label5 · NG · 蓝色
label2 · 暂不支持分类
label3 · 分类失败
```

- [ ] **Step 2: Run rendering tests and verify failure**

Run: `.venv/bin/python -m unittest tests.web.test_rendering -v`

Expected: FAIL because rendering module is missing.

- [ ] **Step 3: Implement deterministic rendering**

Use OpenCV anti-aliased polygons. Since OpenCV's built-in font cannot render Chinese reliably, draw the OBB with OpenCV and render label text through Pillow using a configurable CJK font path. If the configured font is unavailable, use ASCII fallback labels such as `classification unavailable`; never render broken tofu glyphs.

- [ ] **Step 4: Run rendering and legacy tests**

Run:

```bash
.venv/bin/python -m unittest tests.web.test_rendering -v
.venv/bin/python -m unittest tests.test_pipeline_predict -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit result rendering**

```bash
git add terminal_web/inference/rendering.py yolo11_obb/pipeline_predict.py tests/web/test_rendering.py
git commit -m "feat: render classified terminal regions"
```

### Task 8: Implement the independent inference Worker and readiness heartbeat

**Files:**
- Create: `terminal_web/readiness.py`
- Create: `terminal_web/worker.py`
- Create: `scripts/run_terminal_worker.py`
- Test: `tests/web/test_worker.py`

**Interfaces:**
- Consumes: queue functions, repository, storage, `LoadedInferencePipeline`.
- Produces: `InspectionWorker.run_once() -> bool`, `InspectionWorker.run_forever()`, worker/model health persisted for `/health`.

- [ ] **Step 1: Write Worker orchestration tests**

Use fakes and implement these concrete assertions:

- `test_worker_updates_stage_and_progress_for_each_image`: record the fake pipeline callback values, assert the four stages from Task 6, and assert `completed_images == 1`.
- `test_one_corrupt_image_does_not_abort_remaining_images`: make the first fake prediction raise `InvalidImageError`, make the second succeed, and assert both images were attempted.
- `test_mixed_outcomes_finish_partial_failed`: assert `succeeded_images == 1`, `failed_images == 1`, and task status `partial_failed`.
- `test_model_load_failure_marks_readiness_false_and_claims_nothing`: assert readiness is false and the queue fake records zero claim calls.
- `test_worker_heartbeat_is_refreshed_while_processing`: inject a clock, advance it during fake prediction, and assert `heartbeat_at` increases.

- [ ] **Step 2: Run Worker tests and verify failure**

Run: `.venv/bin/python -m unittest tests.web.test_worker -v`

Expected: FAIL because Worker is missing.

- [ ] **Step 3: Implement model registration and readiness**

Compute SHA-256 for all configured weights at startup, validate detector names and classifier class lists, and upsert `model_registry` entries. Readiness must expose the three model names, hashes, and load status but never absolute paths.

- [ ] **Step 4: Implement `run_once` and `run_forever`**

`run_once` recovers stale tasks, claims one queued task, processes its queued images in `sequence_no` order, commits after every stage transition and image result, and recomputes final task status. `run_forever` sleeps for one second only when no work is found and handles SIGTERM by finishing the current database transaction before exit.

- [ ] **Step 5: Run Worker tests**

Run: `.venv/bin/python -m unittest tests.web.test_worker -v`

Expected: all tests pass.

- [ ] **Step 6: Commit the GPU Worker**

```bash
git add terminal_web/readiness.py terminal_web/worker.py scripts/run_terminal_worker.py tests/web/test_worker.py
git commit -m "feat: add persistent terminal inference worker"
```

### Task 9: Smoke-test all three real local checkpoints

**Files:**
- Create: `scripts/smoke_terminal_pipeline.py`
- Test: `tests/web/test_smoke_script.py`
- Create runtime output only: `runs/web-smoke/` (ignored)

**Interfaces:**
- Consumes: the exact three downloaded checkpoints and one image from `datasets/obb_thin_thick/images/test`.
- Produces: JSON summary and rendered result image proving local model compatibility.

- [ ] **Step 1: Write CLI argument and missing-weight tests**

Test that the script exits non-zero and names the missing model when any one checkpoint path is absent. Do not load real models in the unit test.

- [ ] **Step 2: Implement the smoke script**

Arguments:

```text
--detector
--label3
--label5
--image
--output
--det-device
--cls-device
```

Print timings for model load, detection, classification, and render. Write `result.json` containing model hashes, detections, classifications, warnings, and overall result.

- [ ] **Step 3: Run the unit test**

Run: `.venv/bin/python -m unittest tests.web.test_smoke_script -v`

Expected: all tests pass.

- [ ] **Step 4: Run real CPU smoke inference locally**

```bash
MPLCONFIGDIR=/private/tmp/matplotlib-cache .venv/bin/python scripts/smoke_terminal_pipeline.py \
  --detector weights/detector/yolo11l_obb_best.pt \
  --label3 weights/classifiers/label3/resnet18_best.pt \
  --label5 weights/classifiers/label5/resnet18_best.pt \
  --image datasets/obb_thin_thick/images/test/CropImage_20260126092714769_F3-I0_OK-5.bmp \
  --output runs/web-smoke \
  --det-device cpu \
  --cls-device cpu
```

Expected: exit 0, `result.json`, and one result image. Inspect the rendered output for correct OBB placement and green/red/gray semantics.

- [ ] **Step 5: Commit the smoke utility**

```bash
git add scripts/smoke_terminal_pipeline.py tests/web/test_smoke_script.py
git commit -m "test: add real terminal pipeline smoke check"
```

## Milestone 3: Approved React Interface

### Task 10: Initialize the Product Design frontend and API contract

**Files:**
- Create: `web_frontend/` from the Product Design desktop web-app template selected by `product-design:image-to-code`
- Create: `web_frontend/src/api/types.ts`
- Create: `web_frontend/src/api/client.ts`
- Create: `web_frontend/src/styles/tokens.css`
- Create: `web_frontend/src/styles/global.css`
- Test: `web_frontend/src/api/client.test.ts`

**Interfaces:**
- Consumes: selected fusion mock and FastAPI schemas from Task 5.
- Produces: typed `apiClient`, design tokens, application shell, and package scripts `dev`, `build`, `test`.

- [ ] **Step 1: Invoke the selected visual implementation workflow**

Read and follow `product-design:image-to-code` before initializing the frontend. Use the most recent approved fusion image as the sole visual target. Do not initialize a Sites starter and do not use a generic admin-dashboard template.

- [ ] **Step 2: Define exact TypeScript API types**

```typescript
export type TaskStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "partial_failed"
  | "failed";

export type ImageStage =
  | "pending"
  | "object_detection"
  | "anomaly_classification"
  | "rendering"
  | "complete";

export interface TaskSummary {
  id: string;
  displayId: string;
  status: TaskStatus;
  currentStage: ImageStage;
  totalImages: number;
  completedImages: number;
  succeededImages: number;
  failedImages: number;
  detectorModel: string;
  createdAt: string;
}
```

Mirror every backend field rather than inventing frontend-only variants.

- [ ] **Step 3: Write API client tests**

Mock `fetch` and prove:

- `createTask(files)` sends multipart without manually setting a boundary.
- non-2xx JSON error messages are surfaced.
- `listTasks("failed")` preserves the filter.
- abort signals cancel stale polling requests.

- [ ] **Step 4: Implement the API client**

Provide:

```typescript
getHealth(signal?)
createTask(files, metadata, signal?)
listTasks(params, signal?)
getTask(taskId, signal?)
getTaskImages(taskId, signal?)
getImage(imageId, signal?)
retryImage(imageId, signal?)
```

- [ ] **Step 5: Run frontend unit tests and build**

Run:

```bash
cd web_frontend
npm test -- --run
npm run build
```

Expected: API tests pass and production build exits 0.

- [ ] **Step 6: Commit the frontend foundation**

```bash
git add web_frontend
git commit -m "feat: initialize terminal inspection frontend"
```

### Task 11: Build upload, live progress, and before/after comparison

**Files:**
- Create: `web_frontend/src/components/Sidebar.tsx`
- Create: `web_frontend/src/components/UploadPanel.tsx`
- Create: `web_frontend/src/components/ProgressPanel.tsx`
- Create: `web_frontend/src/components/ImageComparison.tsx`
- Create: `web_frontend/src/hooks/useActiveTask.ts`
- Create: `web_frontend/src/pages/TasksPage.tsx`
- Modify: `web_frontend/src/App.tsx`
- Test: `web_frontend/src/pages/TasksPage.test.tsx`

**Interfaces:**
- Consumes: `apiClient`, selected mock tokens, task/image types.
- Produces: the approved top three UI sections and one-second visible-page polling.

- [ ] **Step 1: Write user-flow tests**

Use Testing Library to prove:

- selecting 101 files disables start and shows `单次最多 100 张图片`.
- clearing/removing files updates selected count.
- model-not-ready health disables start and explains which model is unavailable.
- starting a valid batch renders the returned task ID and progress.
- polling stops when task becomes `succeeded`, `partial_failed`, or `failed`.
- `document.hidden` changes polling from one second to ten seconds.
- before/after controls expose useful accessible names.

- [ ] **Step 2: Run tests and verify failure**

Run: `cd web_frontend && npm test -- --run src/pages/TasksPage.test.tsx`

Expected: FAIL because components are missing.

- [ ] **Step 3: Implement the approved shell and upload panel**

Match the fusion mock: 176px left navigation, restrained blue action, no gradients, no nested cards, and 14-16px body text. Support drag/drop, picker, file removal, selected count, and upload progress. Use an icon library already included by the Product Design template; do not draw icons by hand.

- [ ] **Step 4: Implement active task polling and stage mapping**

Map backend stages exactly:

```typescript
const stageLabel = {
  pending: "等待处理",
  object_detection: "分区域目标检测",
  anomaly_classification: "异常分类",
  rendering: "结果生成",
  complete: "已完成",
};
```

- [ ] **Step 5: Implement synchronized image comparison**

Show original and result image side by side, filename/dimensions, previous/next controls, image-level status, and reserved color copy only when the API returns no color model. Never infer box coordinates in the browser; the result image is rendered by the Worker.

- [ ] **Step 6: Run page tests and build**

Run:

```bash
cd web_frontend
npm test -- --run src/pages/TasksPage.test.tsx
npm run build
```

Expected: tests and build pass.

- [ ] **Step 7: Commit the primary inspection flow**

```bash
git add web_frontend/src
git commit -m "feat: add terminal inspection task workspace"
```

### Task 12: Build task table, status filters, history, and retry

**Files:**
- Create: `web_frontend/src/components/TaskFilters.tsx`
- Create: `web_frontend/src/components/TaskTable.tsx`
- Create: `web_frontend/src/components/TaskDetails.tsx`
- Create: `web_frontend/src/pages/HistoryPage.tsx`
- Modify: `web_frontend/src/App.tsx`
- Test: `web_frontend/src/components/TaskTable.test.tsx`
- Test: `web_frontend/src/pages/HistoryPage.test.tsx`

**Interfaces:**
- Consumes: task/image API and selected image state.
- Produces: approved expandable task table, status tabs, persistent history lookup, and failed-image retry.

- [ ] **Step 1: Write table and history tests**

Prove:

- tabs show running/all/success/failed counts.
- `partial_failed` uses the failed visual treatment and appears under failed.
- expanding a row shows task name, storage identifier, creator placeholder, and note.
- selecting a historical task loads its images and opens original/result comparison.
- retry is shown only for failed images and refreshes task state after success.
- API errors preserve the current table and show a retryable inline message.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd web_frontend
npm test -- --run src/components/TaskTable.test.tsx src/pages/HistoryPage.test.tsx
```

Expected: FAIL because table/history components are missing.

- [ ] **Step 3: Implement compact status filters and expandable rows**

Follow the mock exactly: tabs above a single table surface, lightweight row separators, inline progress bars, blue running dot, green success dot, red failed/partial dot, and a details row directly beneath the selected task.

- [ ] **Step 4: Implement history routing and retry**

Routes:

```text
/tasks
/history
/history/:taskId
```

Keep the selected task/image in the URL so browser refresh restores the view.

- [ ] **Step 5: Run tests and production build**

Run:

```bash
cd web_frontend
npm test -- --run
npm run build
```

Expected: all frontend tests pass and build exits 0.

- [ ] **Step 6: Commit task history**

```bash
git add web_frontend/src
git commit -m "feat: add persistent inspection history interface"
```

## Milestone 4: Deployment and End-to-End Verification

### Task 13: Add PostgreSQL development stack and production process definitions

**Files:**
- Create: `compose.yaml`
- Create: `deploy/api.Dockerfile`
- Create: `deploy/frontend.Dockerfile`
- Create: `deploy/nginx.conf`
- Create: `deploy/terminal-api.service`
- Create: `deploy/terminal-worker.service`
- Create: `docs/web-deployment.md`

**Interfaces:**
- Consumes: API/Worker entrypoints, frontend production build, environment variables.
- Produces: repeatable local PostgreSQL/API/Worker stack and documented server deployment.

- [ ] **Step 1: Add Compose services**

Define PostgreSQL 16 with a named volume, API, Worker, and Nginx/front-end. Mount weights read-only and task storage read-write. Add health checks for PostgreSQL and API. Configure exactly one Worker replica.

- [ ] **Step 2: Add Nginx routes**

Serve frontend assets with SPA fallback, proxy `/api/` to FastAPI, and proxy artifact responses without exposing the storage root. Set upload size high enough for 100 configured images and document the matching API limit.

- [ ] **Step 3: Add systemd alternatives for the school server**

Both services use the same environment file. Worker starts after PostgreSQL and uses the RHINO/base Python environment selected during deployment. Do not place credentials in committed unit files.

- [ ] **Step 4: Document exact setup and start commands**

Include database creation, Alembic upgrade, environment configuration, model path verification, frontend build, API/Worker start, health check, log inspection, backup paths, and rollback steps.

- [ ] **Step 5: Validate Compose and Nginx syntax**

Run:

```bash
docker compose config
docker run --rm -v "$PWD/deploy/nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine nginx -t
```

Expected: both commands exit 0. If Docker is unavailable locally, record that as an environment gap and validate on the target server before deployment; do not claim deployment verification.

- [ ] **Step 6: Commit deployment assets**

```bash
git add compose.yaml deploy docs/web-deployment.md
git commit -m "ops: add terminal inspection deployment stack"
```

### Task 14: Run PostgreSQL integration, full tests, and visual QA

**Files:**
- Create: `tests/web/test_postgres_integration.py`
- Create: `tests/web/test_end_to_end.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete application, PostgreSQL service, fake inference for deterministic CI, real weights for smoke verification.
- Produces: evidence that queue locking, restart recovery, API flow, UI build, and visual target all work together.

- [ ] **Step 1: Write PostgreSQL queue concurrency test**

Open two independent sessions, call `claim_next_task` concurrently, and assert only one receives the queued task. Skip with a clear message unless `TEST_DATABASE_URL` is set; run it against Compose in the next step.

- [ ] **Step 2: Write deterministic end-to-end API/Worker test**

Create a two-image task through the API, run `InspectionWorker.run_once()` with a fake pipeline where one image succeeds and one fails, then assert:

```text
task.status == partial_failed
completed_images == 2
succeeded_images == 1
failed_images == 1
successful original/result files are retrievable
failed image exposes a sanitized error
```

- [ ] **Step 3: Run backend test suite**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all existing and new tests pass.

- [ ] **Step 4: Run PostgreSQL integration test**

Run:

```bash
docker compose up -d postgres
TEST_DATABASE_URL=postgresql+psycopg://terminal:terminal@localhost:5432/terminal_inspection \
  .venv/bin/python -m unittest tests.web.test_postgres_integration -v
```

Expected: queue concurrency test passes.

- [ ] **Step 5: Run frontend tests and build**

Run:

```bash
cd web_frontend
npm test -- --run
npm run build
```

Expected: all tests pass and build exits 0.

- [ ] **Step 6: Start the complete local application with fake inference**

Run migrations, start API and Worker, open the frontend, upload 1 image and then a 100-image fixture batch, and verify upload, two-stage progress, before/after comparison, task filters, expandable row, and history restoration after browser refresh.

- [ ] **Step 7: Perform Product Design visual comparison**

Use the Product Design `design-qa` workflow after the implementation is rendered. Compare the selected fusion mock and the app screenshot at the same 1440x1024 viewport in one comparison input. Fix spacing, typography, alignment, borders, color semantics, overflow, and responsive behavior until no material mismatch remains.

- [ ] **Step 8: Run the real three-model smoke check again**

Repeat Task 9 after the final integration. Confirm the API health response shows the same SHA-256 values already verified for all three weights and the rendered result can be opened from task history.

- [ ] **Step 9: Update README with development and production commands**

Document prerequisites, `.env` setup, migration, frontend development, API/Worker commands, Compose, model paths, test commands, and the limitation that color classification is not yet connected.

- [ ] **Step 10: Run final repository verification**

Run:

```bash
git diff --check
.venv/bin/python -m unittest discover -s tests -v
cd web_frontend && npm test -- --run && npm run build
```

Expected: no whitespace errors, all backend tests pass, all frontend tests pass, and build succeeds.

- [ ] **Step 11: Commit final verification and documentation**

```bash
git add tests/web README.md
git commit -m "test: verify terminal inspection web workflow"
```

## Completion Evidence

Do not declare the web system finished until all of the following are recorded in the final handoff:

- Backend test count and zero failures.
- Frontend test count and successful production build.
- PostgreSQL queue concurrency result.
- Real YOLO11l + label3 + label5 smoke inference output path and model hashes.
- API `/health` output with all three models ready.
- 1-image and 100-image task behavior.
- Worker restart recovery result.
- Selected visual target comparison screenshot and remaining known visual differences, if any.
- Exact server deployment commands and persistent storage/database backup locations.
