import logging
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from terminal_web.api import auth, feedback, health, images, tasks
from terminal_web.config import get_settings
from terminal_web.database import build_session_factory
from terminal_web.readiness import ReadinessStore
from terminal_web.storage import ArtifactStorage


logger = logging.getLogger(__name__)


def _effective_port(scheme: str, port: int | None) -> int | None:
    if port is not None:
        return port
    return 443 if scheme == "https" else 80 if scheme == "http" else None


def _same_origin(request: Request) -> bool:
    origin_header = request.headers.get("origin")
    host_header = request.headers.get("host")
    if not origin_header or not host_header:
        return False
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = (
        forwarded_proto.split(",", 1)[0].strip()
        if forwarded_proto
        else request.url.scheme
    ).lower()
    try:
        origin = urlsplit(origin_header)
        expected = urlsplit(f"{scheme}://{host_header}")
        if origin.username or origin.password or origin.query or origin.fragment:
            return False
        if origin.path not in {"", "/"}:
            return False
        return (
            origin.scheme.lower(),
            (origin.hostname or "").lower(),
            _effective_port(origin.scheme.lower(), origin.port),
        ) == (
            expected.scheme.lower(),
            (expected.hostname or "").lower(),
            _effective_port(expected.scheme.lower(), expected.port),
        )
    except ValueError:
        return False


def create_app(
    *,
    settings=None,
    session_factory=None,
    storage: ArtifactStorage | None = None,
    readiness=None,
) -> FastAPI:
    if settings is None:
        settings = get_settings()
    if session_factory is None:
        _, session_factory = build_session_factory(settings.database_url)
    if storage is None:
        storage = ArtifactStorage(settings.storage_root)
    if readiness is None:
        readiness = ReadinessStore(
            settings.storage_root / "worker-readiness.json",
            timeout_seconds=settings.worker_heartbeat_timeout_seconds,
        )

    app = FastAPI(title="端子分区域检测与异常分类", version="1.0.0")
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.storage = storage
    app.state.readiness = readiness

    @app.middleware("http")
    async def enforce_same_origin(request: Request, call_next):
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and not _same_origin(
            request
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "invalid request origin"},
            )
        return await call_next(request)

    @app.exception_handler(SQLAlchemyError)
    async def database_error_handler(
        request: Request, exc: SQLAlchemyError
    ) -> JSONResponse:
        logger.exception(
            "database request failed path=%s",
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=503,
            content={"detail": "database is unavailable"},
        )

    app.include_router(health.router, prefix="/api/v1")
    app.include_router(auth.router, prefix="/api/v1")
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(images.router, prefix="/api/v1")
    app.include_router(feedback.router, prefix="/api/v1")
    return app
