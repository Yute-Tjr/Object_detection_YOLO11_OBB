import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from terminal_web.api import health, images, tasks
from terminal_web.config import get_settings
from terminal_web.database import build_session_factory
from terminal_web.readiness import ReadinessStore
from terminal_web.storage import ArtifactStorage


logger = logging.getLogger(__name__)


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
    app.include_router(tasks.router, prefix="/api/v1")
    app.include_router(images.router, prefix="/api/v1")
    return app
