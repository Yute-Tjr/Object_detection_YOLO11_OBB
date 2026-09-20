from collections.abc import AsyncGenerator
from typing import Annotated, Protocol

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from terminal_web.repositories import TaskRepository
from terminal_web.schemas import HealthResponse
from terminal_web.storage import ArtifactStorage


class ReadinessProvider(Protocol):
    def snapshot(self) -> HealthResponse:
        ...


async def get_session(request: Request) -> AsyncGenerator[Session, None]:
    session = request.app.state.session_factory()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        if session.in_transaction():
            session.rollback()
        session.close()


def get_repository(
    session: Annotated[Session, Depends(get_session)],
) -> TaskRepository:
    return TaskRepository(session)


def get_storage(request: Request) -> ArtifactStorage:
    return request.app.state.storage


def get_readiness(request: Request) -> ReadinessProvider:
    return request.app.state.readiness


def get_app_settings(request: Request):
    return request.app.state.settings
