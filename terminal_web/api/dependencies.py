from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Annotated, Protocol

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from terminal_web.auth import SESSION_COOKIE_NAME, session_token_digest
from terminal_web.auth_repository import AuthRepository
from terminal_web.models import User
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


def get_current_user(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> User:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw_token:
        raise HTTPException(status_code=401, detail="authentication required")

    now = datetime.now(UTC)
    repository = AuthRepository(session)
    user = repository.get_user_for_session(session_token_digest(raw_token), now)
    if user is None or not user.is_active:
        repository.delete_expired_sessions(now)
        session.commit()
        raise HTTPException(status_code=401, detail="authentication required")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
