from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from terminal_web.api.dependencies import (
    CurrentUser,
    get_app_settings,
    get_session,
)
from terminal_web.auth import (
    DUMMY_PASSWORD_HASH,
    SESSION_COOKIE_NAME,
    create_session_token,
    normalize_username,
    session_token_digest,
    verify_password,
)
from terminal_web.auth_repository import AuthRepository
from terminal_web.schemas import CurrentUserResponse, LoginRequest


router = APIRouter(prefix="/auth", tags=["auth"])


def _invalid_credentials() -> HTTPException:
    return HTTPException(status_code=401, detail="用户名或密码错误")


@router.post("/login", response_model=CurrentUserResponse)
def login(
    payload: LoginRequest,
    response: Response,
    session: Annotated[Session, Depends(get_session)],
    settings=Depends(get_app_settings),
) -> CurrentUserResponse:
    try:
        username = normalize_username(payload.username)
    except ValueError:
        username = None

    repository = AuthRepository(session)
    user = repository.get_user_by_username(username) if username is not None else None
    password_hash = user.password_hash if user is not None else DUMMY_PASSWORD_HASH
    password_valid = verify_password(password_hash, payload.password)
    if user is None or not user.is_active or not password_valid:
        raise _invalid_credentials()

    raw_token, token_hash = create_session_token()
    expires_at = datetime.now(UTC) + timedelta(hours=settings.session_ttl_hours)
    repository.create_session(user.id, token_hash, expires_at)
    session.commit()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        raw_token,
        max_age=settings.session_ttl_hours * 3600,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="strict",
        path="/",
    )
    return CurrentUserResponse(id=user.id, username=user.username)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    settings=Depends(get_app_settings),
) -> Response:
    raw_token = request.cookies.get(SESSION_COOKIE_NAME)
    if raw_token:
        AuthRepository(session).delete_session(session_token_digest(raw_token))
        session.commit()
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        secure=settings.session_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return response


@router.get("/me", response_model=CurrentUserResponse)
def current_user(user: CurrentUser) -> CurrentUserResponse:
    return CurrentUserResponse(id=user.id, username=user.username)
