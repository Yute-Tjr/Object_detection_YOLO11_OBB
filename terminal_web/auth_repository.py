from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from terminal_web.models import User, UserSession


class AuthRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_user_by_username(self, username: str) -> User | None:
        return self.session.scalar(select(User).where(User.username == username))

    def get_user(self, user_id: uuid.UUID) -> User | None:
        return self.session.get(User, user_id)

    def list_users(self) -> Sequence[User]:
        return tuple(self.session.scalars(select(User).order_by(User.username)))

    def create_user(self, username: str, password_hash: str) -> User:
        user = User(username=username, password_hash=password_hash)
        self.session.add(user)
        self.session.flush()
        return user

    def update_password(self, user: User, password_hash: str) -> None:
        user.password_hash = password_hash
        self.session.flush()

    def set_active(self, user: User, is_active: bool) -> None:
        user.is_active = is_active
        self.session.flush()

    def create_session(
        self,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime,
    ) -> UserSession:
        user_session = UserSession(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self.session.add(user_session)
        self.session.flush()
        return user_session

    def get_user_for_session(self, token_hash: str, now: datetime) -> User | None:
        statement = (
            select(User)
            .join(UserSession, UserSession.user_id == User.id)
            .where(
                UserSession.token_hash == token_hash,
                UserSession.expires_at > now,
            )
        )
        return self.session.scalar(statement)

    def delete_session(self, token_hash: str) -> None:
        self.session.execute(
            delete(UserSession).where(UserSession.token_hash == token_hash)
        )

    def delete_sessions_for_user(self, user_id: uuid.UUID) -> None:
        self.session.execute(
            delete(UserSession).where(UserSession.user_id == user_id)
        )

    def delete_expired_sessions(self, now: datetime) -> None:
        self.session.execute(
            delete(UserSession).where(UserSession.expires_at <= now)
        )
