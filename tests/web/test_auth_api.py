import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from terminal_web.api.app import create_app
from terminal_web.auth import hash_password, session_token_digest
from terminal_web.database import Base
from terminal_web.models import User, UserSession
from terminal_web.schemas import HealthResponse
from terminal_web.storage import ArtifactStorage


VALID_PASSWORD = "Correct-password-2026"


class ReadyServices:
    def snapshot(self):
        return HealthResponse(
            api_ready=True,
            database_ready=True,
            worker_ready=True,
            models_ready=True,
        )


class AuthenticationApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        settings = SimpleNamespace(
            max_images_per_task=100,
            session_ttl_hours=12,
            session_cookie_secure=False,
        )
        app = create_app(
            settings=settings,
            session_factory=self.session_factory,
            storage=ArtifactStorage(Path(self.temp.name) / "storage"),
            readiness=ReadyServices(),
        )
        self.client = TestClient(app)
        with self.session_factory() as session:
            session.add(
                User(username="Admin", password_hash=hash_password(VALID_PASSWORD))
            )
            session.commit()

    def tearDown(self):
        self.client.close()
        self.engine.dispose()
        self.temp.cleanup()

    @staticmethod
    def credentials(username="Admin", password=VALID_PASSWORD):
        return {"username": username, "password": password}

    def login(self, **overrides):
        credentials = self.credentials(**overrides)
        return self.client.post(
            "/api/v1/auth/login",
            json=credentials,
            headers={"Origin": "http://testserver"},
        )

    def test_health_is_public_but_business_routes_require_login(self):
        self.assertEqual(self.client.get("/api/v1/health").status_code, 200)
        self.assertEqual(self.client.get("/api/v1/tasks").status_code, 401)

    def test_login_is_case_sensitive_and_sets_database_backed_cookie(self):
        wrong_case = self.login(username="admin")
        response = self.login(username="Admin")

        self.assertEqual(wrong_case.status_code, 401)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["username"], "Admin")
        cookie_header = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie_header)
        self.assertIn("SameSite=strict", cookie_header)
        self.assertIn("Max-Age=43200", cookie_header)
        raw_token = self.client.cookies.get("terminal_session")
        with self.session_factory() as session:
            stored = session.scalar(select(UserSession))
            self.assertIsNotNone(stored)
            self.assertNotEqual(stored.token_hash, raw_token)
            self.assertEqual(stored.token_hash, session_token_digest(raw_token))
            remaining = stored.expires_at.replace(tzinfo=UTC) - datetime.now(UTC)
            self.assertGreater(remaining, timedelta(hours=11, minutes=59))
            self.assertLessEqual(remaining, timedelta(hours=12))

    def test_login_failures_do_not_reveal_account_state(self):
        with self.session_factory() as session:
            session.add(
                User(
                    username="disabled",
                    password_hash=hash_password(VALID_PASSWORD),
                    is_active=False,
                )
            )
            session.commit()

        responses = (
            self.login(username="missing"),
            self.login(password="Wrong-password-2026"),
            self.login(username="disabled"),
            self.login(username="bad name"),
        )

        self.assertTrue(all(response.status_code == 401 for response in responses))
        self.assertEqual(
            {response.json()["detail"] for response in responses},
            {"用户名或密码错误"},
        )

    def test_current_user_rejects_expired_and_disabled_sessions(self):
        expired_raw = "expired-session-token"
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == "Admin"))
            session.add(
                UserSession(
                    user_id=user.id,
                    token_hash=session_token_digest(expired_raw),
                    expires_at=datetime.now(UTC) - timedelta(seconds=1),
                )
            )
            session.commit()
        self.client.cookies.set("terminal_session", expired_raw)

        expired = self.client.get("/api/v1/auth/me")

        self.assertEqual(expired.status_code, 401)
        with self.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(UserSession)),
                0,
            )

        self.assertEqual(self.login().status_code, 200)
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == "Admin"))
            user.is_active = False
            session.commit()

        disabled = self.client.get("/api/v1/auth/me")
        self.assertEqual(disabled.status_code, 401)

    def test_logout_is_public_idempotent_and_revokes_current_session(self):
        self.assertEqual(self.login().status_code, 200)

        first = self.client.post(
            "/api/v1/auth/logout",
            headers={"Origin": "http://testserver"},
        )
        second = self.client.post(
            "/api/v1/auth/logout",
            headers={"Origin": "http://testserver"},
        )

        self.assertEqual(first.status_code, 204)
        self.assertEqual(second.status_code, 204)
        with self.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(UserSession)),
                0,
            )

    def test_unsafe_requests_require_same_origin(self):
        missing = self.client.post(
            "/api/v1/auth/login",
            json=self.credentials(),
        )
        foreign = self.client.post(
            "/api/v1/auth/login",
            json=self.credentials(),
            headers={"Origin": "https://evil.example"},
        )

        self.assertEqual(missing.status_code, 403)
        self.assertEqual(foreign.status_code, 403)

    def test_forwarded_origin_matches_public_nginx_host(self):
        response = self.client.post(
            "/api/v1/auth/login",
            json=self.credentials(),
            headers={
                "Origin": "http://factory.local:8080",
                "Host": "factory.local:8080",
                "X-Forwarded-Proto": "http",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)


if __name__ == "__main__":
    unittest.main()
