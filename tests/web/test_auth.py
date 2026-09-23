import unittest
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from terminal_web.auth import (
    PASSWORD_MIN_LENGTH,
    create_session_token,
    hash_password,
    normalize_username,
    session_token_digest,
    verify_password,
)
from terminal_web.auth_repository import AuthRepository
from terminal_web.database import Base
from terminal_web.models import User, UserSession


class AuthenticationPrimitiveTest(unittest.TestCase):
    def test_username_trims_edges_but_preserves_case(self):
        self.assertEqual(normalize_username(" Admin "), "Admin")
        self.assertEqual(normalize_username("admin"), "admin")

    def test_username_rejects_empty_whitespace_control_and_long_values(self):
        for value in ("", "  ", "admin user", "admin\tuser", "admin\nuser", "a" * 65):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    normalize_username(value)

    def test_password_hash_is_argon2id_and_never_returns_plaintext(self):
        password = "correct horse battery staple"
        encoded = hash_password(password)

        self.assertTrue(encoded.startswith("$argon2id$"))
        self.assertNotEqual(encoded, password)
        self.assertTrue(verify_password(encoded, password))
        self.assertFalse(verify_password(encoded, "wrong password"))

    def test_password_hash_rejects_short_password(self):
        self.assertEqual(PASSWORD_MIN_LENGTH, 6)
        with self.assertRaisesRegex(ValueError, "6"):
            hash_password("12345")
        self.assertTrue(hash_password("123456").startswith("$argon2id$"))

    def test_malformed_password_hash_fails_closed(self):
        self.assertFalse(verify_password("not-a-valid-hash", "irrelevant password"))

    def test_session_token_digest_is_deterministic_but_not_the_raw_token(self):
        raw_token, stored_digest = create_session_token()

        self.assertGreaterEqual(len(raw_token), 48)
        self.assertEqual(len(stored_digest), 64)
        self.assertNotEqual(raw_token, stored_digest)
        self.assertEqual(session_token_digest(raw_token), stored_digest)


class AuthRepositoryTest(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite+pysqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()

    def test_user_lookup_is_case_sensitive(self):
        with self.session_factory() as session:
            repository = AuthRepository(session)
            upper = repository.create_user("Admin", hash_password("Password-Admin-2026"))
            lower = repository.create_user("admin", hash_password("Password-admin-2026"))
            session.commit()

            self.assertNotEqual(upper.id, lower.id)
            self.assertEqual(repository.get_user_by_username("Admin").id, upper.id)
            self.assertEqual(repository.get_user_by_username("admin").id, lower.id)
            self.assertIsNone(repository.get_user_by_username("ADMIN"))

    def test_session_lookup_obeys_expiry_and_deletion(self):
        now = datetime.now(UTC)
        with self.session_factory() as session:
            repository = AuthRepository(session)
            user = repository.create_user(
                "operator",
                hash_password("Password-operator-2026"),
            )
            active = repository.create_session(
                user.id,
                "a" * 64,
                now + timedelta(hours=1),
            )
            repository.create_session(
                user.id,
                "b" * 64,
                now - timedelta(seconds=1),
            )
            session.commit()

            self.assertEqual(repository.get_user_for_session("a" * 64, now).id, user.id)
            self.assertIsNone(repository.get_user_for_session("b" * 64, now))
            repository.delete_session(active.token_hash)
            session.commit()
            self.assertIsNone(repository.get_user_for_session("a" * 64, now))

    def test_disabling_user_can_remove_every_session(self):
        with self.session_factory() as session:
            repository = AuthRepository(session)
            user = repository.create_user(
                "operator",
                hash_password("Password-operator-2026"),
            )
            repository.create_session(
                user.id,
                "c" * 64,
                datetime.now(UTC) + timedelta(hours=1),
            )
            repository.delete_sessions_for_user(user.id)
            session.commit()

            count = session.scalar(select(func.count()).select_from(UserSession))
            self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
