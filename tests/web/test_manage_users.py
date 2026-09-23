import io
import unittest
from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from scripts.manage_users import main
from terminal_web.auth import hash_password, verify_password
from terminal_web.auth_repository import AuthRepository
from terminal_web.database import Base
from terminal_web.models import User, UserSession


class PasswordReader:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self, prompt):
        return next(self.values)


class ManageUsersTest(unittest.TestCase):
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

    def run_command(self, argv, passwords=()):
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = main(
            argv,
            session_factory=self.session_factory,
            password_reader=PasswordReader(passwords),
            stdout=stdout,
            stderr=stderr,
        )
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_add_preserves_case_and_list_never_outputs_password_hash(self):
        first_password = "Password-Admin-2026"
        second_password = "Password-admin-2026"
        self.assertEqual(
            self.run_command(["add", "Admin"], [first_password, first_password])[0],
            0,
        )
        self.assertEqual(
            self.run_command(["add", "admin"], [second_password, second_password])[0],
            0,
        )

        exit_code, output, error = self.run_command(["list"])

        self.assertEqual(exit_code, 0, error)
        self.assertIn("Admin", output)
        self.assertIn("admin", output)
        self.assertNotIn("$argon2", output)
        with self.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(User)),
                2,
            )

    def test_duplicate_exact_username_is_rejected_without_overwrite(self):
        password = "Password-Admin-2026"
        self.assertEqual(
            self.run_command(["add", "Admin"], [password, password])[0],
            0,
        )

        exit_code, _, error = self.run_command(
            ["add", "Admin"],
            ["Different-pass-2026", "Different-pass-2026"],
        )

        self.assertNotEqual(exit_code, 0)
        self.assertIn("已存在", error)
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == "Admin"))
            self.assertTrue(verify_password(user.password_hash, password))

    def test_add_rejects_short_and_mismatched_passwords(self):
        short_code, _, short_error = self.run_command(
            ["add", "short-user"],
            ["short", "short"],
        )
        mismatch_code, _, mismatch_error = self.run_command(
            ["add", "mismatch-user"],
            ["Password-one-2026", "Password-two-2026"],
        )

        self.assertNotEqual(short_code, 0)
        self.assertIn("12", short_error)
        self.assertNotEqual(mismatch_code, 0)
        self.assertIn("不一致", mismatch_error)
        with self.session_factory() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(User)),
                0,
            )

    def test_reset_password_replaces_hash_and_revokes_sessions(self):
        with self.session_factory() as session:
            repository = AuthRepository(session)
            user = repository.create_user(
                "operator",
                hash_password("Old-password-2026"),
            )
            repository.create_session(
                user.id,
                "d" * 64,
                datetime.now(UTC) + timedelta(hours=1),
            )
            session.commit()

        new_password = "New-password-2026"
        exit_code, _, error = self.run_command(
            ["reset-password", "operator"],
            [new_password, new_password],
        )

        self.assertEqual(exit_code, 0, error)
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == "operator"))
            self.assertTrue(verify_password(user.password_hash, new_password))
            self.assertEqual(
                session.scalar(select(func.count()).select_from(UserSession)),
                0,
            )

    def test_disable_revokes_sessions_and_enable_restores_account(self):
        with self.session_factory() as session:
            repository = AuthRepository(session)
            user = repository.create_user(
                "operator",
                hash_password("Password-operator-2026"),
            )
            repository.create_session(
                user.id,
                "e" * 64,
                datetime.now(UTC) + timedelta(hours=1),
            )
            session.commit()

        self.assertEqual(self.run_command(["disable", "operator"])[0], 0)
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == "operator"))
            self.assertFalse(user.is_active)
            self.assertEqual(
                session.scalar(select(func.count()).select_from(UserSession)),
                0,
            )

        self.assertEqual(self.run_command(["enable", "operator"])[0], 0)
        with self.session_factory() as session:
            user = session.scalar(select(User).where(User.username == "operator"))
            self.assertTrue(user.is_active)


if __name__ == "__main__":
    unittest.main()
