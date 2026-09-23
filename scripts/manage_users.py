#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

from sqlalchemy.exc import IntegrityError


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from terminal_web.auth import PASSWORD_MIN_LENGTH, hash_password, normalize_username
from terminal_web.auth_repository import AuthRepository
from terminal_web.config import get_settings
from terminal_web.database import SessionFactory, build_session_factory


PasswordReader = Callable[[str], str]


class UserCommandError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理端子检测系统用户")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("add", "reset-password", "disable", "enable"):
        command = subcommands.add_parser(name)
        command.add_argument("username")
    subcommands.add_parser("list")
    return parser


def _read_confirmed_password(password_reader: PasswordReader) -> str:
    password = password_reader("请输入密码：")
    confirmation = password_reader("请再次输入密码：")
    if password != confirmation:
        raise UserCommandError("两次输入的密码不一致")
    if len(password) < PASSWORD_MIN_LENGTH:
        raise UserCommandError(f"密码长度不能少于 {PASSWORD_MIN_LENGTH} 个字符")
    return password


def _require_user(repository: AuthRepository, username: str):
    user = repository.get_user_by_username(username)
    if user is None:
        raise UserCommandError(f"用户不存在：{username}")
    return user


def _execute(
    args: argparse.Namespace,
    session_factory: SessionFactory,
    password_reader: PasswordReader,
    stdout: TextIO,
) -> None:
    with session_factory() as session:
        repository = AuthRepository(session)
        if args.command == "list":
            users = repository.list_users()
            if not users:
                print("暂无用户", file=stdout)
                return
            print("用户名\t状态\t创建时间", file=stdout)
            for user in users:
                state = "启用" if user.is_active else "禁用"
                print(
                    f"{user.username}\t{state}\t{user.created_at.isoformat()}",
                    file=stdout,
                )
            return

        username = normalize_username(args.username)
        if args.command == "add":
            if repository.get_user_by_username(username) is not None:
                raise UserCommandError(f"用户已存在：{username}")
            password = _read_confirmed_password(password_reader)
            repository.create_user(username, hash_password(password))
            success_message = f"用户已创建：{username}"
        elif args.command == "reset-password":
            user = _require_user(repository, username)
            password = _read_confirmed_password(password_reader)
            repository.update_password(user, hash_password(password))
            repository.delete_sessions_for_user(user.id)
            success_message = f"密码已重置，现有会话已退出：{username}"
        elif args.command == "disable":
            user = _require_user(repository, username)
            repository.set_active(user, False)
            repository.delete_sessions_for_user(user.id)
            success_message = f"用户已禁用，现有会话已退出：{username}"
        elif args.command == "enable":
            user = _require_user(repository, username)
            repository.set_active(user, True)
            success_message = f"用户已启用：{username}"
        else:
            raise UserCommandError(f"不支持的命令：{args.command}")

        session.commit()
        print(success_message, file=stdout)


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: SessionFactory | None = None,
    password_reader: PasswordReader = getpass.getpass,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    args = build_parser().parse_args(argv)
    engine = None
    if session_factory is None:
        engine, session_factory = build_session_factory(get_settings().database_url)
    try:
        _execute(args, session_factory, password_reader, stdout)
        return 0
    except (IntegrityError, UserCommandError, ValueError) as exc:
        print(f"错误：{exc}", file=stderr)
        return 2
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
