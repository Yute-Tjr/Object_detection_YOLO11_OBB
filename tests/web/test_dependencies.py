import asyncio
import unittest
from types import SimpleNamespace

from terminal_web.api.dependencies import get_session


class FakeSession:
    def __init__(self):
        self.rollbacks = 0
        self.closes = 0
        self.active = True

    def rollback(self):
        self.rollbacks += 1
        self.active = False

    def close(self):
        self.closes += 1

    def in_transaction(self):
        return self.active


async def consume_dependency(generator, *, fail=False):
    session = await anext(generator)
    if fail:
        try:
            await generator.athrow(RuntimeError("boom"))
        except RuntimeError:
            pass
    else:
        await generator.aclose()
    return session


class SessionDependencyTest(unittest.TestCase):
    @staticmethod
    def request_for(session):
        return SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(session_factory=lambda: session)
            )
        )

    def test_session_dependency_rolls_back_and_closes_after_response(self):
        fake = FakeSession()

        session = asyncio.run(
            consume_dependency(get_session(self.request_for(fake)))
        )

        self.assertIs(session, fake)
        self.assertEqual(fake.rollbacks, 1)
        self.assertEqual(fake.closes, 1)

    def test_session_dependency_rolls_back_and_closes_after_exception(self):
        fake = FakeSession()

        asyncio.run(
            consume_dependency(get_session(self.request_for(fake)), fail=True)
        )

        self.assertEqual(fake.rollbacks, 1)
        self.assertEqual(fake.closes, 1)
