from terminal_web.auth import hash_password
from terminal_web.models import User


TEST_PASSWORD = "Test-password-2026"


def create_and_login(client, session_factory, username="tester"):
    with session_factory() as session:
        session.add(User(username=username, password_hash=hash_password(TEST_PASSWORD)))
        session.commit()
    client.headers.update({"Origin": "http://testserver"})
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": TEST_PASSWORD},
    )
    if response.status_code != 200:
        raise AssertionError(f"test login failed: {response.status_code} {response.text}")
    return response.json()
