"""Per-user auth tests. Runs the REAL Motor store code against an in-memory
mongomock database (no mongod needed), driven through the FastAPI app.

    python -m pytest tests/test_auth.py -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import auth                    # noqa: E402
from app.authstore import UserStore     # noqa: E402


# ---- unit: password + token handling ----

def test_password_hash_is_bcrypt_and_verifies():
    h = auth.hash_password("correcthorse")
    assert h != "correcthorse"              # never plaintext
    assert h.startswith("$2")               # bcrypt marker
    assert auth.verify_password("correcthorse", h)
    assert not auth.verify_password("wrong", h)


def test_short_password_rejected():
    with pytest.raises(ValueError):
        auth.validate_password("short")


def test_token_is_stored_only_as_hash():
    token = auth.new_session_token()
    assert auth.hash_token(token) != token
    assert len(auth.hash_token(token)) == 64   # sha256 hex


# ---- integration: full app with a mongomock-backed store ----

@pytest.fixture()
def client(monkeypatch):
    """A TestClient whose auth store is an in-memory mongomock Mongo."""
    from mongomock_motor import AsyncMongoMockClient
    monkeypatch.setenv("EARSHOT_MONGO_URI", "mongodb://mock")

    async def fake_build():
        db = AsyncMongoMockClient()["earshot_test"]
        store = UserStore(db)
        await store.ensure_indexes()
        return store, None

    import app.main as main
    monkeypatch.setattr(main, "_build_user_store", fake_build)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as test_client:
        yield test_client


def test_healthz_reports_auth_enabled(client):
    assert client.get("/healthz").json()["auth"] is True


def test_register_login_logout_flow(client):
    # register auto-logs-in and sets a session cookie
    r = client.post("/auth/register",
                    json={"username": "kai", "password": "hunter2pass"})
    assert r.status_code == 200
    assert r.json()["username"] == "kai"
    assert "password_hash" not in r.json()          # never leaked
    assert client.get("/auth/me").json()["username"] == "kai"

    client.post("/auth/logout")
    assert client.get("/auth/me").status_code == 401  # session ended

    # log back in
    assert client.post("/auth/login",
                       json={"username": "kai",
                             "password": "hunter2pass"}).status_code == 200
    assert client.get("/auth/me").json()["username"] == "kai"


def test_duplicate_username_rejected(client):
    client.post("/auth/register",
                json={"username": "dup", "password": "password123"})
    r = client.post("/auth/register",
                    json={"username": "dup", "password": "password123"})
    assert r.status_code == 422


def test_wrong_password_is_401_and_generic(client):
    client.post("/auth/register",
                json={"username": "kai", "password": "hunter2pass"})
    client.post("/auth/logout")
    r = client.post("/auth/login",
                    json={"username": "kai", "password": "WRONG"})
    assert r.status_code == 401
    assert "invalid username or password" in r.json()["detail"]


def test_login_history_records_each_session(client):
    client.post("/auth/register",
                json={"username": "kai", "password": "hunter2pass"})
    client.post("/auth/logout")
    client.post("/auth/login",
                json={"username": "kai", "password": "hunter2pass"})
    sessions = client.get("/auth/sessions").json()
    assert len(sessions) == 2                        # register + login
    assert sessions[0]["logout_at"] is None          # current session open
    assert sessions[1]["logout_at"] is not None      # first was closed
    assert all("token_hash" not in s for s in sessions)  # never exposed


def test_per_user_rules_are_isolated(client):
    # user A mutes the doorbell
    client.post("/auth/register",
                json={"username": "alice", "password": "password123"})
    client.put("/me/rules/doorbell", json={"enabled": False})
    assert client.get("/me/rules").json() == {
        "doorbell": {"enabled": False, "urgency": None}}
    client.post("/auth/logout")

    # user B has their own, empty rule set
    client.post("/auth/register",
                json={"username": "bob", "password": "password123"})
    assert client.get("/me/rules").json() == {}
    client.put("/me/rules/knock", json={"enabled": True, "urgency": "high"})
    assert client.get("/me/rules").json() == {
        "knock": {"enabled": True, "urgency": "high"}}
    client.post("/auth/logout")

    # back to A — still just the doorbell rule
    client.post("/auth/login",
                json={"username": "alice", "password": "password123"})
    assert client.get("/me/rules").json() == {
        "doorbell": {"enabled": False, "urgency": None}}


def test_per_user_endpoints_require_login(client):
    assert client.get("/me/rules").status_code == 401
    assert client.get("/me/prefs").status_code == 401
    assert client.get("/auth/sessions").status_code == 401


def test_prefs_roundtrip_including_own_ntfy_topic(client):
    client.post("/auth/register",
                json={"username": "kai", "password": "hunter2pass"})
    client.put("/me/prefs", json={"ntfy_topic": "kai-earshot-7f3a9c",
                                  "shown_categories": ["urgent", "presence"]})
    assert client.get("/me/prefs").json() == {
        "ntfy_topic": "kai-earshot-7f3a9c",
        "shown_categories": ["urgent", "presence"]}


def test_bad_urgency_on_user_rule_is_422(client):
    client.post("/auth/register",
                json={"username": "kai", "password": "hunter2pass"})
    assert client.put("/me/rules/doorbell",
                      json={"enabled": True,
                            "urgency": "nuclear"}).status_code == 422


def test_alert_path_not_gated_by_auth(client):
    """The live event path must work without a login even when auth is on."""
    r = client.post("/debug/event",
                    json={"label": "smoke_alarm", "urgency": "high"})
    assert r.status_code == 200 and r.json()["accepted"] is True


def test_bad_mongo_uri_degrades_to_auth_off(monkeypatch):
    """An unreachable/misconfigured Mongo (e.g. wrong Atlas password or an
    un-allow-listed IP) must not crash the server: auth turns off and the
    live alert path keeps working."""
    from app import config
    monkeypatch.setattr(config, "MONGO_URI", "mongodb://127.0.0.1:1")
    monkeypatch.setattr(config, "MONGO_TIMEOUT_MS", 500)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        assert c.get("/healthz").json()["auth"] is False
        assert c.get("/auth/me").status_code == 503        # accounts disabled
        assert c.post("/debug/event",
                      json={"label": "knock", "urgency": "medium"}
                      ).json()["accepted"] is True           # alerts still fire
