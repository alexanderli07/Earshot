"""MongoDB-backed per-user store: accounts, sessions, per-user rules + prefs.

Wraps a Motor async database. The database is injected, so tests run the exact
same code against an in-memory mongomock database (no mongod needed). Collections:

  users       { _id, username, password_hash, display_name, created_at }
  sessions    { _id, user_id, token_hash, login_at, logout_at, user_agent }
  user_rules  { _id, user_id, label, enabled, urgency }   (per-user overrides)
  user_prefs  { _id: user_id, ntfy_topic, shown_categories, ... }

Passwords are never stored here — only bcrypt hashes, produced in auth.py.
Session tokens are never stored either — only their SHA-256 hashes, so a DB
leak can't be replayed as a live session.
"""

import time
from uuid import uuid4


class UserStore:
    def __init__(self, db):
        self.db = db
        self.users = db["users"]
        self.sessions = db["sessions"]
        self.user_rules = db["user_rules"]
        self.user_prefs = db["user_prefs"]

    async def ensure_indexes(self):
        await self.users.create_index("username", unique=True)
        await self.sessions.create_index("token_hash", unique=True)
        await self.user_rules.create_index([("user_id", 1), ("label", 1)],
                                           unique=True)

    # ---- users ----

    async def create_user(self, username, password_hash, display_name=None):
        """Insert a new account. Raises ValueError if the username is taken."""
        username = username.strip()
        if await self.users.find_one({"username": username}):
            raise ValueError("username already exists")
        doc = {
            "_id": uuid4().hex,
            "username": username,
            "password_hash": password_hash,
            "display_name": (display_name or username).strip(),
            "created_at": time.time(),
        }
        await self.users.insert_one(doc)
        return doc

    async def get_user_by_username(self, username):
        return await self.users.find_one({"username": username.strip()})

    async def get_user(self, user_id):
        return await self.users.find_one({"_id": user_id})

    # ---- sessions (this is the login/logout record) ----

    async def create_session(self, user_id, token_hash, user_agent=None):
        doc = {
            "_id": uuid4().hex,
            "user_id": user_id,
            "token_hash": token_hash,
            "login_at": time.time(),
            "logout_at": None,
            "user_agent": user_agent,
        }
        await self.sessions.insert_one(doc)
        return doc

    async def get_active_session(self, token_hash):
        return await self.sessions.find_one(
            {"token_hash": token_hash, "logout_at": None})

    async def end_session(self, token_hash):
        """Stamp logout_at; returns True if an active session was closed."""
        result = await self.sessions.update_one(
            {"token_hash": token_hash, "logout_at": None},
            {"$set": {"logout_at": time.time()}})
        return result.modified_count > 0

    async def list_sessions(self, user_id, limit=20):
        cursor = self.sessions.find({"user_id": user_id}).sort("login_at", -1)
        docs = await cursor.to_list(length=limit)
        for doc in docs:
            doc.pop("token_hash", None)   # never expose, even hashed
        return docs

    # ---- per-user rules ----

    async def get_user_rules(self, user_id):
        cursor = self.user_rules.find({"user_id": user_id})
        docs = await cursor.to_list(length=1000)
        return {d["label"]: {"enabled": d["enabled"], "urgency": d["urgency"]}
                for d in docs}

    async def set_user_rule(self, user_id, label, enabled, urgency):
        await self.user_rules.update_one(
            {"user_id": user_id, "label": label},
            {"$set": {"enabled": bool(enabled), "urgency": urgency}},
            upsert=True)
        return {"enabled": bool(enabled), "urgency": urgency}

    # ---- per-user preferences ----

    async def get_prefs(self, user_id):
        doc = await self.user_prefs.find_one({"_id": user_id})
        return doc.get("prefs", {}) if doc else {}

    async def set_prefs(self, user_id, prefs):
        await self.user_prefs.update_one(
            {"_id": user_id}, {"$set": {"prefs": prefs}}, upsert=True)
        return prefs
