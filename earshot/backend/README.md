# Earshot backend — one event in, four alerts out

FastAPI switchboard on the Pi. Any event — real (from ML) or fake (debug
endpoint) — becomes **light, buzz, phone push, and a dashboard row** within
1 second. Broadcasts over WebSocket, pushes to phones via ntfy, drives the
RGB LED + vibration motor over GPIO.

This is demo software, not a certified alarm or life-safety service. Do not
expose its unauthenticated debug, teach, or rule endpoints to an untrusted
network, and never use it to replace approved alarms or emergency procedures.

Files: [config.py](app/config.py) (pins, ntfy, alert profiles — the tuning
knobs), [core.py](app/core.py) (events | recent | rules | dispatch),
[sinks.py](app/sinks.py) (WebSocket | GPIO | ntfy), [ml_bridge.py](app/ml_bridge.py)
(optional ML integration), [main.py](app/main.py) (FastAPI app).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ../ml

export EARSHOT_MODEL_DIR="$HOME/.local/share/earshot/models"
mkdir -p "$EARSHOT_MODEL_DIR"
earshot download
python -m pip check

export EARSHOT_NTFY_TOPIC=earshot-<pick-something-unguessable>   # optional; push off if unset
python -m app.main            # serves on 0.0.0.0:8000
```

Use one environment for the backend and ML runtime. Installing only
`requirements.txt` leaves NumPy, sounddevice, and LiteRT unavailable and makes
`/healthz` correctly report ML as unavailable. The Pi does not need the
Windows-only scikit-learn training extra.

Off a Pi, GPIO auto-falls back to a logging mock, so it runs on a laptop for
development. On the Pi it drives the real pins (R=17, G=27, B=22, motor=18).

## Prove it works (the "done when")

```bash
curl -X POST localhost:8000/debug/event \
     -H 'content-type: application/json' \
     -d '{"label":"smoke_alarm","urgency":"high"}'
```
→ LED turns red + strobes, motor does a long buzz, the phone push arrives (if
ntfy configured), and every connected dashboard gets a new row — all at once.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| WS   | `/ws`            | broadcasts every event as JSON (dashboard + wearable) |
| GET  | `/healthz`       | server + ML + GPIO + ntfy status |
| GET  | `/events/recent?limit=` | recent events (in-memory) |
| POST | `/debug/event`   | fire a fake event — build without waiting on ML |
| POST | `/teach`         | `name` + 3 audio files → ML teach |
| GET  | `/sounds`        | taught sounds |
| GET  | `/rules`         | per-sound rules |
| PUT  | `/rules/{label}` | `{enabled, urgency}` — mute or override, persisted to JSON |

## Urgency → alert (edit in [config.py](app/config.py))

| urgency | LED | motor | ntfy |
|---------|-----|-------|------|
| high    | red, strobe | strobe + long buzz | urgent |
| medium  | yellow, pulse | one pulse | high |
| low     | blue, blink | short tick | default |

Rules can mute a sound or override its urgency at runtime.

## Networking (read this before the demo)

Venue Wi-Fi almost always isolates clients, so phones can't reach the Pi on
it. **Run everything on a phone hotspot from minute one, including the demo.**
ntfy still works on any network because it only needs outbound internet.

## ML integration

A compatible trained head emits the same public event used by the YAMNet
fallback:

```json
{
  "label": "smoke_alarm",
  "urgency": "high",
  "source": "trained"
}
```

The source is preserved through dispatch, and high urgency receives the top
GPIO/push priority. Incoming `fire_alarm` and `fire_smoke_alarm` events are
canonicalized to `smoke_alarm`. Saved rules are exposed under that same key;
an existing `smoke_alarm` rule wins, otherwise one legacy alarm rule is
migrated deterministically.

The backend runs standalone (debug events) with no ML present. When the
sibling `../ml` package is importable and its model is downloaded, live mic
events flow in automatically and `/teach` works — including while detection
is live (the ML serializes interpreter access internally).

`/teach` requires exactly 3 clips (max 5 MB each), runs decode + inference
off the event loop with a 30 s deadline, and deletes the temporary audio
files whether teaching succeeds or fails. `/healthz` reports `ml.alive`
(listener thread actually running) separately from `ml.available` (package
imported), plus engine, listener, asynchronous-dispatch, or stop-timeout
errors. `smoke_alarm` and every configured event label are reserved from
both the CLI and direct/backend teach API.

## Per-user accounts (MongoDB, optional)

Login + per-user rules and preferences live in MongoDB. **Auth is disabled
until `EARSHOT_MONGO_URI` is set**, so the demo runs with zero Mongo and
nothing here changes unless you opt in.

Any MongoDB works — a local server or **MongoDB Atlas** (cloud). Same driver,
just a different connection string.

**Option A — local (fully offline):** `./setup-mongo.sh` downloads the
community server to `~/.local/mongodb` and starts it on `localhost:27017` — no
Homebrew, Docker, or admin rights needed. Then:

```bash
export EARSHOT_MONGO_URI=mongodb://localhost:27017   # enable accounts
export EARSHOT_MONGO_DB=earshot                       # optional (default)
export EARSHOT_COOKIE_SECURE=1                         # only behind HTTPS
```

**Option B — MongoDB Atlas (cloud, shared across devices):**

1. Create a free cluster at <https://cloud.mongodb.com>.
2. **Database Access** → add a database user (username + password).
3. **Network Access** → allow your IP (or `0.0.0.0/0` for a hackathon demo —
   understand that opens it to the internet; the DB user password is the only
   guard).
4. **Connect → Drivers → Python** → copy the `mongodb+srv://…` string and set it
   (URL-encode any special characters in the password):

```bash
export EARSHOT_MONGO_URI='mongodb+srv://USER:PASS@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority'
export EARSHOT_MONGO_DB=earshot
```

The `dnspython` and `certifi` dependencies (in `requirements.txt`) handle the
`+srv` lookup and Atlas TLS. On startup the backend pings the cluster; if it
can't connect (wrong password, IP not allow-listed) it logs the reason and
**degrades to auth-off rather than crashing** — the alert path keeps working.

> **Atlas is a cloud database.** Account data (usernames, bcrypt hashes,
> per-user rules, login history) leaves the room. Detection audio never does,
> and accounts aren't on the alert path, but this trades Earshot's offline
> story for cross-device convenience. Use local Mongo (Option A) to stay fully
> offline. Never commit the connection string — it contains a password; keep it
> in the environment only.

Endpoints (all 503 when auth is off):

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/auth/register` | create account + log in |
| POST | `/auth/login` | log in (sets an httpOnly session cookie) |
| POST | `/auth/logout` | log out (stamps the session's logout time) |
| GET  | `/auth/me` | current user |
| GET  | `/auth/sessions` | this user's login/logout history |
| GET/PUT | `/me/rules`, `/me/rules/{label}` | per-user rule overrides |
| GET/PUT | `/me/prefs` | per-user preferences (own ntfy topic, shown categories) |

Security posture:

- Passwords are **bcrypt**-hashed; plaintext is never stored or logged.
- Session tokens are random 256-bit values in an **httpOnly** cookie; only
  their SHA-256 hash is stored, so a DB leak can't be replayed as a session.
- **Only the per-user (`/me/*`) and account (`/auth/*`) surface is gated.**
  The live alert path — `/ws`, `/debug/event`, device `/rules` — stays open:
  a login must never stand between a user and a smoke alarm.
- No HTTPS on the hotspot means credentials cross the local network in
  cleartext. Acceptable on a private hotspot demo; set `EARSHOT_COOKIE_SECURE`
  and terminate TLS for anything beyond that.

Collections: `users`, `sessions` (the login/logout record), `user_rules`,
`user_prefs`.

## Tests (no hardware or network)

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

The tests inject fake ML engines and sinks, and run the real Mongo store code
against an in-memory `mongomock` database (`pip install mongomock-motor`).
They do not open a microphone, load a model, drive GPIO, send notifications,
or need a running `mongod`.
