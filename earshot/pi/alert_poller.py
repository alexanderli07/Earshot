"""Pull-mode bridge: poll the laptop backend for events, trigger local alerts.

Use this when the hotspot blocks laptop -> Pi traffic (the backend's push
sink can't get through) but Pi -> laptop requests work. The poller asks the
backend for recent events twice a second and POSTs each new one to the
alert server already running on this Pi (alert-server.service, port 8000).

    python3 alert_poller.py http://172.20.10.9:8000    # laptop backend URL

Stdlib only. Events that happened before the poller started are ignored.
"""
import json
import sys
import time
import urllib.request

BACKEND = (sys.argv[1] if len(sys.argv) > 1 else "http://172.20.10.9:8000").rstrip("/")
ALERT = "http://127.0.0.1:8000/alarm"
POLL_S = 0.5

seen = set()
started = time.time()
print(f"polling {BACKEND}/events/recent -> {ALERT}")

while True:
    try:
        with urllib.request.urlopen(f"{BACKEND}/events/recent?limit=10", timeout=3) as r:
            events = json.loads(r.read())
    except Exception as exc:
        print(f"backend unreachable: {exc}", file=sys.stderr)
        time.sleep(2)
        continue
    for event in events:
        key = (event.get("id"), event.get("received_at"))
        if key in seen or (event.get("received_at") or 0) < started:
            continue
        seen.add(key)
        payload = json.dumps({"label": event.get("label"),
                              "urgency": event.get("urgency")}).encode()
        try:
            req = urllib.request.Request(
                ALERT, data=payload,
                headers={"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as r:
                print(f"alert {event.get('label')}/{event.get('urgency')}: "
                      f"{r.read().decode().strip()}")
        except Exception as exc:
            print(f"alert server failed: {exc}", file=sys.stderr)
    if len(seen) > 500:
        seen = set(list(seen)[-100:])
    time.sleep(POLL_S)
