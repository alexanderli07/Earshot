# Pi wearable alert unit

The physical alert device is a **Raspberry Pi driving its own GPIO pins**
(RGB LED + vibration motor via TB6612 driver + buzzer). There is **no
Arduino** in the rig — the SparkFun RedBoard stayed in its box.

`alert_server.py` is the code deployed on the demo Pi (user `pi`, at
`/home/pi/alert_server.py`, hotspot IP `172.20.10.11`). It runs under
systemd as `alert-server.service` — auto-starts on boot, auto-restarts on
crash — so the Pi is a plug-and-play appliance: power it and it listens.

## How alerts reach it

The backend's `pi_alert` sink (see `backend/app/sinks.py`) POSTs
`{"label": ..., "urgency": ...}` to `http://<pi>:8000/alarm` for every
forwarded event. Set `EARSHOT_PI_URL=http://172.20.10.11:8000` before
launching the backend, and keep every machine on the same phone hotspot.

Urgency → behavior: `high` = red strobe + hard shake + T-3 beeps (20 s),
`medium` = yellow pulse + medium buzz (8 s), `low` = blue blink + tick (4 s).
`POST /stop` cancels. GET works too (phone bookmark = manual trigger).

## Deploying changes to the Pi

```bash
# on the Pi (SSH):
curl -o /home/pi/alert_server.py https://raw.githubusercontent.com/alexanderli07/Earshot/main/earshot/pi/alert_server.py
sudo systemctl restart alert-server
```

## Note on backend/pi_alarm_receiver.py

`pi_alarm_receiver.py` was written for a Pi-relays-to-Arduino-over-USB
setup that the current hardware does not use. Do NOT run it on the demo
Pi — it binds the same port 8000 and would conflict with (or replace) the
real alert server that drives the actual hardware.
