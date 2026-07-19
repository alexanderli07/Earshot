"""Alarm receiver for the Pi wearable/alert unit.

The laptop backend forwards every detection here as
POST /alarm {"label": ..., "urgency": ...} (see sinks.pi_alert and
EARSHOT_PI_URL). This end relays each alarm to an Arduino over USB
serial as one line: "label,urgency\n" — parse that in the sketch and
drive the LED/buzzer/motor from it.

Stdlib only, so it runs on a bare Pi image:

    python3 pi_alarm_receiver.py            # listens on 0.0.0.0:8000

With pyserial installed it auto-opens the first /dev/ttyACM* or
/dev/ttyUSB* port at 9600 baud; without it (or with no Arduino
plugged in) alarms still print to stdout so the wire is testable.
"""
import glob
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BAUD = 9600


def open_serial():
    try:
        import serial
    except ImportError:
        print("pyserial not installed; printing alarms only", file=sys.stderr)
        return None
    for port in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
        try:
            conn = serial.Serial(port, BAUD, timeout=1)
            print(f"arduino on {port}")
            return conn
        except Exception as exc:
            print(f"could not open {port}: {exc}", file=sys.stderr)
    print("no arduino serial port found; printing alarms only", file=sys.stderr)
    return None


ARDUINO = None


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/alarm":
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400)
            return
        label = str(body.get("label", "unknown"))
        urgency = str(body.get("urgency", "high"))
        line = f"{label},{urgency}\n"
        print(f"ALARM: {line.strip()}")
        if ARDUINO is not None:
            try:
                ARDUINO.write(line.encode())
            except Exception as exc:
                print(f"serial write failed: {exc}", file=sys.stderr)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ARDUINO = open_serial()
    print(f"listening on 0.0.0.0:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
