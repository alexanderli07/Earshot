# Earshot frontend — the dashboard and the wearable

Two static pages, TypeScript sources, no bundler. `tsc` compiles `src/` to
plain scripts in `js/`, which are committed — so the Pi (and judges) never
need Node; the backend just serves the files.

| Page | Runs on | Does |
|------|---------|------|
| `dashboard.html` | laptop | live event feed over WebSocket, teach flow (3 recorded clips), per-sound rules |
| `wearable.html` | the wrist phone (Android Chrome) | full-screen flash in the urgency color, giant label, vibration pattern |

## Use

Served by the backend at `http://<pi>:8000/ui/dashboard.html` and
`/ui/wearable.html` — same origin, so the pages find the backend
automatically. Opened any other way (file://, another host), set the Pi
`host:port` in the input on either page (also `?host=pi-ip:8000`).

Browser microphone capture requires a secure context. `http://localhost` is
treated as trustworthy, but `http://<pi-ip>` generally is not. For laptop
teaching while the backend runs on the Pi, serve the static files locally:

```powershell
Set-Location .\earshot\frontend
..\ml\.venv\Scripts\python.exe -m http.server 5173
```

Open `http://localhost:5173/dashboard.html?host=<PI-IP>:8000`. For a deployed
non-localhost origin, configure HTTPS instead of weakening browser security.

**Wearable:** tap **ARM** once at demo start — mobile browsers only allow
vibration and wake lock after a user gesture. iOS ignores the vibration API,
so the wrist phone is an Android.

## Develop

```bash
cd frontend
npm install        # just typescript
npm run build      # tsc: src/*.ts -> js/*.js   (commit the js/ output)
npm run watch
```

`src/shared.ts` — event types, palette mapping, host resolution, reconnecting
WebSocket, WAV encoder. `src/dashboard.ts`, `src/wearable.ts` — one file per
page, compiled as classic scripts (no modules) so pages work over `file://`
too.

## Palette (flat, no gradients)

| Category | Color | Meaning |
|----------|-------|---------|
| urgent | `#D93036` alarm red | smoke/fire alarm, baby cry, glass break |
| presence | `#2456D6` door blue | doorbell, knock — someone is here |
| appliance | `#DB8B00` amber | kettles and other appliances |
| taught | `#178A50` green | user-taught sounds |

Theme matches the pitch page: cool-fog light background, white cards, ink
text, IBM Plex Mono labels, Bricolage Grotesque headings, Atkinson
Hyperlegible body. The wearable is the one dark (ink) surface. Google Fonts
load when internet is available and fall back to system fonts offline.

Wearable vibration patterns mirror the Pi motor patterns
(high = strobe + long buzz, medium = one pulse, low = short tick).

The trained `fire_smoke_alarm` label is explicitly urgent and receives its own
base rule. Legacy `smoke_alarm` and `fire_alarm` labels remain for operation
without a trained head. Existing saved rules are not automatically copied to
the new label; configure it explicitly after deployment.

## Tests

```bash
node tests/test_frontend.mjs                            # logic + WAV roundtrip
EARSHOT_TEST_HOST=localhost:8000 node tests/test_frontend.mjs   # + live WS test
```

On Windows PowerShell, point the test at the shared ML environment explicitly:

```powershell
$env:EARSHOT_TEST_PYTHON = (Resolve-Path '..\ml\.venv\Scripts\python.exe')
node .\tests\test_frontend.mjs
```

Without the override, the test uses `.venv\Scripts\python.exe` on Windows and
`.venv/bin/python` elsewhere.

The WAV roundtrip writes the encoder's output through the ML component's
actual Python wav loader, proving browser-recorded teach clips parse on the
other side.

This interface is part of a demo, not a certified alerting or life-safety
system. A browser tab, WebSocket, phone, vibration API, or network can fail;
never use the UI to replace approved alarms or emergency procedures.
