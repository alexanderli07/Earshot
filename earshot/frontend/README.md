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
| urgent | `#d92d20` red | smoke/fire alarm, baby cry, glass break |
| presence | `#2e6fe8` blue | doorbell, knock — someone is here |
| appliance | `#d97706` amber | microwave and other appliances |
| taught | `#12805c` green | user-taught sounds |

Wearable vibration patterns mirror the Pi motor patterns
(high = strobe + long buzz, medium = one pulse, low = short tick).

## Tests

```bash
node tests/test_frontend.mjs                            # logic + WAV roundtrip
EARSHOT_TEST_HOST=localhost:8000 node tests/test_frontend.mjs   # + live WS test
```

The WAV roundtrip writes the encoder's output through the ML component's
actual Python wav loader, proving browser-recorded teach clips parse on the
other side.
