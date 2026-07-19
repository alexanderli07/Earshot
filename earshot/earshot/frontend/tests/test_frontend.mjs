/* Integration tests for the COMPILED frontend (js/shared.js), run in Node 22+
 * (global fetch + WebSocket).
 *
 *   node tests/test_frontend.mjs            # logic + WAV roundtrip
 *   EARSHOT_TEST_HOST=localhost:8077 \
 *   node tests/test_frontend.mjs            # + live backend WebSocket test
 *
 * The WAV roundtrip feeds the encoder's output into the ML component's actual
 * Python loader (ml/earshot_ml/audio path) — proving the browser-recorded
 * teach clips are readable on the other side.
 */

import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const repo = join(here, "..", "..");

let passed = 0, failed = 0;
function check(name, ok, detail = "") {
  if (ok) { passed++; console.log(`ok   ${name}`); }
  else { failed++; console.log(`FAIL ${name} ${detail}`); }
}

/* ---- load the real compiled shared.js into a sandbox ---- */
const sandbox = {
  location: { search: "", host: "test:1", protocol: "http:" },
  localStorage: { getItem: () => null, setItem: () => {} },
  document: { getElementById: () => null },
  setTimeout, clearTimeout, WebSocket: globalThis.WebSocket, Blob, console,
};
vm.createContext(sandbox);
vm.runInContext(readFileSync(join(here, "..", "js", "shared.js"), "utf8"),
                sandbox);
const dashboardSource = readFileSync(
  join(here, "..", "js", "dashboard.js"), "utf8",
);

/* ---- categoryOf: palette mapping ---- */
const c = (ev) => sandbox.categoryOf(ev);
check("smoke_alarm is urgent", c({ label: "smoke_alarm", urgency: "high" }) === "urgent");
check("doorbell is presence", c({ label: "doorbell", urgency: "medium" }) === "presence");
check("unknown low is appliance", c({ label: "mystery", urgency: "low" }) === "appliance");
check("taught source wins", c({ label: "kettle", source: "taught", urgency: "high" }) === "taught");
check("unknown high is urgent", c({ label: "mystery", urgency: "high" }) === "urgent");
check("wsUrl shape", sandbox.wsUrl("pi:8000") === "ws://pi:8000/ws");
check(
  "legacy alarm labels display as smoke alarm",
  sandbox.prettyLabel("fire_alarm") === "smoke alarm" &&
    sandbox.prettyLabel("fire_smoke_alarm") === "smoke alarm",
);
const baseSounds = /const BASE_SOUNDS = \[([\s\S]*?)\];/.exec(dashboardSource)?.[1] ?? "";
check(
  "smoke_alarm is the only alarm base rule",
  baseSounds.includes('"smoke_alarm"') &&
    !baseSounds.includes('"fire_alarm"') &&
    !baseSounds.includes('"fire_smoke_alarm"'),
);

/* ---- WAV roundtrip: encodeWav -> ML python loader ---- */
const rate = 48000, seconds = 1.0, freq = 440;
const samples = new Float32Array(rate * seconds);
for (let i = 0; i < samples.length; i++) {
  samples[i] = 0.5 * Math.sin(2 * Math.PI * freq * i / rate);
}
const blob = sandbox.encodeWav(samples, rate);
const wavBytes = Buffer.from(await blob.arrayBuffer());
check("wav has RIFF/WAVE header",
      wavBytes.subarray(0, 4).toString() === "RIFF" &&
      wavBytes.subarray(8, 12).toString() === "WAVE");
check("wav size = 44 + 2N", wavBytes.length === 44 + samples.length * 2);

const tmp = mkdtempSync(join(tmpdir(), "earshot-fe-"));
const wavPath = join(tmp, "tone.wav");
writeFileSync(wavPath, wavBytes);
try {
  const py = join(repo, "ml", ".venv", "bin", "python");
  const out = execFileSync(py, ["-c", `
import sys; sys.path.insert(0, ${JSON.stringify(join(repo, "ml"))}
)
from earshot_ml.pipeline import load_wav_16k_mono
import numpy as np
x = load_wav_16k_mono(${JSON.stringify(wavPath)})
print(len(x), round(float(np.abs(x).max()), 2))
`], { encoding: "utf8" }).trim();
  const [n, peak] = out.split(" ");
  check("ML loader resamples 48k->16k (~16000 samples)",
        Math.abs(Number(n) - 16000) <= 2, `got ${n}`);
  check("ML loader preserves amplitude (~0.5 peak)",
        Math.abs(Number(peak) - 0.5) < 0.05, `got ${peak}`);
} catch (err) {
  check("ML loader roundtrip", false, String(err).slice(0, 200));
}

/* ---- live backend: EventSocket receives a debug-fired event ---- */
const host = process.env.EARSHOT_TEST_HOST;
if (!host) {
  console.log("skip live backend test (set EARSHOT_TEST_HOST=host:port)");
} else {
  const got = await new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), 5000);
    /* class declarations live in the vm context's lexical scope, not on the
     * sandbox object — pull the constructor out with an expression */
    const EventSocket = vm.runInContext("EventSocket", sandbox);
    const socket = new EventSocket(() => host, {
      onEvent: (ev) => { clearTimeout(timer); resolve(ev); },
      onStatus: (up) => {
        if (up) {
          fetch(`http://${host}/debug/event`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ label: "glass_break", urgency: "high" }),
          }).catch(() => {});
        }
      },
    });
    socket.connect();
  });
  check("live: event received over WebSocket", got !== null);
  if (got) {
    check("live: label round-tripped", got.label === "glass_break");
    check("live: categorized urgent", c(got) === "urgent");
  }
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
