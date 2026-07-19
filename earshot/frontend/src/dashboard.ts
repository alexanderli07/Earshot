/* Dashboard: live event feed over WebSocket, teach flow (3 recorded clips ->
 * /teach), and per-sound rules (on/off + urgency override -> /rules). */

/// <reference path="shared.ts" />

(() => {

const BASE_SOUNDS = [
  "smoke_alarm", "doorbell", "knock", "baby_cry", "glass_break",
];
const FEED_LIMIT = 100;
const CLIPS_REQUIRED = 3;
const CLIP_SECONDS = 2.0;

const hostInput = el<HTMLInputElement>("host");
hostInput.value = resolveHost();

const statusDot = el<HTMLSpanElement>("dot");
const statusText = el<HTMLSpanElement>("stat");
const feedList = el<HTMLUListElement>("feed");

/* ---- theme ---- */

const themeButton = el<HTMLButtonElement>("theme");

function applyTheme(theme: string): void {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("earshot_theme", theme); } catch { /* file:// */ }
  themeButton.textContent = theme === "dark" ? "light mode" : "dark mode";
}

themeButton.addEventListener("click", () => {
  applyTheme(document.documentElement.dataset.theme === "dark"
    ? "light" : "dark");
});

applyTheme(document.documentElement.dataset.theme || "light");

/* ---- auth state ---- */

const authEl = el<HTMLSpanElement>("auth");
let loggedIn = false;

function apiBase(): string { return httpBase(hostInput.value); }

/* When logged in, rules are the user's own (/me/rules); otherwise the
 * device-global rules (/rules). Same request/response shape either way. */
function rulesPath(): string { return loggedIn ? "/me/rules" : "/rules"; }

async function checkAuth(): Promise<void> {
  try {
    const r = await fetch(`${apiBase()}/auth/me`,
                          { credentials: "same-origin" });
    if (r.ok) {
      const user = await r.json() as { display_name: string };
      loggedIn = true;
      renderAuth(user.display_name);
    } else {
      loggedIn = false;
      renderAuth(null);
    }
  } catch {
    loggedIn = false;
    renderAuth(null);
  }
  void loadRules();
}

function renderAuth(displayName: string | null): void {
  authEl.innerHTML = "";
  if (displayName) {
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = displayName;
    const out = document.createElement("button");
    out.type = "button";
    out.textContent = "Log out";
    out.addEventListener("click", () => { void logout(); });
    authEl.append(who, out);
  } else {
    const link = document.createElement("a");
    link.textContent = "Log in";
    const host = hostInput.value;
    link.href = host && host !== location.host
      ? `login.html?host=${encodeURIComponent(host)}` : "login.html";
    authEl.append(link);
  }
}

async function logout(): Promise<void> {
  try {
    await fetch(`${apiBase()}/auth/logout`,
                { method: "POST", credentials: "same-origin" });
  } catch { /* ignore — cookie clears on the server side anyway */ }
  loggedIn = false;
  renderAuth(null);
  void loadRules();
}

/* ---- live feed ---- */

const seenIds = new Set<string>();

function addEventRow(ev: EarshotEvent): void {
  if (ev.id) {
    if (seenIds.has(ev.id)) return;
    seenIds.add(ev.id);
  }
  document.getElementById("empty")?.remove();

  const category = categoryOf(ev);
  const row = document.createElement("li");

  const time = document.createElement("span");
  time.className = "time";
  time.textContent = clockTime(ev.timestamp);

  const label = document.createElement("span");
  label.className = "label";
  label.textContent = prettyLabel(ev.label);

  const confidence = document.createElement("span");
  confidence.className = "conf";
  confidence.textContent =
    `${Math.round((ev.confidence ?? 1) * 100)}% certainty`;

  const chip = document.createElement("span");
  chip.className = `chip ${category}`;
  chip.textContent = category;

  row.append(time, label, confidence, chip);
  feedList.prepend(row);
  while (feedList.children.length > FEED_LIMIT) feedList.lastChild?.remove();

  allEvents.push(ev);
  renderStats();
}

/* ---- stats: tiles + category donut, recomputed on every event ---- */

const STAT_CATEGORIES: Category[] =
  ["urgent", "presence", "appliance", "taught"];
const CATEGORY_CSS_VAR: Record<Category, string> = {
  urgent: "--alarm",
  presence: "--door",
  appliance: "--appliance",
  taught: "--taught",
};
const SVG_NS = "http://www.w3.org/2000/svg";

const allEvents: EarshotEvent[] = [];
const donutSvg = el<SVGSVGElement & HTMLElement>("donut");
const legendList = el<HTMLUListElement>("legend");

function formatGap(seconds: number): string {
  if (!isFinite(seconds)) return "—";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  if (seconds < 5400) return `${Math.round(seconds / 60)}m`;
  return `${(seconds / 3600).toFixed(1)}h`;
}

function renderStats(): void {
  const now = new Date();
  const midnight = new Date(
    now.getFullYear(), now.getMonth(), now.getDate()).getTime() / 1000;
  const today = allEvents.filter((e) => (e.timestamp ?? 0) >= midnight);
  el("stAlerts").textContent = String(today.length);

  const times = allEvents
    .map((e) => e.timestamp ?? 0)
    .filter(Boolean)
    .sort((a, b) => a - b);
  const gap = times.length >= 2
    ? (times[times.length - 1] - times[0]) / (times.length - 1)
    : NaN;
  el("stGap").textContent = formatGap(gap);

  const confs = allEvents
    .map((e) => e.confidence)
    .filter((c): c is number => typeof c === "number");
  el("stConf").textContent = confs.length
    ? `${Math.round((confs.reduce((s, c) => s + c, 0) / confs.length) * 100)}%`
    : "—";

  const counts = new Map<string, number>();
  for (const e of allEvents) {
    const key = prettyLabel(e.label);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  let top = "—";
  let best = 0;
  for (const [label, n] of counts) {
    if (n > best) { best = n; top = label; }
  }
  el("stTop").textContent = top;

  renderDonut();
}

function donutCircle(radius: number, stroke: string): SVGCircleElement {
  const circle = document.createElementNS(SVG_NS, "circle");
  circle.setAttribute("cx", "70");
  circle.setAttribute("cy", "70");
  circle.setAttribute("r", String(radius));
  circle.setAttribute("fill", "none");
  circle.setAttribute("stroke", stroke);
  circle.setAttribute("stroke-width", "16");
  return circle;
}

function renderDonut(): void {
  const byCategory: Record<Category, number> =
    { urgent: 0, presence: 0, appliance: 0, taught: 0 };
  for (const e of allEvents) byCategory[categoryOf(e)] += 1;
  const total = allEvents.length;
  const radius = 52;
  const circumference = 2 * Math.PI * radius;

  donutSvg.innerHTML = "";
  legendList.innerHTML = "";
  donutSvg.appendChild(donutCircle(radius, "var(--line)"));

  let offset = 0;
  for (const category of STAT_CATEGORIES) {
    const n = byCategory[category];
    if (!n) continue;
    const fraction = n / total;
    const segment =
      donutCircle(radius, `var(${CATEGORY_CSS_VAR[category]})`);
    segment.setAttribute(
      "stroke-dasharray", `${fraction * circumference} ${circumference}`);
    segment.setAttribute(
      "stroke-dashoffset", String(-offset * circumference));
    segment.setAttribute("transform", "rotate(-90 70 70)");
    donutSvg.appendChild(segment);
    offset += fraction;

    const item = document.createElement("li");
    const swatch = document.createElement("i");
    swatch.style.background = `var(${CATEGORY_CSS_VAR[category]})`;
    const text = document.createElement("span");
    text.textContent = `${category} · ${n}`;
    item.append(swatch, text);
    legendList.appendChild(item);
  }

  const center = document.createElementNS(SVG_NS, "text");
  center.setAttribute("x", "70");
  center.setAttribute("y", "67");
  center.setAttribute("text-anchor", "middle");
  center.setAttribute("fill", "var(--ink)");
  center.setAttribute(
    "style", "font: 800 26px 'Bricolage Grotesque', sans-serif;");
  center.textContent = String(total);
  const centerSub = document.createElementNS(SVG_NS, "text");
  centerSub.setAttribute("x", "70");
  centerSub.setAttribute("y", "86");
  centerSub.setAttribute("text-anchor", "middle");
  centerSub.setAttribute("fill", "var(--ink-soft)");
  centerSub.setAttribute(
    "style",
    "font: 500 10px 'IBM Plex Mono', monospace; " +
    "letter-spacing: .12em; text-transform: uppercase;");
  centerSub.textContent = total === 1 ? "alert" : "alerts";
  donutSvg.append(center, centerSub);

  if (!total) {
    const item = document.createElement("li");
    const text = document.createElement("span");
    text.textContent = "no alerts yet";
    item.append(text);
    legendList.appendChild(item);
  }
}

renderStats();

/* ---- socket ---- */

const socket = new EventSocket(() => hostInput.value, {
  onEvent: addEventRow,
  onStatus: (connected) => {
    statusDot.className = connected ? "dot on" : "dot off";
    statusText.textContent = connected ? "connected" : "reconnecting";
    if (connected) void loadRules();
  },
});

hostInput.addEventListener("change", () => {
  saveHost(hostInput.value);
  seenIds.clear();
  feedList.innerHTML = "";
  allEvents.length = 0;
  renderStats();
  socket.restart();
});

socket.connect();

/* ---- teach flow ---- */

const recordButton = el<HTMLButtonElement>("rec");
const teachButton = el<HTMLButtonElement>("teach");
const teachNote = el<HTMLDivElement>("tnote");
const nameInput = el<HTMLInputElement>("tname");
const clipMarks = [0, 1, 2].map((i) => el<HTMLElement>(`d${i}`));

const clips: Blob[] = [];

function setNote(text: string, kind: "" | "ok" | "err" = ""): void {
  teachNote.className = kind ? `note ${kind}` : "note";
  teachNote.textContent = text;
}

function refreshTeachButton(): void {
  teachButton.disabled =
    !(nameInput.value.trim() && clips.length === CLIPS_REQUIRED);
}

recordButton.addEventListener("click", () => {
  void (async () => {
    if (clips.length >= CLIPS_REQUIRED) return;
    recordButton.disabled = true;
    setNote("Recording...");
    try {
      clips.push(await recordWav(CLIP_SECONDS));
    } catch (err) {
      setNote(`Mic error: ${(err as Error).message}`, "err");
      recordButton.disabled = false;
      return;
    }
    clipMarks[clips.length - 1].className = "done";
    setNote(`Captured clip ${clips.length} of ${CLIPS_REQUIRED}`);
    if (clips.length < CLIPS_REQUIRED) {
      recordButton.textContent =
        `Record clip ${clips.length + 1} of ${CLIPS_REQUIRED}`;
      recordButton.disabled = false;
    } else {
      recordButton.textContent = "3 clips recorded";
    }
    refreshTeachButton();
  })();
});

nameInput.addEventListener("input", refreshTeachButton);

teachButton.addEventListener("click", () => {
  void (async () => {
    teachButton.disabled = true;
    setNote("Teaching...");
    const form = new FormData();
    form.append("name", nameInput.value.trim());
    clips.forEach((blob, i) => form.append("clips", blob, `clip${i}.wav`));
    try {
      const response = await fetch(`${httpBase(hostInput.value)}/teach`, {
        method: "POST",
        body: form,
      });
      const result = await response.json() as
        { ok?: boolean; learned?: { name: string }[]; detail?: string };
      if (response.ok && result.ok) {
        const names = (result.learned ?? []).map((s) => s.name).join(", ");
        setNote(`Learned "${nameInput.value.trim()}". Known: ${names}`, "ok");
        resetTeach();
        void loadRules();
      } else {
        // Backend reports failures as HTTP errors with a `detail` message
        // (422 bad clips, 503 ML unavailable, 504 timeout).
        setNote(`Teach failed: ${result.detail ?? response.statusText}`, "err");
        teachButton.disabled = false;
      }
    } catch (err) {
      setNote(`Network error: ${(err as Error).message}`, "err");
    }
  })();
});

function resetTeach(): void {
  clips.length = 0;
  clipMarks.forEach((mark) => { mark.className = ""; });
  recordButton.textContent = "Record clip 1 of 3";
  recordButton.disabled = false;
  teachButton.disabled = true;
  nameInput.value = "";
}

/* ---- audio capture: PCM float via WebAudio, encoded as 16-bit WAV so the
 * ML side (stdlib wave reader) can parse it ---- */

async function recordWav(seconds: number): Promise<Blob> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  type AudioContextCtor = typeof AudioContext;
  const Ctor: AudioContextCtor = window.AudioContext
    ?? (window as unknown as { webkitAudioContext: AudioContextCtor })
      .webkitAudioContext;
  const context = new Ctor();
  const source = context.createMediaStreamSource(stream);
  const processor = context.createScriptProcessor(4096, 1, 1);
  const muted = context.createGain();
  muted.gain.value = 0; // keep the graph alive without mic-to-speaker feedback

  const buffers: Float32Array[] = [];
  processor.onaudioprocess = (event: AudioProcessingEvent) => {
    buffers.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(muted);
  muted.connect(context.destination);

  await new Promise((resolve) => setTimeout(resolve, seconds * 1000));

  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((track) => track.stop());
  const sampleRate = context.sampleRate;
  await context.close();

  const total = buffers.reduce((sum, b) => sum + b.length, 0);
  const samples = new Float32Array(total);
  let offset = 0;
  for (const buffer of buffers) {
    samples.set(buffer, offset);
    offset += buffer.length;
  }
  return encodeWav(samples, sampleRate);
}

/* ---- rules ---- */

interface Rule { enabled: boolean; urgency: string | null }

const rulesList = el<HTMLUListElement>("rules");

async function loadRules(): Promise<void> {
  const base = httpBase(hostInput.value);
  let rules: Record<string, Rule> = {};
  let learned: { name: string }[] = [];
  try {
    rules = await (await fetch(`${base}${rulesPath()}`,
                               { credentials: "same-origin" })).json() as
      Record<string, Rule>;
  } catch { /* backend not reachable yet */ }
  try {
    learned = await (await fetch(`${base}/sounds`)).json() as
      { name: string }[];
  } catch { /* ML may be absent; base sounds still render */ }

  const labels =
    [...new Set([
      ...BASE_SOUNDS,
      ...learned.map((sound) => canonicalEventLabel(sound.name)),
    ])];
  rulesList.innerHTML = "";
  for (const label of labels) {
    const rule = rules[label] ?? { enabled: true, urgency: null };
    rulesList.appendChild(buildRuleRow(label, rule));
  }
}

function buildRuleRow(label: string, rule: Rule): HTMLLIElement {
  const row = document.createElement("li");

  const name = document.createElement("span");
  name.className = "rl";
  name.textContent = prettyLabel(label);

  const select = document.createElement("select");
  for (const option of ["auto", "high", "medium", "low"]) {
    const opt = document.createElement("option");
    opt.value = option === "auto" ? "" : option;
    opt.textContent = option;
    opt.selected = (rule.urgency ?? "") === opt.value;
    select.appendChild(opt);
  }

  const toggleWrap = document.createElement("label");
  toggleWrap.className = "sw";
  const toggle = document.createElement("input");
  toggle.type = "checkbox";
  toggle.checked = rule.enabled !== false;
  const knob = document.createElement("span");
  toggleWrap.append(toggle, knob);

  const save = () => void putRule(label, toggle.checked, select.value || null);
  select.addEventListener("change", save);
  toggle.addEventListener("change", save);

  row.append(name, select, toggleWrap);
  return row;
}

async function putRule(label: string, enabled: boolean,
                       urgency: string | null): Promise<void> {
  try {
    await fetch(
      `${apiBase()}${rulesPath()}/${encodeURIComponent(label)}`,
      {
        method: "PUT",
        headers: { "content-type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ enabled, urgency }),
      });
  } catch { /* dropped connection; next load re-syncs */ }
}

void checkAuth();

})();
