/* Virtual puck: browser stand-in for the physical alert hardware.
 *
 * Renders the RGB LED and vibration motor with the same urgency profiles,
 * pattern timings, and priority rules as the Pi GPIO controller, so the
 * full experience demos on a bare laptop. No hardware required.
 */

/// <reference path="shared.ts" />

(() => {

/* Mirrors backend config: ALERT_PROFILES + MOTOR_PATTERNS + LED_PATTERNS.
 * The physical LED encodes urgency (red / yellow / blue); on screen the
 * yellow channel reads as the palette's amber. */
interface PuckProfile {
  color: string;
  led: { on: number; off: number };
  motor: number[];               // alternating on/off seconds, starts ON
}

const PROFILES: Record<string, PuckProfile> = {
  high:   { color: "#D93036", led: { on: 0.12, off: 0.12 },
            motor: [0.5, 0.12, 0.5, 0.12, 1.6] },
  medium: { color: "#DB8B00", led: { on: 0.45, off: 0.25 },
            motor: [0.30] },
  low:    { color: "#2456D6", led: { on: 0.20, off: 0.60 },
            motor: [0.12] },
};

/* Mirrors backend URGENCY_RANK + the alert controller's latch: a lower
 * urgency never interrupts an active higher one. */
const RANK: Record<string, number> = { low: 1, medium: 2, high: 3 };

const REPLAY_GRACE_MS = 400;
const IDLE_HOLD_MS = 900;        // linger on the label after the pattern

const hostInput = el<HTMLInputElement>("host");
hostInput.value = resolveHost();
const statusDot = el<HTMLSpanElement>("dot");
const statusText = el<HTMLSpanElement>("stat");
const ledEl = el<HTMLDivElement>("led");
const motorEl = el<HTMLDivElement>("motor");
const eventEl = el<HTMLDivElement>("evt");
const stateEl = el<HTMLDivElement>("state");

/* ---- timeline: port of the backend's build_timeline ---- */

type Step = { duration: number; ledOn: boolean; motorOn: boolean };

function buildTimeline(profile: PuckProfile): Step[] {
  const total = profile.motor.reduce((sum, d) => sum + d, 0);
  const period = profile.led.on + profile.led.off;

  const boundaries = new Set<number>([0, total]);
  let t = 0;
  for (const duration of profile.motor) {
    t += duration;
    boundaries.add(Math.min(t, total));
  }
  for (t = 0; t < total; t += period) {
    boundaries.add(t);
    boundaries.add(Math.min(t + profile.led.on, total));
  }

  const motorOnAt = (instant: number): boolean => {
    let at = 0;
    let on = true;
    for (const duration of profile.motor) {
      if (instant < at + duration) return on;
      at += duration;
      on = !on;
    }
    return false;
  };
  const ledOnAt = (instant: number): boolean =>
    (instant % period) < profile.led.on;

  const ordered = [...boundaries].sort((a, b) => a - b);
  const steps: Step[] = [];
  for (let i = 0; i + 1 < ordered.length; i++) {
    const duration = ordered[i + 1] - ordered[i];
    if (duration > 1e-9) {
      steps.push({ duration, ledOn: ledOnAt(ordered[i]),
                   motorOn: motorOnAt(ordered[i]) });
    }
  }
  return steps;
}

/* ---- rendering ---- */

function setLed(color: string | null): void {
  if (color === null) {
    ledEl.style.background = "";
    ledEl.style.boxShadow = "";
    ledEl.classList.remove("lit");
  } else {
    ledEl.style.background = color;
    ledEl.style.boxShadow = `0 0 34px 6px ${color}66`;
    ledEl.classList.add("lit");
  }
}

function setMotor(on: boolean): void {
  motorEl.classList.toggle("buzz", on);
}

const sleep = (seconds: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, seconds * 1000));

/* ---- the controller: token preemption + priority latch ---- */

let runToken = 0;
let activeRank = 0;

async function play(ev: EarshotEvent): Promise<void> {
  const rank = RANK[ev.urgency] ?? 2;
  if (rank < activeRank) return;         // latched: lower can't preempt
  const token = ++runToken;
  activeRank = rank;

  const profile = PROFILES[ev.urgency] ?? PROFILES.medium;
  eventEl.textContent = prettyLabel(ev.label);
  eventEl.style.color = profile.color;
  stateEl.textContent = `${ev.urgency} alert`;
  if (navigator.vibrate) {
    navigator.vibrate(profile.motor.map((s) => Math.round(s * 1000)));
  }

  for (const step of buildTimeline(profile)) {
    if (token !== runToken) return;      // preempted by an admitted alert
    setLed(step.ledOn ? profile.color : null);
    setMotor(step.motorOn);
    await sleep(step.duration);
  }
  if (token !== runToken) return;
  setLed(null);
  setMotor(false);
  await sleep(IDLE_HOLD_MS / 1000);
  if (token !== runToken) return;
  activeRank = 0;
  eventEl.textContent = "";
  stateEl.textContent = "listening";
}

/* ---- socket ---- */

let replayGraceUntil = 0;

const socket = new EventSocket(() => hostInput.value, {
  onEvent: (ev) => {
    if (Date.now() < replayGraceUntil) return;   // don't replay history
    void play(ev);
  },
  onStatus: (connected) => {
    statusDot.className = connected ? "dot on" : "dot off";
    statusText.textContent = connected ? "connected" : "reconnecting";
    if (connected) replayGraceUntil = Date.now() + REPLAY_GRACE_MS;
  },
});

hostInput.addEventListener("change", () => {
  saveHost(hostInput.value);
  socket.restart();
});

socket.connect();

})();
