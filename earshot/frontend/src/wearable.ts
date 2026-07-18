/* Wearable page: the phone IS the wearable. Arm (user gesture unlocks
 * vibration + wake lock), then idle until an event arrives; on event,
 * full-screen flash in the urgency color, giant label, vibration pattern. */

/// <reference path="shared.ts" />

(() => {

/* Vibration patterns mirror the Pi motor patterns in backend config. */
const VIBRATION: Record<string, number[]> = {
  high: [500, 120, 500, 120, 1600],
  medium: [300],
  low: [120],
};
const ALERT_HOLD_MS = 3000;
const REPLAY_GRACE_MS = 400;

const armScreen = el<HTMLDivElement>("arm");
const idleScreen = el<HTMLDivElement>("idle");
const alertScreen = el<HTMLDivElement>("alert");
const alertLabel = el<HTMLDivElement>("alertLabel");
const alertCategory = el<HTMLDivElement>("alertCat");
const connectionText = el<HTMLDivElement>("conn");
const armHostInput = el<HTMLInputElement>("armhost");
armHostInput.value = resolveHost();

function show(screen: "arm" | "idle" | "alert"): void {
  armScreen.classList.toggle("hidden", screen !== "arm");
  idleScreen.classList.toggle("hidden", screen !== "idle");
  alertScreen.classList.toggle("hidden", screen !== "alert");
}

/* ---- wake lock (screen must never sleep during the demo) ---- */

async function requestWakeLock(): Promise<void> {
  try {
    if ("wakeLock" in navigator) {
      await navigator.wakeLock.request("screen");
    }
  } catch { /* unsupported or denied; demo still works, screen may dim */ }
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") void requestWakeLock();
});

/* ---- alert display ---- */

let alertTimer: number | undefined;
let replayGraceUntil = 0;

function fireAlert(ev: EarshotEvent): void {
  /* The server replays recent events on connect; don't flash on history. */
  if (Date.now() < replayGraceUntil) return;

  const category = categoryOf(ev);
  alertScreen.style.background = CATEGORY_COLOR[category];
  alertLabel.textContent = prettyLabel(ev.label);
  alertCategory.textContent = category;
  show("alert");
  alertScreen.classList.remove("flash");
  void alertScreen.offsetWidth;          // restart the CSS animation
  alertScreen.classList.add("flash");

  if (navigator.vibrate) {
    navigator.vibrate(VIBRATION[ev.urgency] ?? VIBRATION.medium);
  }
  window.clearTimeout(alertTimer);
  alertTimer = window.setTimeout(() => show("idle"), ALERT_HOLD_MS);
}

/* ---- socket ---- */

const socket = new EventSocket(() => armHostInput.value, {
  onEvent: fireAlert,
  onStatus: (connected) => {
    connectionText.textContent = connected ? "connected" : "reconnecting";
    connectionText.classList.toggle("off", !connected);
    if (connected) replayGraceUntil = Date.now() + REPLAY_GRACE_MS;
  },
});

/* ---- arm: the one required user tap ---- */

el<HTMLButtonElement>("armBtn").addEventListener("click", () => {
  void (async () => {
    saveHost(armHostInput.value);
    if (navigator.vibrate) navigator.vibrate(60);  // prime inside the gesture
    await requestWakeLock();
    show("idle");
    socket.connect();
  })();
});

})();
