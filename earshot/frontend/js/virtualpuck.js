"use strict";
/* Virtual puck: browser stand-in for the physical alert hardware.
 *
 * Renders the RGB LED and vibration motor with the same urgency profiles,
 * pattern timings, and priority rules as the Pi GPIO controller, so the
 * full experience demos on a bare laptop. No hardware required.
 */
/// <reference path="shared.ts" />
(() => {
    const PROFILES = {
        high: { color: "#D93036", led: { on: 0.12, off: 0.12 },
            motor: [0.5, 0.12, 0.5, 0.12, 1.6] },
        medium: { color: "#DB8B00", led: { on: 0.45, off: 0.25 },
            motor: [0.30] },
        low: { color: "#2456D6", led: { on: 0.20, off: 0.60 },
            motor: [0.12] },
    };
    /* Mirrors backend URGENCY_RANK + the alert controller's latch: a lower
     * urgency never interrupts an active higher one. */
    const RANK = { low: 1, medium: 2, high: 3 };
    const REPLAY_GRACE_MS = 400;
    const IDLE_HOLD_MS = 900; // linger on the label after the pattern
    const hostInput = el("host");
    hostInput.value = resolveHost();
    const statusDot = el("dot");
    const statusText = el("stat");
    const ledEl = el("led");
    const motorEl = el("motor");
    const eventEl = el("evt");
    const stateEl = el("state");
    function buildTimeline(profile) {
        const total = profile.motor.reduce((sum, d) => sum + d, 0);
        const period = profile.led.on + profile.led.off;
        const boundaries = new Set([0, total]);
        let t = 0;
        for (const duration of profile.motor) {
            t += duration;
            boundaries.add(Math.min(t, total));
        }
        for (t = 0; t < total; t += period) {
            boundaries.add(t);
            boundaries.add(Math.min(t + profile.led.on, total));
        }
        const motorOnAt = (instant) => {
            let at = 0;
            let on = true;
            for (const duration of profile.motor) {
                if (instant < at + duration)
                    return on;
                at += duration;
                on = !on;
            }
            return false;
        };
        const ledOnAt = (instant) => (instant % period) < profile.led.on;
        const ordered = [...boundaries].sort((a, b) => a - b);
        const steps = [];
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
    function setLed(color) {
        if (color === null) {
            ledEl.style.background = "";
            ledEl.style.boxShadow = "";
            ledEl.classList.remove("lit");
        }
        else {
            ledEl.style.background = color;
            ledEl.style.boxShadow = `0 0 34px 6px ${color}66`;
            ledEl.classList.add("lit");
        }
    }
    function setMotor(on) {
        motorEl.classList.toggle("buzz", on);
    }
    const sleep = (seconds) => new Promise((resolve) => setTimeout(resolve, seconds * 1000));
    /* ---- the controller: token preemption + priority latch ---- */
    let runToken = 0;
    let activeRank = 0;
    async function play(ev) {
        const rank = RANK[ev.urgency] ?? 2;
        if (rank < activeRank)
            return; // latched: lower can't preempt
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
            if (token !== runToken)
                return; // preempted by an admitted alert
            setLed(step.ledOn ? profile.color : null);
            setMotor(step.motorOn);
            await sleep(step.duration);
        }
        if (token !== runToken)
            return;
        setLed(null);
        setMotor(false);
        await sleep(IDLE_HOLD_MS / 1000);
        if (token !== runToken)
            return;
        activeRank = 0;
        eventEl.textContent = "";
        stateEl.textContent = "listening";
    }
    /* ---- socket ---- */
    let replayGraceUntil = 0;
    const socket = new EventSocket(() => hostInput.value, {
        onEvent: (ev) => {
            if (Date.now() < replayGraceUntil)
                return; // don't replay history
            void play(ev);
        },
        onStatus: (connected) => {
            statusDot.className = connected ? "dot on" : "dot off";
            statusText.textContent = connected ? "connected" : "reconnecting";
            if (connected)
                replayGraceUntil = Date.now() + REPLAY_GRACE_MS;
        },
    });
    hostInput.addEventListener("change", () => {
        saveHost(hostInput.value);
        socket.restart();
    });
    socket.connect();
})();
