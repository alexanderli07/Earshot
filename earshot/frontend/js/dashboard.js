"use strict";
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
    const hostInput = el("host");
    hostInput.value = resolveHost();
    const statusDot = el("dot");
    const statusText = el("stat");
    const feedList = el("feed");
    /* ---- live feed ---- */
    const seenIds = new Set();
    function addEventRow(ev) {
        if (ev.id) {
            if (seenIds.has(ev.id))
                return;
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
        confidence.textContent = `${Math.round((ev.confidence ?? 1) * 100)}%`;
        const source = document.createElement("span");
        source.className = "src";
        source.textContent = ev.source ?? "";
        const chip = document.createElement("span");
        chip.className = `chip ${category}`;
        chip.textContent = category;
        row.append(time, label, confidence, source, chip);
        feedList.prepend(row);
        while (feedList.children.length > FEED_LIMIT)
            feedList.lastChild?.remove();
    }
    /* ---- socket ---- */
    const socket = new EventSocket(() => hostInput.value, {
        onEvent: addEventRow,
        onStatus: (connected) => {
            statusDot.className = connected ? "dot on" : "dot off";
            statusText.textContent = connected ? "connected" : "reconnecting";
            if (connected)
                void loadRules();
        },
    });
    hostInput.addEventListener("change", () => {
        saveHost(hostInput.value);
        seenIds.clear();
        feedList.innerHTML = "";
        socket.restart();
    });
    socket.connect();
    /* ---- teach flow ---- */
    const recordButton = el("rec");
    const teachButton = el("teach");
    const teachNote = el("tnote");
    const nameInput = el("tname");
    const clipMarks = [0, 1, 2].map((i) => el(`d${i}`));
    const clips = [];
    function setNote(text, kind = "") {
        teachNote.className = kind ? `note ${kind}` : "note";
        teachNote.textContent = text;
    }
    function refreshTeachButton() {
        teachButton.disabled =
            !(nameInput.value.trim() && clips.length === CLIPS_REQUIRED);
    }
    recordButton.addEventListener("click", () => {
        void (async () => {
            if (clips.length >= CLIPS_REQUIRED)
                return;
            recordButton.disabled = true;
            setNote("Recording...");
            try {
                clips.push(await recordWav(CLIP_SECONDS));
            }
            catch (err) {
                setNote(`Mic error: ${err.message}`, "err");
                recordButton.disabled = false;
                return;
            }
            clipMarks[clips.length - 1].className = "done";
            setNote(`Captured clip ${clips.length} of ${CLIPS_REQUIRED}`);
            if (clips.length < CLIPS_REQUIRED) {
                recordButton.textContent =
                    `Record clip ${clips.length + 1} of ${CLIPS_REQUIRED}`;
                recordButton.disabled = false;
            }
            else {
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
                const result = await response.json();
                if (response.ok && result.ok) {
                    const names = (result.learned ?? []).map((s) => s.name).join(", ");
                    setNote(`Learned "${nameInput.value.trim()}". Known: ${names}`, "ok");
                    resetTeach();
                    void loadRules();
                }
                else {
                    // Backend reports failures as HTTP errors with a `detail` message
                    // (422 bad clips, 503 ML unavailable, 504 timeout).
                    setNote(`Teach failed: ${result.detail ?? response.statusText}`, "err");
                    teachButton.disabled = false;
                }
            }
            catch (err) {
                setNote(`Network error: ${err.message}`, "err");
            }
        })();
    });
    function resetTeach() {
        clips.length = 0;
        clipMarks.forEach((mark) => { mark.className = ""; });
        recordButton.textContent = "Record clip 1 of 3";
        recordButton.disabled = false;
        teachButton.disabled = true;
        nameInput.value = "";
    }
    /* ---- audio capture: PCM float via WebAudio, encoded as 16-bit WAV so the
     * ML side (stdlib wave reader) can parse it ---- */
    async function recordWav(seconds) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const Ctor = window.AudioContext
            ?? window
                .webkitAudioContext;
        const context = new Ctor();
        const source = context.createMediaStreamSource(stream);
        const processor = context.createScriptProcessor(4096, 1, 1);
        const muted = context.createGain();
        muted.gain.value = 0; // keep the graph alive without mic-to-speaker feedback
        const buffers = [];
        processor.onaudioprocess = (event) => {
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
    const rulesList = el("rules");
    async function loadRules() {
        const base = httpBase(hostInput.value);
        let rules = {};
        let learned = [];
        try {
            rules = await (await fetch(`${base}/rules`)).json();
        }
        catch { /* backend not reachable yet */ }
        try {
            learned = await (await fetch(`${base}/sounds`)).json();
        }
        catch { /* ML may be absent; base sounds still render */ }
        const labels = [...new Set([
                ...BASE_SOUNDS,
                ...learned.map((sound) => canonicalEventLabel(sound.name)),
            ])];
        rulesList.innerHTML = "";
        for (const label of labels) {
            const rule = rules[label] ?? { enabled: true, urgency: null };
            rulesList.appendChild(buildRuleRow(label, rule));
        }
    }
    function buildRuleRow(label, rule) {
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
    async function putRule(label, enabled, urgency) {
        try {
            await fetch(`${httpBase(hostInput.value)}/rules/${encodeURIComponent(label)}`, {
                method: "PUT",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({ enabled, urgency }),
            });
        }
        catch { /* dropped connection; next load re-syncs */ }
    }
})();
