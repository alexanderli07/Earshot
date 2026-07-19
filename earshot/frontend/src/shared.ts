/* Shared between dashboard and wearable: event types, palette mapping,
 * backend host resolution, and a reconnecting WebSocket.
 *
 * Compiled as classic scripts (no modules) so the pages work over file://
 * and same-origin from the Pi with zero tooling at runtime. */

interface EarshotEvent {
  id?: string;
  label: string;
  urgency: "high" | "medium" | "low" | string;
  confidence?: number;
  source?: "pretrained" | "trained" | "taught" | "debug" | string;
  timestamp?: number;
  received_at?: number;
}

type Category = "urgent" | "presence" | "appliance" | "taught";

/* Palette per the pitch: red urgent, blue someone-is-here, amber appliance,
 * green taught. */
const CATEGORY_BY_LABEL: Record<string, Category> = {
  smoke_alarm: "urgent",
  baby_cry: "urgent",
  glass_break: "urgent",
  doorbell: "presence",
  knock: "presence",
};

/* Colors match the pitch page: alarm red, door blue, appliance amber,
 * taught green. */
const CATEGORY_COLOR: Record<Category, string> = {
  urgent: "#D93036",
  presence: "#2456D6",
  appliance: "#DB8B00",
  taught: "#178A50",
};

const LEGACY_EVENT_LABELS: Record<string, string> = {
  fire_alarm: "smoke_alarm",
  fire_smoke_alarm: "smoke_alarm",
};

function canonicalEventLabel(label: string): string {
  return LEGACY_EVENT_LABELS[label] ?? label;
}

function categoryOf(ev: EarshotEvent): Category {
  if (ev.source === "taught") return "taught";
  const mapped = CATEGORY_BY_LABEL[canonicalEventLabel(ev.label)];
  if (mapped) return mapped;
  if (ev.urgency === "high") return "urgent";
  if (ev.urgency === "low") return "appliance";
  return "presence";
}

function prettyLabel(label: string): string {
  return canonicalEventLabel(label || "").replace(/_/g, " ");
}

function clockTime(ts?: number): string {
  const ms = (ts ?? Date.now() / 1000) * 1000;
  return new Date(ms).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

/* ---- backend host: ?host= param > saved > page origin ---- */

const HOST_KEY = "earshot_host";

function resolveHost(): string {
  const fromQuery = new URLSearchParams(location.search).get("host");
  return fromQuery || localStorage.getItem(HOST_KEY) || location.host || "";
}

function saveHost(host: string): void {
  localStorage.setItem(HOST_KEY, host);
}

function httpBase(host: string): string {
  const scheme = location.protocol === "https:" ? "https" : "http";
  return `${scheme}://${host}`;
}

function wsUrl(host: string): string {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${host}/ws`;
}

/* ---- reconnecting WebSocket (phones love to drop connections) ---- */

interface SocketHandlers {
  onEvent: (ev: EarshotEvent) => void;
  onStatus: (connected: boolean) => void;
}

class EventSocket {
  private ws: WebSocket | null = null;
  private retries = 0;
  private closedByUser = false;

  constructor(private getHost: () => string,
              private handlers: SocketHandlers) {}

  connect(): void {
    this.closedByUser = false;
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl(this.getHost()));
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.retries = 0;
      this.handlers.onStatus(true);
    };
    ws.onmessage = (msg: MessageEvent) => {
      try {
        this.handlers.onEvent(JSON.parse(msg.data as string) as EarshotEvent);
      } catch {
        /* ignore malformed frames */
      }
    };
    ws.onclose = () => {
      this.handlers.onStatus(false);
      if (!this.closedByUser) this.scheduleReconnect();
    };
    ws.onerror = () => ws.close();
  }

  restart(): void {
    this.closedByUser = true;
    try { this.ws?.close(); } catch { /* already closed */ }
    this.retries = 0;
    this.connect();
  }

  private scheduleReconnect(): void {
    this.retries = Math.min(this.retries + 1, 6);
    setTimeout(() => this.connect(), 400 * this.retries);
  }
}

/* ---- tiny DOM helper: getElementById that throws instead of null ---- */

function el<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

/* ---- 16-bit PCM mono WAV encoder. The teach endpoint hands these to the ML
 * side, whose stdlib wave reader needs plain PCM — keep this boring. ---- */

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i++) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };
  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true);         // fmt chunk size
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);          // block align
  view.setUint16(34, 16, true);         // bits per sample
  writeAscii(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff,
                  true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}
