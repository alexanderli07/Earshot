/* Login / register page. Posts to /auth/login or /auth/register (same origin
 * when served from the backend at /ui/), then redirects to the dashboard.
 * The session is an httpOnly cookie set by the backend — never touched here. */

/// <reference path="shared.ts" />

(() => {

const form = el<HTMLFormElement>("form");
const submit = el<HTMLButtonElement>("submit");
const note = el<HTMLDivElement>("note");
const username = el<HTMLInputElement>("username");
const displayName = el<HTMLInputElement>("display_name");
const password = el<HTMLInputElement>("password");
const tabLogin = el<HTMLButtonElement>("tab-login");
const tabRegister = el<HTMLButtonElement>("tab-register");

let mode: "login" | "register" = "login";

function setMode(next: "login" | "register"): void {
  mode = next;
  document.body.classList.toggle("mode-register", next === "register");
  tabLogin.classList.toggle("active", next === "login");
  tabRegister.classList.toggle("active", next === "register");
  submit.textContent = next === "login" ? "Sign in" : "Create account";
  password.autocomplete = next === "login" ? "current-password" : "new-password";
  note.textContent = "";
  note.className = "note";
}

tabLogin.addEventListener("click", () => setMode("login"));
tabRegister.addEventListener("click", () => setMode("register"));

/* Pages are served same-origin from the backend, so the session cookie flows
 * without CORS credentials. A ?host= override targets a different backend. */
function apiBase(): string {
  const host = new URLSearchParams(location.search).get("host");
  return host ? httpBase(host) : "";
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void (async () => {
    submit.disabled = true;
    note.className = "note";
    note.textContent = mode === "login" ? "Signing in..." : "Creating account...";

    const payload: Record<string, string> = {
      username: username.value.trim(),
      password: password.value,
    };
    if (mode === "register" && displayName.value.trim()) {
      payload.display_name = displayName.value.trim();
    }

    try {
      const response = await fetch(`${apiBase()}/auth/${mode}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify(payload),
      });
      if (response.ok) {
        note.className = "note ok";
        note.textContent = "Success — redirecting...";
        const host = new URLSearchParams(location.search).get("host");
        const suffix = host ? `?host=${encodeURIComponent(host)}` : "";
        location.href = `dashboard.html${suffix}`;
        return;
      }
      const body = await response.json().catch(() => ({}));
      note.className = "note err";
      note.textContent = response.status === 503
        ? "Accounts aren't enabled on this server (no database configured)."
        : (body.detail ?? "Something went wrong.");
    } catch (err) {
      note.className = "note err";
      note.textContent = `Network error: ${(err as Error).message}`;
    } finally {
      submit.disabled = false;
    }
  })();
});

})();
