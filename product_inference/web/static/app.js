/* Yojana Mitra — web chat surface.
 *
 * Consumes the same event stream every other surface renders. SSE is read with
 * fetch + ReadableStream rather than EventSource, because EventSource is GET-only
 * and a turn is a POST.
 */

const $ = (id) => document.getElementById(id);

const log = $("log");
const statusBar = $("status");
const statusText = $("status-text");
const gate = $("gate");
const gateText = $("gate-text");
const trace = $("trace");
const shots = $("shots");
const input = $("input");
const sendBtn = $("btn-send");

let busy = false;

/* ── Rendering helpers ───────────────────────────────────────── */

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/* Deliberately tiny. The text comes from an LLM, so it is escaped first and only
 * then given structure — never the other way round. */
function renderMarkdown(text) {
  let html = escapeHtml(text);

  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, _lang, code) =>
    `<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  return html
    .split(/\n{2,}/)
    .map((block) => (block.startsWith("<pre>") ? block : `<p>${block.replace(/\n/g, "<br>")}</p>`))
    .join("");
}

function addMessage(who, text, kind) {
  const row = document.createElement("div");
  row.className = "msg " + (who === "user" ? "user" : "bot") + (kind === "error" ? " error" : "");

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = who === "user" ? "🙂" : kind === "error" ? "⚠️" : "🇮🇳";

  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = renderMarkdown(text);

  row.append(avatar, body);
  log.append(row);
  log.scrollTop = log.scrollHeight;
  return body;
}

function setStatus(text) {
  if (!text) { statusBar.classList.add("hidden"); return; }
  statusText.textContent = text.replace(/[*_`]/g, "");
  statusBar.classList.remove("hidden");
}

function newTurnInTrace() {
  if (trace.querySelector(".empty")) trace.innerHTML = "";
  trace.querySelectorAll(".node.hot").forEach((n) => n.classList.remove("hot"));
  if (trace.children.length) {
    const sep = document.createElement("div");
    sep.className = "turn-sep";
    trace.append(sep);
  }
}

function addNode(name) {
  const el = document.createElement("div");
  el.className = "node hot";
  el.textContent = name;
  trace.append(el);
  trace.scrollTop = trace.scrollHeight;
}

function addScreenshot(name, caption) {
  if (shots.querySelector(".empty")) shots.innerHTML = "";
  const url = `/api/screenshot/${encodeURIComponent(name)}?ts=${Date.now()}`;
  const img = document.createElement("img");
  img.src = url;
  img.alt = caption || "portal screenshot";
  img.onclick = () => window.open(url, "_blank", "noopener");
  shots.prepend(img);
  return url;
}

function setBusy(state) {
  busy = state;
  sendBtn.disabled = state;
  sendBtn.textContent = state ? "…" : "Send";
}

/* ── The event stream ────────────────────────────────────────── */

async function consume(response) {
  if (response.status === 401) {
    location.reload();
    return;
  }
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    addMessage("bot", detail, "error");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let lastBody = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      if (!frame.startsWith("data: ")) continue;

      let ev;
      try { ev = JSON.parse(frame.slice(6)); } catch (_) { continue; }

      switch (ev.kind) {
        case "status":
          setStatus(ev.text);
          break;

        case "nodeenter":
          addNode(ev.node);
          break;

        case "message":
          setStatus("");
          lastBody = addMessage("bot", ev.text);
          break;

        case "image": {
          const url = addScreenshot(ev.path, ev.caption);
          const target = lastBody || addMessage("bot", "");
          const img = document.createElement("img");
          img.src = url;
          img.alt = ev.caption || "portal screenshot";
          target.append(img);
          log.scrollTop = log.scrollHeight;
          break;
        }

        case "awaitconfirm":
          gateText.textContent = ev.prompt || "Ready to submit.";
          gate.classList.remove("hidden");
          break;

        case "profilerequired":
          setStatus("");
          addMessage("bot", ev.text || "Please fill your profile first.");
          openProfile();
          break;

        case "error":
          setStatus("");
          addMessage("bot", ev.text, "error");
          if (ev.detail) console.warn("[yojana]", ev.detail);
          break;

        case "cancelled":
          setStatus("");
          addMessage("bot", ev.text || "🛑 Request cancelled.");
          break;

        case "end":
          setStatus("");
          break;
      }
    }
  }
  setStatus("");
}

/* ── Actions ─────────────────────────────────────────────────── */

async function send(text) {
  if (busy || !text.trim()) return;
  addMessage("user", text);
  gate.classList.add("hidden");
  newTurnInTrace();
  setBusy(true);
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    await consume(res);
  } catch (err) {
    addMessage("bot", "Lost connection to the server.", "error");
    console.error(err);
  } finally {
    setBusy(false);
  }
}

async function upload(file) {
  if (busy) return;
  addMessage("user", `📎 ${file.name}`);
  newTurnInTrace();
  setBusy(true);
  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/upload", { method: "POST", body: form });
    await consume(res);
  } catch (err) {
    addMessage("bot", "Upload failed.", "error");
    console.error(err);
  } finally {
    setBusy(false);
  }
}

/* ── Profile modal ───────────────────────────────────────────── */

const overlay = $("profile-overlay");
const fieldsBox = $("profile-fields");
let formSchema = [];

async function openProfile() {
  overlay.classList.remove("hidden");
  $("profile-msg").textContent = "";
  fieldsBox.innerHTML = "<div class='empty'>Loading…</div>";

  const res = await fetch("/api/profile");
  if (res.status === 401) { location.reload(); return; }
  if (!res.ok) { fieldsBox.innerHTML = "<div class='empty'>Profile store unavailable.</div>"; return; }

  const data = await res.json();
  formSchema = data.schema;
  fieldsBox.innerHTML = "";

  for (const f of formSchema) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    wrap.dataset.field = f.field;

    const label = document.createElement("label");
    label.textContent = f.question;
    label.htmlFor = `f-${f.field}`;

    let control;
    const saved = data.profile[f.field];

    if (f.kind === "choice" && !f.allows_free_text) {
      control = document.createElement("select");
      control.innerHTML = `<option value="">—</option>` + f.options
        .map((o) => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`)
        .join("");
      if (saved !== undefined && saved !== null) control.value = String(saved);
    } else {
      control = document.createElement("input");
      control.type = f.numeric ? "number" : "text";
      if (f.options.length) control.setAttribute("list", `dl-${f.field}`);
      if (saved !== undefined && saved !== null) control.value = saved;
    }
    control.id = `f-${f.field}`;

    const err = document.createElement("div");
    err.className = "err";

    wrap.append(label, control);
    if (f.kind === "choice" && f.allows_free_text) {
      const dl = document.createElement("datalist");
      dl.id = `dl-${f.field}`;
      dl.innerHTML = f.options.map((o) => `<option value="${escapeHtml(o.value)}">`).join("");
      wrap.append(dl);
    }
    wrap.append(err);
    fieldsBox.append(wrap);
  }
}

async function saveProfile() {
  const profile = {};
  for (const f of formSchema) {
    const el = $(`f-${f.field}`);
    if (el && el.value !== "") profile[f.field] = el.value;
  }

  fieldsBox.querySelectorAll(".field").forEach((w) => w.classList.remove("bad"));
  $("profile-msg").textContent = "Saving…";

  const res = await fetch("/api/profile", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile }),
  });
  const data = await res.json().catch(() => ({}));

  if (res.ok && data.ok) {
    overlay.classList.add("hidden");
    addMessage("bot", data.summary);
    return;
  }

  $("profile-msg").textContent = "";
  if (data.errors) {
    for (const [field, message] of Object.entries(data.errors)) {
      const wrap = fieldsBox.querySelector(`.field[data-field="${field}"]`);
      if (wrap) { wrap.classList.add("bad"); wrap.querySelector(".err").textContent = message; }
    }
  } else if (data.problems) {
    $("profile-msg").textContent = data.problems.join("; ");
  } else {
    $("profile-msg").textContent = data.detail || "Could not save.";
  }
}

/* ── Wiring ──────────────────────────────────────────────────── */

$("composer").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value;
  input.value = "";
  input.style.height = "auto";
  send(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("composer").requestSubmit();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
});

$("file").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) upload(file);
  e.target.value = "";
});

$("btn-confirm").addEventListener("click", () => {
  gate.classList.add("hidden");
  // Goes through the ordinary chat path as literal text. The browser button is a
  // convenience for typing CONFIRM — it is not a privileged route, and the graph's
  // interrupt/resume gate is still what decides whether the submit happens.
  send("CONFIRM");
});

$("btn-dismiss").addEventListener("click", () => gate.classList.add("hidden"));

$("btn-profile").addEventListener("click", openProfile);
$("btn-profile-close").addEventListener("click", () => overlay.classList.add("hidden"));
$("btn-profile-save").addEventListener("click", saveProfile);
overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.classList.add("hidden"); });

$("btn-reset").addEventListener("click", async () => {
  if (!confirm("Wipe your profile, documents and conversation history? This cannot be undone.")) return;
  const res = await fetch("/api/reset", { method: "POST" });
  if (res.ok) {
    log.innerHTML = "";
    trace.innerHTML = "<div class='empty'>Nodes light up here as the state machine runs.</div>";
    shots.innerHTML = "<div class='empty'>Portal screenshots appear here during an application.</div>";
    addMessage("bot", "Everything has been wiped. Fill your profile to start again.");
  } else {
    addMessage("bot", "Reset failed.", "error");
  }
});

$("btn-panel").addEventListener("click", () => $("shell").classList.toggle("show-side"));

/* Boot */
(async () => {
  try {
    const res = await fetch("/api/me");
    if (res.status === 401) { location.reload(); return; }
    const me = await res.json();
    $("who").textContent = me.user_id;
  } catch (_) { /* header detail is cosmetic */ }

  if (window.matchMedia("(max-width: 900px)").matches) {
    $("btn-panel").style.display = "";
  }

  addMessage("bot",
    "Namaste! 🙏 I can help you find government welfare schemes you're eligible for, " +
    "read your documents, and fill out application portals for you.\n\n" +
    "Nothing is ever submitted without your explicit confirmation.\n\n" +
    "Try **find schemes for me**, or upload a document with 📎.");
})();
