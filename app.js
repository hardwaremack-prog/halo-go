// Halo Go — offline voice AI that runs entirely on an iPhone (Safari / iOS 26+, WebGPU).
// Ears: Whisper (transformers.js, in a worker)  Brain: WebLLM (WebGPU, in a worker)  Voice: iOS speech.
const VERSION = "1.0.1";
const WEBLLM = "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm";
const DEMO = new URLSearchParams(location.search).has("demo");

const MODELS = [
  { key: "quick", name: "Quick", desc: "Llama 3.2 1B · about 0.9 GB · fastest replies",
    f16: "Llama-3.2-1B-Instruct-q4f16_1-MLC", f32: "Llama-3.2-1B-Instruct-q4f32_1-MLC", gb: 0.9 },
  { key: "balanced", name: "Balanced", desc: "Gemma 2 2B · about 1.6 GB · smarter, a little slower",
    f16: "gemma-2-2b-it-q4f16_1-MLC-1k", f32: "gemma-2-2b-it-q4f32_1-MLC-1k", gb: 1.6, noSystem: true },
  { key: "big", name: "Smartest", desc: "Llama 3.2 3B · about 2.3 GB · may be too big for an iPhone 14",
    f16: "Llama-3.2-3B-Instruct-q4f16_1-MLC", f32: "Llama-3.2-3B-Instruct-q4f32_1-MLC", gb: 2.3 },
];
const EARS = [
  { id: "Xenova/whisper-tiny.en", name: "Quick", desc: "Whisper tiny · about 40 MB" },
  { id: "Xenova/whisper-base.en", name: "Accurate", desc: "Whisper base · about 80 MB" },
];
const DEFAULT_PROMPT = "You are Halo Go, a calm, friendly and clever voice assistant that runs entirely on the user's iPhone, with no internet. Keep answers short and natural to listen to, usually one to three sentences. Speak in plain sentences: no lists, markdown or emoji unless asked. If you're not sure, say so.";

const $ = (id) => document.getElementById(id);

// ---------------- Settings (kept on this iPhone) ----------------
const ls = {
  get(k, d) { try { const v = localStorage.getItem("hg_" + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("hg_" + k, JSON.stringify(v)); } catch {} },
};
const S = {
  model: ls.get("model", "quick"),
  ears: ls.get("ears", EARS[0].id),
  voice: ls.get("voice", ""),
  speed: ls.get("speed", 1.0),
  speak: ls.get("speak", true),
  mode: ls.get("mode", "ptt"),
  thr: ls.get("thr", 0.02),
  sil: ls.get("sil", 900),
  prompt: ls.get("prompt", DEFAULT_PROMPT),
};
const save = (k, v) => { S[k] = v; ls.set(k, v); };
let history = ls.get("history", []);

// ---------------- State ----------------
let state = "off";
let hasF16 = true;
let engine = null, engineModel = null;
let sttWorker = null, sttReq = 0;
const sttWait = new Map();
let ctx = null, stream = null, src = null, proc = null;
let recording = false, rec = [], preRoll = [], pttHeld = false;
let hf = { on: false, loud: 0, sil: 0, dur: 0 };
let micLevel = 0;
let replyId = 0, generating = false;
let speakQueue = 0;

// Safari can't read back WebLLM's default (Cache API) storage for Hugging Face files,
// so keep the brain in IndexedDB instead.
let appConfig = null;
async function getAppConfig() {
  if (!appConfig) { const webllm = await import(WEBLLM); appConfig = { ...webllm.prebuiltAppConfig, cacheBackend: "indexeddb" }; }
  return appConfig;
}
const modelInfo = (key) => MODELS.find((m) => m.key === key) || MODELS[0];
const modelId = (key) => { const m = modelInfo(key); return hasF16 ? m.f16 : m.f32; };

// ---------------- UI helpers ----------------
function setState(s, label) {
  state = s;
  $("avatar").className = "avatar state-" + (s === "transcribing" ? "thinking" : s);
  const L = { off: "Tap start", idle: S.mode === "hf" ? "Listening for you…" : "Ready", listening: "Listening…",
    transcribing: "Got it…", thinking: "Thinking…", speaking: "Speaking…", loading: "Loading…" };
  $("stateLabel").textContent = label || L[s] || "";
  $("talkBtn").classList.toggle("active", s === "listening");
  // Give the speaker back to the voice while Halo talks; listen again afterwards
  if (s === "idle" || s === "listening") ensureMic(); else if (s === "thinking" || s === "speaking") releaseMic();
}
const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function render(t) {
  return esc(t).replace(/```\w*\n?([\s\S]*?)```/g, "<pre>$1</pre>").replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");
}
function bubble(role, text, cls = "") {
  $("empty").style.display = "none";
  const d = document.createElement("div");
  d.className = `msg ${role} ${cls}`;
  d.innerHTML = render(text);
  $("transcript").appendChild(d);
  $("transcript").scrollTop = 1e9;
  return d;
}
const note = (t) => bubble("note", t);
const err = (t) => bubble("error", t);
function renderHistory() {
  $("transcript").querySelectorAll(".msg").forEach((n) => n.remove());
  history.forEach((m) => bubble(m.role, m.content));
  $("empty").style.display = history.length ? "none" : "";
}

// ---------------- Brain (WebLLM on the GPU) ----------------
async function loadBrain(key, onProgress) {
  const id = modelId(key);
  if (DEMO) { await fakeProgress(onProgress, 1200); engine = makeDemoEngine(); engineModel = id; return; }
  const webllm = await import(WEBLLM);
  ls.set("loading", key);                         // if iOS runs out of memory, we'll know next launch
  const cb = (r) => onProgress && onProgress(r.progress || 0, r.text || "");
  try {
    if (!engine) {
      const w = new Worker(new URL("./llm-worker.js", import.meta.url), { type: "module" });
      engine = await webllm.CreateWebWorkerMLCEngine(w, id, { initProgressCallback: cb, appConfig: await getAppConfig() });
    } else {
      engine.setInitProgressCallback?.(cb);
      await engine.reload(id);
    }
    engineModel = id;
  } finally {
    ls.set("loading", null);
  }
}
async function brainCached(key) {
  if (DEMO) return ls.get("demo_cached", false);
  try { const webllm = await import(WEBLLM); return await webllm.hasModelInCache(modelId(key), await getAppConfig()); } catch { return false; }
}

// ---------------- Ears (Whisper in a worker) ----------------
function startEars(onProgress) {
  if (DEMO) return fakeProgress(onProgress, 600);
  return new Promise((resolve, reject) => {
    sttWorker = new Worker(new URL("./stt-worker.js", import.meta.url), { type: "module" });
    sttWorker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "progress") onProgress && onProgress(m.total ? m.loaded / m.total : 0, m.total);
      else if (m.type === "ready") resolve();
      else if (m.type === "result" || m.type === "error") {
        const w = sttWait.get(m.reqId);
        if (w) { sttWait.delete(m.reqId); m.type === "result" ? w.res(m.text) : w.rej(new Error(m.error)); }
        else if (m.type === "error") reject(new Error(m.error));
      }
    };
    sttWorker.onerror = (e) => reject(new Error(e.message || "Speech recognizer failed to start"));
    sttWorker.postMessage({ type: "load", id: S.ears });
  });
}
function transcribe(audio) {
  if (DEMO) return new Promise((r) => setTimeout(() => r("Hello from the demo microphone"), 400));
  const reqId = ++sttReq;
  return new Promise((res, rej) => {
    sttWait.set(reqId, { res, rej });
    sttWorker.postMessage({ type: "transcribe", reqId, audio, model: S.ears }, [audio.buffer]);
  });
}

// ---------------- Microphone ----------------
let micOpening = null;
async function ensureMic() {
  if (stream || !ctx) return;
  if (micOpening) return micOpening;
  micOpening = (async () => {
    try { if (navigator.audioSession) navigator.audioSession.type = "play-and-record"; } catch {}
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    if (ctx.state !== "running") await ctx.resume().catch(() => {});
    src = ctx.createMediaStreamSource(stream);
    proc = ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = onFrame;
    src.connect(proc); proc.connect(ctx.destination);
  })();
  try { await micOpening; } catch (e) { err("Couldn't use the microphone: " + e.message); } finally { micOpening = null; }
}
function releaseMic() {
  if (!stream || recording) return;
  try { proc.disconnect(); src.disconnect(); } catch {}
  stream.getTracks().forEach((t) => t.stop());
  stream = src = proc = null;
  micLevel = 0; preRoll = [];
  try { if (navigator.audioSession) navigator.audioSession.type = "playback"; } catch {}
}

function onFrame(e) {
  const d = e.inputBuffer.getChannelData(0);
  let sum = 0; for (let i = 0; i < d.length; i++) sum += d[i] * d[i];
  const rms = Math.sqrt(sum / d.length);
  micLevel = micLevel * 0.6 + rms * 0.4;
  const copy = new Float32Array(d);
  const ms = (d.length / ctx.sampleRate) * 1000;
  if (recording) {
    rec.push(copy);
    if (rec.length * d.length > ctx.sampleRate * 30) stopRec(true);
  } else {
    preRoll.push(copy); if (preRoll.length > 3) preRoll.shift();
  }
  if (S.mode !== "hf" || pttHeld || (state !== "idle" && state !== "listening")) return;
  if (!hf.on) {
    hf.loud = rms > S.thr ? hf.loud + 1 : 0;
    if (hf.loud >= 2) { hf.on = true; hf.sil = 0; hf.dur = 0; startRec(); }
  } else {
    hf.dur += ms;
    hf.sil = rms < S.thr * 0.75 ? hf.sil + ms : 0;
    if (hf.sil >= S.sil) { hf.on = false; hf.loud = 0; stopRec(hf.dur - hf.sil > 350); }
  }
}
function startRec() { rec = preRoll.slice(); preRoll = []; recording = true; setState("listening"); }
function stopRec(send) {
  if (!recording) return;
  recording = false;
  const chunks = rec; rec = [];
  const n = chunks.reduce((a, c) => a + c.length, 0);
  if (!send || n < ctx.sampleRate * 0.35) { setState("idle"); return; }
  const audio = to16k(chunks, n, ctx.sampleRate);
  setState("transcribing");
  transcribe(audio).then((text) => {
    if (!text) { setState("idle", "Didn't catch that"); setTimeout(() => state === "idle" && setState("idle"), 1500); return; }
    send_(text);
  }).catch((e) => { err("Couldn't understand the audio: " + e.message); setState("idle"); });
}
function to16k(chunks, n, rate) {
  const all = new Float32Array(n); let o = 0;
  for (const c of chunks) { all.set(c, o); o += c.length; }
  const ratio = rate / 16000, m = Math.floor(n / ratio), out = new Float32Array(m);
  for (let i = 0; i < m; i++) {
    const s = Math.floor(i * ratio), e = Math.min(n, Math.floor((i + 1) * ratio));
    let a = 0; for (let j = s; j < e; j++) a += all[j];
    out[i] = a / Math.max(1, e - s);
  }
  return out;
}

// ---------------- Chat ----------------
async function send_(text) {
  stopReply();
  if (!engine) { err("The brain isn't loaded yet."); return; }
  const id = ++replyId;
  history.push({ role: "user", content: text });
  bubble("user", text);
  saveHistory();
  setState("thinking");
  const b = bubble("assistant", "", "typing");
  let full = "", pending = "";
  const t0 = performance.now();
  generating = true;
  try {
    const info = modelInfo(S.model);
    const recent = history.slice(-10);
    let msgs = recent.map((m) => ({ role: m.role, content: m.content }));
    const sys = S.prompt.trim() + " Your words are read aloud, so keep them easy to listen to.";
    if (info.noSystem) {
      const first = msgs.findIndex((m) => m.role === "user");
      if (first >= 0) msgs[first] = { role: "user", content: sys + "\n\n" + msgs[first].content };
      while (msgs.length && msgs[0].role !== "user") msgs.shift();
    } else msgs = [{ role: "system", content: sys }, ...msgs];
    const chunks = await engine.chat.completions.create({ messages: msgs, stream: true, temperature: 0.7, max_tokens: 320 });
    for await (const c of chunks) {
      if (id !== replyId) break;
      const piece = c.choices?.[0]?.delta?.content || "";
      if (!piece) continue;
      full += piece; pending += piece;
      b.innerHTML = render(full.trimStart());
      $("transcript").scrollTop = 1e9;
      pending = flush(pending, false, id);
    }
    if (id !== replyId) return;
    flush(pending, true, id);
    b.classList.remove("typing");
    b.insertAdjacentHTML("beforeend", `<span class="meta">${esc(modelInfo(S.model).name)} · ${((performance.now() - t0) / 1000).toFixed(1)}s</span>`);
    if (full.trim()) { history.push({ role: "assistant", content: full.trim() }); saveHistory(); }
  } catch (e) {
    b.classList.remove("typing");
    if (!full) b.remove();
    err("The brain hit a problem: " + (e.message || e));
  } finally {
    generating = false;
    if (id === replyId && speakQueue === 0) setState("idle");
  }
}
function saveHistory() { history = history.slice(-40); ls.set("history", history); }

// Speak sentence by sentence as the reply streams in
function flush(text, final, id) {
  if (!S.speak) return "";
  let out = text.replace(/```[\s\S]*?```/g, " I put that on the screen. ");
  if (/```/.test(out) && !final) return out;
  const re = /^([\s\S]*?[.!?…:;])(\s+)/;
  let m;
  while ((m = re.exec(out))) { say(m[1], id); out = out.slice(m[0].length); }
  if (!final && out.length > 160) { const c = out.lastIndexOf(", ", 140); if (c > 40) { say(out.slice(0, c + 1), id); out = out.slice(c + 2); } }
  if (final && out.trim()) { say(out, id); out = ""; }
  return out;
}
function say(text, id) {
  const clean = text.replace(/[*_#`>~|]/g, "").replace(/https?:\/\/\S+/g, "a link").trim();
  if (!clean.replace(/[^\p{L}\p{N}]/gu, "") || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(clean);
  const v = speechSynthesis.getVoices().find((x) => x.voiceURI === S.voice);
  if (v) u.voice = v;
  u.rate = S.speed; u.lang = v ? v.lang : "en-US";
  speakQueue++;
  u.onstart = () => { if (id === replyId) setState("speaking"); };
  u.onboundary = () => pulse();
  u.onend = u.onerror = () => {
    speakQueue = Math.max(0, speakQueue - 1);
    if (speakQueue === 0 && !generating && id === replyId) setState("idle");
  };
  speechSynthesis.speak(u);
}
let pulseLvl = 0;
const pulse = () => { pulseLvl = 1; };
function stopReply() {
  replyId++;
  if (generating && engine) { try { engine.interruptGenerate(); } catch {} }
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  speakQueue = 0;
  if (state === "speaking" || state === "thinking") setState("idle");
}

// ---------------- Animation ----------------
function animate() {
  let lvl = 0;
  if (state === "speaking") { pulseLvl *= 0.9; lvl = 0.25 + pulseLvl * 0.6 + Math.abs(Math.sin(performance.now() / 160)) * 0.15; }
  else if (state === "idle" || state === "listening") lvl = Math.min(1, micLevel * 9);
  $("avatar").style.setProperty("--lvl", lvl.toFixed(3));
  $("micMeter").style.width = Math.min(100, Math.sqrt(micLevel / 0.2) * 100) + "%";
  requestAnimationFrame(animate);
}

// ---------------- Controls ----------------
async function pttDown(e) {
  e && e.preventDefault();
  if (state === "off" || state === "loading" || pttHeld) return;
  pttHeld = true;
  if (state === "speaking" || state === "thinking") stopReply();
  await ensureMic();
  if (pttHeld) { hf.on = false; startRec(); }
}
function pttUp() { if (!pttHeld) return; pttHeld = false; stopRec(true); }
const tb = $("talkBtn");
tb.addEventListener("pointerdown", (e) => { tb.setPointerCapture?.(e.pointerId); pttDown(e); });
tb.addEventListener("pointerup", pttUp);
tb.addEventListener("pointercancel", pttUp);
tb.addEventListener("contextmenu", (e) => e.preventDefault());
document.addEventListener("keydown", (e) => { if (e.code === "Space" && document.activeElement?.tagName !== "INPUT" && !e.repeat) pttDown(e); });
document.addEventListener("keyup", (e) => { if (e.code === "Space" && document.activeElement?.tagName !== "INPUT") pttUp(); });

function setMode(m, persist = true) {
  if (persist) save("mode", m);
  else S.mode = m;
  $("modePtt").setAttribute("aria-checked", m !== "hf");
  $("modeHf").setAttribute("aria-checked", m === "hf");
  $("talkTxt").textContent = m === "hf" ? "Hands-free on · hold to talk anyway" : "Hold to talk";
  tb.classList.toggle("hf", m === "hf");
  hf.on = false;
  if (state === "idle") setState("idle");
}
$("modePtt").onclick = () => setMode("ptt");
$("modeHf").onclick = () => setMode("hf");
$("stopBtn").onclick = () => stopReply();
$("typeForm").onsubmit = (e) => { e.preventDefault(); const t = $("typeBox").value.trim(); if (!t || state === "off") return; $("typeBox").value = ""; $("typeBox").blur(); send_(t); };
$("newChat").onclick = () => { stopReply(); history = []; saveHistory(); renderHistory(); };

// ---------------- Settings sheet ----------------
const openSheet = () => { $("sheet").classList.add("open"); $("scrim").classList.add("show"); refreshSheet(); };
const closeSheet = () => { $("sheet").classList.remove("open"); $("scrim").classList.remove("show"); };
$("gear").onclick = openSheet; $("closeSheet").onclick = closeSheet; $("scrim").onclick = closeSheet;

async function refreshSheet() {
  const ml = $("modelList"); ml.innerHTML = "";
  for (const m of MODELS) {
    const cached = await brainCached(m.key);
    const b = document.createElement("button");
    b.className = "choice" + (m.key === S.model ? " on" : "");
    b.innerHTML = `${cached ? '<span class="tag">downloaded</span>' : ""}<b>${m.name}</b><span>${m.desc}</span>`;
    b.onclick = () => switchBrain(m.key);
    ml.appendChild(b);
  }
  $("modelNote").textContent = hasF16 ? "" : "This iPhone's browser lacks 16-bit GPU math, so slightly larger versions are used.";
  const el = $("earsList"); el.innerHTML = "";
  for (const x of EARS) {
    const b = document.createElement("button");
    b.className = "choice" + (x.id === S.ears ? " on" : "");
    b.innerHTML = `<b>${x.name}</b><span>${x.desc}</span>`;
    b.onclick = () => switchEars(x.id);
    el.appendChild(b);
  }
  fillVoices();
  $("speed").value = S.speed; $("speedVal").textContent = (+S.speed).toFixed(2) + "×";
  $("speak").checked = S.speak;
  $("thr").value = S.thr; $("thrVal").textContent = (+S.thr).toFixed(3);
  $("sil").value = S.sil; $("silVal").textContent = (S.sil / 1000).toFixed(1) + " s";
  $("sysPrompt").value = S.prompt;
  $("micThr").style.left = Math.min(100, Math.sqrt(S.thr / 0.2) * 100) + "%";
  try {
    const est = await navigator.storage.estimate();
    const persisted = await navigator.storage.persisted?.();
    $("storageInfo").textContent = `Using ${(est.usage / 1e9).toFixed(2)} GB on this iPhone.` + (persisted ? " Protected from automatic clean-up." : "");
  } catch { $("storageInfo").textContent = ""; }
  $("about").textContent = `Halo Go ${VERSION} · runs fully on this iPhone (WebGPU${hasF16 ? ", f16" : ""}). Ears: Whisper · Brain: ${engineModel || "not loaded"} · Voice: iOS.`;
}
function fillVoices() {
  const vs = ("speechSynthesis" in window ? speechSynthesis.getVoices() : []).filter((v) => /^en/i.test(v.lang));
  const sel = $("voiceSel");
  const rank = (v) => (/premium/i.test(v.name) ? 0 : /enhanced/i.test(v.name) ? 1 : 2);
  vs.sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
  sel.innerHTML = vs.map((v) => `<option value="${esc(v.voiceURI)}" ${v.voiceURI === S.voice ? "selected" : ""}>${esc(v.name)} (${esc(v.lang)})</option>`).join("") || "<option>Default voice</option>";
  if (!S.voice && vs[0]) save("voice", vs[0].voiceURI);
}
if ("speechSynthesis" in window) speechSynthesis.onvoiceschanged = () => fillVoices();
$("voiceSel").onchange = (e) => { save("voice", e.target.value); sample(); };
$("speed").oninput = (e) => { $("speedVal").textContent = (+e.target.value).toFixed(2) + "×"; };
$("speed").onchange = (e) => { save("speed", +e.target.value); sample(); };
$("speak").onchange = (e) => save("speak", e.target.checked);
$("thr").oninput = (e) => { save("thr", +e.target.value); $("thrVal").textContent = (+e.target.value).toFixed(3); $("micThr").style.left = Math.min(100, Math.sqrt(S.thr / 0.2) * 100) + "%"; };
$("sil").oninput = (e) => { save("sil", +e.target.value); $("silVal").textContent = (e.target.value / 1000).toFixed(1) + " s"; };
$("savePrompt").onclick = () => { save("prompt", $("sysPrompt").value.trim() || DEFAULT_PROMPT); $("savePrompt").textContent = "Saved ✓"; setTimeout(() => ($("savePrompt").textContent = "Save personality"), 1400); };
$("clearModels").onclick = async () => {
  stopReply();
  const keys = await caches.keys();
  for (const k of keys) if (!k.startsWith("halo-go-")) await caches.delete(k);
  for (const db of ["webllm/model", "webllm/config", "webllm/wasm"]) {
    await new Promise((r) => { const q = indexedDB.deleteDatabase(db); q.onsuccess = q.onerror = q.onblocked = () => r(); });
  }
  note("Deleted the downloaded brains and ears. Halo Go will download what it needs next time.");
  setTimeout(() => location.reload(), 1500);
};
function sample() { stopReply(); const id = replyId; say("Hi, I'm Halo Go. This is how I sound.", id); }

async function switchBrain(key) {
  if (key === S.model || state === "loading") return;
  const before = S.model;
  const info = modelInfo(key);
  if (!(await brainCached(key)) && !confirm(`Download the ${info.name} brain (${info.gb} GB)? Use Wi-Fi.`)) return;
  closeSheet(); stopReply(); releaseMic();
  setState("loading", `Loading ${info.name} brain…`);
  try {
    await loadBrain(key, (p) => setState("loading", `Loading ${info.name} brain… ${Math.round(p * 100)}%`));
    save("model", key);
    note(`Switched brain: ${modelInfo(before).name} → ${info.name}`);
    setState("idle");
    say(`Okay. I'm now using the ${info.name} brain.`, replyId);
  } catch (e) {
    err(`Couldn't load the ${info.name} brain: ${e.message || e}. Going back to ${modelInfo(before).name}.`);
    try { await loadBrain(before); } catch {}
    setState("idle");
  }
}
async function switchEars(id) {
  if (id === S.ears) return;
  save("ears", id);
  closeSheet();
  if (DEMO) { note("Ears switched."); return; }
  setState("loading", "Loading ears…");
  sttWorker.postMessage({ type: "load", id });
  note("Switching ears — the first time it downloads a little.");
  setTimeout(() => state === "loading" && setState("idle"), 1500);
}

// ---------------- Updates (service worker) ----------------
let waitingSW = null;
function showUpdate(w) {
  waitingSW = w;
  if (!$("gate").classList.contains("hidden") || state === "off") { w.postMessage("skipWaiting"); return; }
  $("updChip").style.display = "";
}
$("updChip").onclick = () => { if (waitingSW) waitingSW.postMessage("skipWaiting"); };
if ("serviceWorker" in navigator && !DEMO) {
  navigator.serviceWorker.register("sw.js").then((reg) => {
    if (reg.waiting && navigator.serviceWorker.controller) showUpdate(reg.waiting);
    reg.addEventListener("updatefound", () => {
      const nw = reg.installing;
      nw && nw.addEventListener("statechange", () => {
        if (nw.state === "installed" && navigator.serviceWorker.controller) showUpdate(nw);
      });
    });
    if (navigator.onLine) reg.update().catch(() => {});
  }).catch(() => {});
  let reloading = false;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (reloading) return;
    if (!$("gate").classList.contains("hidden")) { reloading = true; location.reload(); }   // safe: not chatting yet
    else { $("updChip").style.display = ""; $("updChip").onclick = () => location.reload(); }  // let the user choose
  });
}

// ---------------- Demo mode (for testing without WebGPU) ----------------
function fakeProgress(cb, ms) {
  return new Promise((r) => { const t0 = performance.now(); const step = () => { const p = Math.min(1, (performance.now() - t0) / ms); cb && cb(p, ""); p < 1 ? setTimeout(step, 60) : r(); }; step(); });
}
function makeDemoEngine() {
  return {
    interruptGenerate() {},
    chat: { completions: { async create({ messages }) {
      const last = messages[messages.length - 1].content.slice(-60);
      const text = `This is demo mode. On your iPhone I'd really answer "${last}". Everything runs right here on the phone.`;
      return (async function* () { for (let i = 0; i < text.length; i += 5) { await new Promise((r) => setTimeout(r, 25)); yield { choices: [{ delta: { content: text.slice(i, i + 5) } }] }; } })();
    } } },
  };
}

// ---------------- Start-up ----------------
async function checkDevice() {
  const box = $("gateCheck");
  const standalone = navigator.standalone || matchMedia("(display-mode: standalone)").matches;
  const iOS = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  if (iOS && !standalone) $("gateHome").style.display = "";
  if (DEMO) { box.innerHTML = "Demo mode: no real AI, just the screens."; return true; }
  if (!navigator.gpu) {
    box.classList.add("bad");
    box.innerHTML = "<b>This needs Safari on iOS 26 or newer.</b> Halo Go runs the AI on the iPhone's graphics chip (WebGPU). Update in Settings → General → Software Update, then open this page in Safari.";
    return false;
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no adapter");
    hasF16 = adapter.features.has("shader-f16");
  } catch {
    box.classList.add("bad");
    box.innerHTML = "<b>Couldn't reach the graphics chip.</b> Close other apps and try again, or restart the iPhone.";
    return false;
  }
  // Did the last launch run out of memory loading a big brain?
  const crashed = ls.get("loading", null);
  if (crashed && crashed !== "quick") {
    ls.set("loading", null); save("model", "quick");
    box.innerHTML = `The <b>${modelInfo(crashed).name}</b> brain was too big for this iPhone, so Halo Go switched back to <b>Quick</b>.`;
  } else {
    box.innerHTML = "This iPhone can run Halo Go. 👍";
  }
  return true;
}

async function boot() {
  renderHistory();
  setMode(S.mode, false);
  requestAnimationFrame(animate);
  const ok = await checkDevice();
  const btn = $("startBtn");
  if (!ok) { btn.textContent = "Not supported here"; return; }
  const cached = await brainCached(S.model);
  btn.disabled = false;
  btn.textContent = cached ? "Start talking" : `Download & start (${(modelInfo(S.model).gb + 0.05).toFixed(1)} GB)`;
  btn.onclick = start;
}

async function start() {
  const btn = $("startBtn");
  btn.disabled = true;
  // These must happen right inside the tap (iOS rules for sound and microphone)
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  if ("speechSynthesis" in window) { const u = new SpeechSynthesisUtterance(" "); u.volume = 0; speechSynthesis.speak(u); }
  const micReady = ensureMic();
  try { await navigator.storage?.persist?.(); } catch {}
  $("dlBox").style.display = "";
  btn.textContent = "Getting ready…";
  const bar = (id, p, txt) => { $(id).style.width = Math.round(p * 100) + "%"; $(id + "Txt").textContent = txt; };
  try {
    await micReady;
    await Promise.all([
      startEars((p, total) => bar("dlEars", p, p >= 1 ? "ready" : total ? `${Math.round(p * 100)}% of ${(total / 1e6).toFixed(0)} MB` : "starting…")),
      loadBrain(S.model, (p, text) => bar("dlBrain", p, p >= 1 ? "ready" : /cache/i.test(text) ? `waking up… ${Math.round(p * 100)}%` : `${Math.round(p * 100)}%`)),
    ]);
    bar("dlEars", 1, "ready"); bar("dlBrain", 1, "ready");
    if (DEMO) ls.set("demo_cached", true);
    $("gate").classList.add("hidden");
    setState("idle");
    if (!history.length) note(`Ready · ${modelInfo(S.model).name} brain · everything runs on this iPhone`);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = "Try again";
    $("dlNote").textContent = "Something went wrong: " + (e.message || e) + ". Check Wi-Fi for the first download, then try again.";
  }
}

boot();
