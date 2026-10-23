// Speech recognition (Whisper) running on the iPhone, off the main thread.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";

env.allowLocalModels = false;
env.useBrowserCache = true;

let asr = null;
let loadedId = null;
let loading = null;
const files = {};

function report() {
  let loaded = 0, total = 0;
  for (const f of Object.values(files)) { loaded += f.loaded || 0; total += f.total || 0; }
  self.postMessage({ type: "progress", loaded, total });
}

async function load(id) {
  if (asr && loadedId === id) return;
  if (loading && loading.id === id) return loading.p;
  const p = (async () => {
    asr = await pipeline("automatic-speech-recognition", id, {
      dtype: "q8",
      device: "wasm",
      progress_callback: (e) => {
        if (e.status === "progress" || e.status === "download") {
          files[e.file] = { loaded: e.loaded || 0, total: e.total || files[e.file]?.total || 0 };
          report();
        } else if (e.status === "done" && files[e.file]) {
          files[e.file].loaded = files[e.file].total;
          report();
        }
      },
    });
    loadedId = id;
  })();
  loading = { id, p };
  try { await p; } finally { loading = null; }
}

// Things Whisper tends to "hear" in silence
const JUNK = new Set(["", "you", "you.", "thank you.", "thank you", "thanks for watching!", "bye.", ".", "[blank_audio]", "(silence)"]);

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") {
      await load(m.id);
      self.postMessage({ type: "ready", id: m.id });
    } else if (m.type === "transcribe") {
      await load(m.model);
      const out = await asr(m.audio);
      let text = (out && out.text ? out.text : "").trim();
      if (JUNK.has(text.toLowerCase()) || /^\[.*\]$/.test(text) || /^\(.*\)$/.test(text)) text = "";
      self.postMessage({ type: "result", reqId: m.reqId, text });
    }
  } catch (err) {
    self.postMessage({ type: "error", reqId: m.reqId, error: String((err && err.message) || err) });
  }
};
