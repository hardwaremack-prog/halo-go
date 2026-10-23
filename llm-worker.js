// Runs the language model on the iPhone's GPU (WebGPU), off the main thread.
import { WebWorkerMLCEngineHandler } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm";
import { installSafeStore, status as store } from "./safe-store.js";

// If Safari won't save the brain, keep it in memory instead of failing
installSafeStore();

// ---- Know which step failed, so errors say where ----
let stage = "starting";
const realFetch = self.fetch.bind(self);
self.fetch = (input, init) => {
  const url = typeof input === "string" ? input : input.url || String(input);
  const file = url.split("?")[0].split("/").pop() || url;
  if (!/^params_shard/.test(file) || !/shard/.test(stage)) stage = "downloading " + file;
  return realFetch(input, init).catch((e) => { throw new Error(`Download failed for ${file}: ${e.message || e}`); });
};
const realInst = WebAssembly.instantiate;
WebAssembly.instantiate = function (...a) { stage = "starting the brain program (WebAssembly)"; return realInst.apply(this, a); };

// ---- The graphics chip: if Safari refuses WebLLM's big memory request, ask for less ----
if (self.GPUAdapter && GPUAdapter.prototype.requestDevice) {
  const realReq = GPUAdapter.prototype.requestDevice;
  GPUAdapter.prototype.requestDevice = async function (desc = {}) {
    stage = "opening the graphics chip";
    try { return await realReq.call(this, desc); }
    catch (e1) {
      const lim = { ...(desc.requiredLimits || {}) };
      if (lim.maxBufferSize) lim.maxBufferSize = Math.min(lim.maxBufferSize, 1 << 28);
      if (lim.maxStorageBufferBindingSize) lim.maxStorageBufferBindingSize = Math.min(lim.maxStorageBufferBindingSize, 1 << 27);
      const feats = (desc.requiredFeatures || []).filter((f) => f !== "subgroups");
      try { return await realReq.call(this, { ...desc, requiredLimits: lim, requiredFeatures: feats }); }
      catch (e2) { throw new Error(`The graphics chip refused (${e1.name || "Error"}: ${e1.message}; smaller request: ${e2.name || "Error"}: ${e2.message})`); }
    }
  };
}

const handler = new WebWorkerMLCEngineHandler();
handler.postMessage = (msg) => {
  if (msg && msg.kind === "initProgressCallback") stage = "loading the brain's knowledge (shards)";
  if (msg && msg.kind === "throw" && typeof msg.content === "string") {
    msg.content += ` [step: ${stage}${store.diskOk ? "" : "; storage: " + store.firstError}]`;
  }
  self.postMessage(msg);
};
self.onmessage = (msg) => { if (msg.data && msg.data.kind === "reload") stage = "starting"; handler.onmessage(msg); };
