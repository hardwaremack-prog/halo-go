// Runs the language model on the iPhone's GPU (WebGPU), off the main thread.
import { WebWorkerMLCEngineHandler } from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm";
import { installSafeStore } from "./safe-store.js";

// If Safari won't save the brain, keep it in memory instead of failing
installSafeStore();

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => handler.onmessage(msg);
