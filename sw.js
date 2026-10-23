// Halo Go service worker: makes the app work offline and handles updates.
const VERSION = "1.0.2";
const APP = "halo-go-app-" + VERSION;
const LIBS = "halo-go-libs";
const SHELL = ["./", "index.html", "style.css", "app.js", "llm-worker.js", "stt-worker.js",
  "manifest.webmanifest", "icons/apple-touch-icon.png", "icons/icon-192.png", "icons/icon-512.png"];

self.addEventListener("install", (e) => {
  // Take over right away; the page reloads itself when it's safe (not mid-conversation)
  e.waitUntil(caches.open(APP).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k.startsWith("halo-go-app-") && k !== APP) await caches.delete(k);
    await self.clients.claim();
  })());
});

self.addEventListener("message", (e) => { if (e.data === "skipWaiting") self.skipWaiting(); });

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    // App files: from the installed version (works offline)
    e.respondWith((async () => {
      const c = await caches.open(APP);
      const hit = await c.match(req, { ignoreSearch: true }) || (req.mode === "navigate" ? await c.match("index.html") : null);
      return hit || fetch(req);
    })());
  } else if (url.hostname === "cdn.jsdelivr.net") {
    // AI libraries: pinned versions, keep a copy for offline use
    e.respondWith((async () => {
      const c = await caches.open(LIBS);
      const hit = await c.match(req);
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) c.put(req, res.clone());
      return res;
    })());
  }
  // Model downloads (Hugging Face, GitHub) are cached by the AI libraries themselves
});
