// Offline shell for the analysis board. The app shell and /api/topology are
// cached so saved positions can be edited offline; every other /api call
// (analysis, validation, games) goes to the network only.
const CACHE = "hexhaven-v1";

self.addEventListener("install", (e) => {
  // Precache the shell plus the hashed bundles that index.html references.
  e.waitUntil((async () => {
    const c = await caches.open(CACHE);
    await c.addAll(["/", "/index.html", "/manifest.webmanifest"]);
    const html = await (await c.match("/index.html")).text();
    await c.addAll([...html.matchAll(/(?:src|href)="(\/assets\/[^"]+)"/g)].map((m) => m[1]));
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

// Network first, cache as fallback; successful responses refresh the cache.
async function networkFirst(req, key) {
  const c = await caches.open(CACHE);
  try {
    const r = await fetch(req);
    if (r.ok) c.put(key || req, r.clone());
    return r;
  } catch (err) {
    const hit = await c.match(key || req);
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  if (url.pathname.startsWith("/api/")) {
    if (url.pathname === "/api/topology") e.respondWith(networkFirst(e.request));
    return;
  }
  if (e.request.mode === "navigate") { e.respondWith(networkFirst(e.request, "/index.html")); return; }
  e.respondWith(networkFirst(e.request));
});
