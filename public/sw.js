// Service worker: (1) caches the app shell so the PWA works offline,
// (2) adds COOP/COEP headers so the page is cross-origin isolated, which
// enables multi-threaded WASM for the embedder and the 3B model on static
// hosts (GitHub Pages) that cannot set headers. It never sees document data:
// PDFs are read from a local file input and never fetched.
const CACHE = 'filing-lens-v1';

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return; // model weights: browser/wllama cache handles them
  e.respondWith(
    (async () => {
      let res;
      try {
        res = await fetch(req);
        if (res.ok && res.type === 'basic') {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        }
      } catch {
        res = await caches.match(req);
        if (!res) return new Response('offline', { status: 503 });
      }
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
    })(),
  );
});
