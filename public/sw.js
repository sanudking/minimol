const CACHE = 'minimol-mobile-v9';

self.addEventListener('install', event => event.waitUntil((async () => {
  const cache = await caches.open(CACHE);
  const manifestResponse = await fetch('./precache-manifest.json', { cache: 'no-store' });
  if (!manifestResponse.ok) throw new Error('MiniMoL precache manifest could not be loaded.');
  const manifest = await manifestResponse.json();
  const assets = ['./', './index.html', './precache-manifest.json', ...(manifest.files || []).map(file => `./${file}`)];
  await cache.addAll([...new Set(assets)]);
  await self.skipWaiting();
})()));

self.addEventListener('activate', event => event.waitUntil((async () => {
  const keys = await caches.keys();
  await Promise.all(keys.filter(key => key.startsWith('minimol-') && key !== CACHE).map(key => caches.delete(key)));
  await self.clients.claim();
})()));

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.method !== 'GET' || new URL(request.url).origin !== self.location.origin) return;
  event.respondWith((async () => {
    const cached = await caches.match(request);
    if (cached) return cached;
    const response = await fetch(request);
    const url = new URL(request.url);
    const isPartition = /\/models\/phone[12]\.bin$/.test(url.pathname);
    if (response.ok && request.url.startsWith(self.registration.scope) && !isPartition) {
      const copy = response.clone();
      const cache = await caches.open(CACHE);
      await cache.put(request, copy);
    }
    return response;
  })());
});
