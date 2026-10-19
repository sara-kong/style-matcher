// Relays requests from pages and the popup to the local scoring server.
// Images are downloaded here, inside Chrome, because many stores block downloads
// that don't come from a browser. They're shrunk before being sent to the server.
const SERVER = "http://127.0.0.1:8000";
const MAX_SIDE = 512; // px; the model looks at 224px, so this keeps plenty of detail

async function call(path, body) {
  const res = await fetch(SERVER + path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Server error ${res.status}`);
  return data;
}

async function toDataUrl(url) {
  if (url.startsWith("data:")) return url;
  const res = await fetch(url, { credentials: "include" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const bitmap = await createImageBitmap(await res.blob());
  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = new OffscreenCanvas(Math.max(1, Math.round(bitmap.width * scale)),
                                     Math.max(1, Math.round(bitmap.height * scale)));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const jpeg = new Uint8Array(await (await canvas.convertToBlob({ type: "image/jpeg", quality: 0.9 })).arrayBuffer());
  let binary = "";
  for (let i = 0; i < jpeg.length; i += 0x8000) binary += String.fromCharCode(...jpeg.subarray(i, i + 0x8000));
  return "data:image/jpeg;base64," + btoa(binary);
}

async function score(urls, page) {
  // 1) Ask for scores of images the server has already seen.
  const first = await call("/score", { items: urls.map((url) => ({ url })), page, download: false });
  if (!first.missing.length) return first;
  // 2) Download the rest here and send the images themselves. If the browser can't
  //    get one, send just the link and let the server try.
  const items = await Promise.all(first.missing.map(async (url) => {
    try { return { url, data: await toDataUrl(url) }; } catch { return { url }; }
  }));
  const second = await call("/score", { items, page, download: true });
  return { scores: { ...first.scores, ...second.scores }, cutoff: second.cutoff };
}

async function label(url, value, page) {
  let data = null;
  try { data = await toDataUrl(url); } catch { /* server will try the link */ }
  return call("/label", { url, label: value, data, page });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const routes = {
    score: () => score(msg.urls, msg.page),
    label: () => label(msg.url, msg.label, msg.page),
    retrain: () => call("/retrain", {}),
    status: () => call("/status"),
    fetchImage: async () => ({ data: await toDataUrl(msg.url) }),
  };
  const run = routes[msg.type];
  if (!run) return false;
  run()
    .then((data) => sendResponse({ ok: true, data }))
    .catch((err) => sendResponse({ ok: false, error: err.message || "Scoring server isn't running" }));
  return true; // keeps the message channel open for the async reply
});
