// Taste Score content script: finds product images, asks the local server for scores,
// and draws a hang tag on each one with Like / Maybe / Dislike buttons.
(() => {
  if (window.__tasteScore) return;
  window.__tasteScore = true;

  // On the purchase review page (purchases.py), act only as a helper that fetches photos
  // through the extension, which stores like Aritzia accept. No tags are drawn there.
  if (location.origin === "http://127.0.0.1:8001") {
    document.documentElement.dataset.tasteBridge = "1";
    addEventListener("message", (e) => {
      if (e.source !== window || e.data?.type !== "taste-fetch-image") return;
      const reply = (data) => postMessage({ type: "taste-image", id: e.data.id, data }, location.origin);
      try {
        chrome.runtime.sendMessage({ type: "fetchImage", url: e.data.url },
          (res) => reply(!chrome.runtime.lastError && res?.ok ? res.data.data : null));
      } catch { reply(null); }
    });
    return;
  }

  const MIN_SIZE = 120;     // ignore icons, logos and thumbnails smaller than this (px)
  const BATCH_DELAY = 300;  // wait this long to collect images before asking the server
  const BATCH_MAX = 40;

  const scores = new Map();   // url -> score
  const labels = new Map();   // url -> "like" | "dislike" | "maybe" | "owned"
  const nearest = new Map();  // url -> similarity to the closest thing you own (0-1)
  let ownT = 0.9, similarT = 0.82;
  const tracked = new Map();  // <img> -> { url, tag }
  const dimmed = new Map();   // <img> -> original inline styles
  const queue = new Set();
  let cutoff = 0.5;
  let enabled = true;
  let flushTimer = null;
  let frame = null;
  let serverDown = false;
  let observer = null;

  // After the extension is reloaded or updated, this old copy of the script can no longer
  // reach it. Instead of throwing errors, it removes its tags and goes quiet.
  function send(msg, callback) {
    try {
      if (!chrome.runtime?.id) throw new Error("extension reloaded");
      chrome.runtime.sendMessage(msg, callback);
    } catch {
      shutdown();
    }
  }

  function shutdown() {
    enabled = false;
    observer?.disconnect();
    clearAll();
    host.remove();
  }

  // ---------- overlay (shadow DOM keeps our styles away from the site's) ----------
  const host = document.createElement("div");
  host.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:2147483646;";
  const root = host.attachShadow({ mode: "open" });
  // Inter ships with the extension. Fonts load at the document level, so the face is declared there
  // under its own name (it won't clash with a site's own Inter); the tags fall back to the system font
  // if a site's security policy blocks it.
  const fontFace = document.createElement("style");
  fontFace.textContent = `@font-face { font-family: "Taste Score Inter"; font-weight: 100 900;
    font-display: swap; src: url("${chrome.runtime.getURL("fonts/inter-latin.woff2")}") format("woff2"); }`;
  document.documentElement.appendChild(fontFace);

  root.innerHTML = `<style>
    :host { all: initial; }
    .tag {
      --rose: #b89393; --rose-2: #c0a5a5; --mist: #dcd5d5; --paper: #f2f2f2; --ink: #4a3d3d;
      position: fixed; pointer-events: auto; display: flex; align-items: center; gap: 6px;
      height: 23px; box-sizing: border-box; padding: 0 10px; border-radius: 5px;
      background: var(--mist); color: #fff;
      font: 600 10px/1 "Taste Score Inter", Inter, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      font-variant-numeric: tabular-nums;
      filter: drop-shadow(0 2px 2px rgba(184, 147, 147, .25));
    }
    .tag.keep { background: var(--rose); }
    .tag.owned { background: var(--paper); color: var(--ink); box-shadow: inset 0 0 0 1px var(--mist); }
    .tag.done { font-weight: 500; }
    .note { font-weight: 500; padding-left: 6px; border-left: 1px solid currentColor; opacity: .9; }
    .actions { display: none; gap: 2px; margin-right: -6px; }
    .tag:hover .actions, .tag:focus-within .actions { display: flex; }
    button {
      all: unset; cursor: pointer; height: 15px; padding: 0 8px; border-radius: 5px;
      display: inline-flex; align-items: center;
      font: 600 8px/1 "Taste Score Inter", Inter, -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
      color: #fff; background: var(--rose-2); white-space: nowrap;
    }
    .tag.owned button { background: var(--rose); }
    button:hover { background: #a98282; }
    button:focus-visible { outline: 2px solid #4a7bd0; outline-offset: 1px; }
  </style>`;
  document.documentElement.appendChild(host);

  // ---------- finding images ----------
  function imageUrl(img) {
    const u = img.currentSrc || img.src || "";
    return /^(https?:|data:image\/)/.test(u) ? u : null;
  }

  // True only if the image is actually visible: not hidden, not transparent,
  // and not clipped away inside a carousel or other overflow-hidden container.
  function isShown(img, r) {
    if (r.width < MIN_SIZE || r.height < MIN_SIZE) return false;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    for (let el = img, depth = 0; el && el !== document.body && depth < 15; el = el.parentElement, depth++) {
      const cs = getComputedStyle(el);
      if (cs.display === "none" || cs.visibility === "hidden" || parseFloat(cs.opacity) < 0.1) return false;
      if (el !== img && (cs.overflowX !== "visible" || cs.overflowY !== "visible")) {
        const a = el.getBoundingClientRect();
        if (cx < a.left || cx > a.right || cy < a.top || cy > a.bottom) return false;
      }
    }
    return true;
  }

  function overlap(a, b) {
    const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
    const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
    if (w <= 0 || h <= 0) return 0;
    return (w * h) / Math.min(a.width * a.height, b.width * b.height);
  }

  function scan() {
    if (!enabled) return;
    for (const img of document.images) {
      if (!img.complete || !img.naturalWidth) continue;
      if (!isShown(img, img.getBoundingClientRect())) continue;
      const url = imageUrl(img);
      if (!url) continue;
      const t = tracked.get(img);
      if (t && t.url === url) continue;       // already handled
      t?.tag?.remove();                       // image changed (e.g. lazy-loaded): start over
      tracked.set(img, { url, tag: null });
      if (scores.has(url)) render(img);
      else queue.add(url);
    }
    if (queue.size && !flushTimer) flushTimer = setTimeout(flush, BATCH_DELAY);
  }

  function flush() {
    flushTimer = null;
    const urls = [...queue].slice(0, BATCH_MAX);
    urls.forEach((u) => queue.delete(u));
    send({ type: "score", urls, page: location.href }, (res) => {
      if (chrome.runtime.lastError || !res) return;
      if (!res.ok) {
        if (!serverDown) console.info("[Taste Score]", res.error);
        serverDown = true;
        return;
      }
      serverDown = false;
      cutoff = res.data.cutoff;
      ownT = res.data.own_threshold ?? ownT;
      similarT = res.data.similar_threshold ?? similarT;
      for (const [u, s] of Object.entries(res.data.scores)) scores.set(u, s);
      for (const [u, s] of Object.entries(res.data.nearest || {})) nearest.set(u, s);
      for (const img of tracked.keys()) render(img);
      if (queue.size) flushTimer = setTimeout(flush, 50);
    });
  }

  // ---------- drawing ----------
  function render(img) {
    const t = tracked.get(img);
    if (!t || !scores.has(t.url)) return;
    const s = scores.get(t.url);
    const near = nearest.get(t.url);
    const owned = labels.get(t.url) === "owned" || (near != null && near >= ownT);
    const similar = !owned && near != null && near >= similarT;
    const keep = s >= cutoff && !owned;  // never highlight something you already own
    const label = labels.get(t.url);

    if (!t.tag) {
      t.tag = document.createElement("div");
      t.tag.className = "tag";
      root.appendChild(t.tag);
    }
    t.tag.classList.toggle("keep", keep);
    t.tag.classList.toggle("done", !!label);
    t.tag.classList.toggle("owned", owned);
    t.tag.title = `Predicted chance you'd like this: ${Math.round(s * 100)}%` +
      (near != null ? `\nClosest thing you own: ${Math.round(near * 100)}% similar` : "");
    t.tag.innerHTML = "";
    const score = document.createElement("span");
    score.textContent = label ? { like: "Liked", dislike: "Disliked", maybe: "Maybe", owned: "You own this" }[label]
      : owned ? "You own this" : Math.round(s * 100);
    t.tag.appendChild(score);
    if (similar && !label) {
      const note = document.createElement("span");
      note.className = "note";
      note.textContent = "Similar to yours";
      t.tag.appendChild(note);
    }

    const actions = document.createElement("span");
    actions.className = "actions";
    for (const [value, text] of [["like", "Like"], ["maybe", "Maybe"], ["dislike", "Dislike"], ["owned", "I own this"]]) {
      const b = document.createElement("button");
      b.textContent = text;
      b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); sendLabel(img, value); });
      actions.appendChild(b);
    }
    t.tag.appendChild(actions);

    // Fade images below the cutoff; outline the ones worth a look.
    if (!dimmed.has(img)) dimmed.set(img, { filter: img.style.filter, outline: img.style.outline, offset: img.style.outlineOffset });
    img.style.filter = keep ? dimmed.get(img).filter: "grayscale(0.7) brightness(1.2) contrast (0.8)";
    img.style.outline = keep ? "3px solid #b89393" : "";
    img.style.outlineOffset = keep ? "-3px" : "";
    place();
  }

  function sendLabel(img, value) {
    const t = tracked.get(img);
    if (!t) return;
    send({ type: "label", url: t.url, label: value, page: location.href }, (res) => {
      if (chrome.runtime.lastError || !res) return;
      if (!res.ok) { alert(`Taste Score: ${res.error}`); return; }
      labels.set(t.url, value);
      render(img);
    });
  }

  function place() {
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = null;
      const kept = [];  // rects that already have a tag, so stacked images get only one
      for (const [img, t] of tracked) {
        if (!t.tag) continue;
        if (!img.isConnected) { t.tag.remove(); tracked.delete(img); continue; }
        const r = img.getBoundingClientRect();
        const visible = r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth
          && isShown(img, r) && !kept.some((k) => overlap(k, r) > 0.5);
        t.tag.style.display = visible ? "flex" : "none";
        if (visible) {
          kept.push(r);
          t.tag.style.top = `${Math.max(r.top, 0) + 6}px`;
          t.tag.style.left = `${r.left + 6}px`;
        }
      }
    });
  }

  function clearAll() {
    for (const [img, t] of tracked) if (t.tag) t.tag.remove();
    for (const [img, orig] of dimmed) {
      img.style.filter = orig.filter; img.style.outline = orig.outline; img.style.outlineOffset = orig.offset;
    }
    tracked.clear(); dimmed.clear(); queue.clear();
  }

  // ---------- wiring ----------
  let scanTimer = null;
  const scheduleScan = () => { clearTimeout(scanTimer); scanTimer = setTimeout(scan, 400); };
  addEventListener("scroll", () => { place(); scheduleScan(); }, { passive: true, capture: true });
  addEventListener("resize", () => { place(); scheduleScan(); });
  addEventListener("load", scheduleScan, true); // image load events bubble in capture phase
  observer = new MutationObserver(scheduleScan);
  observer.observe(document.documentElement,
    { childList: true, subtree: true, attributes: true, attributeFilter: ["src", "srcset"] });

  chrome.storage.local.get({ enabled: true }, (v) => { enabled = v.enabled; if (enabled) scan(); });
  chrome.storage.onChanged.addListener((changes) => {
    if (!changes.enabled) return;
    enabled = changes.enabled.newValue;
    if (enabled) scan(); else clearAll();
  });
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "rescore") { scores.clear(); clearAll(); serverDown = false; scan(); }
  });
})();
