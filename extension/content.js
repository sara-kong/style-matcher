// Taste Score content script: finds product images, asks the local server for scores,
// and draws a hang tag on each one with Like / Maybe / Dislike buttons.
(() => {
  if (window.__tasteScore) return;
  window.__tasteScore = true;

  const MIN_SIZE = 120;     // ignore icons, logos and thumbnails smaller than this (px)
  const BATCH_DELAY = 300;  // wait this long to collect images before asking the server
  const BATCH_MAX = 40;

  const scores = new Map();   // url -> score
  const labels = new Map();   // url -> "like" | "dislike" | "maybe"
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
  root.innerHTML = `<style>
    :host { all: initial; }
    .tag {
      --ink: #2b2a33; --paper: #fdfdfb; --keep: #41583a; --line: rgba(43,42,51,.18);
      position: fixed; pointer-events: auto; display: flex; align-items: center; gap: 6px;
      padding: 5px 9px 5px 7px; border-radius: 3px 10px 10px 3px;
      background: var(--paper); color: var(--ink); border: 1px solid var(--line);
      font: 600 12px/1 "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      font-variant-numeric: tabular-nums; letter-spacing: .01em;
      box-shadow: 0 1px 2px rgba(0,0,0,.18);
    }
    .tag::before {  /* the punched hole of a hang tag */
      content: ""; width: 6px; height: 6px; border-radius: 50%;
      background: transparent; box-shadow: inset 0 0 0 1.5px currentColor; opacity: .55;
    }
    .tag.keep { background: var(--keep); color: #fff; border-color: transparent; }
    .tag.done { font-weight: 500; }
    .actions { display: none; gap: 2px; margin-left: 2px; }
    .tag:hover .actions, .tag:focus-within .actions { display: flex; }
    button {
      all: unset; cursor: pointer; padding: 3px 6px; border-radius: 6px;
      font: 500 11px/1 "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: inherit; background: rgba(127,127,127,.14);
    }
    button:hover { background: rgba(127,127,127,.28); }
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
      for (const [u, s] of Object.entries(res.data.scores)) scores.set(u, s);
      for (const img of tracked.keys()) render(img);
      if (queue.size) flushTimer = setTimeout(flush, 50);
    });
  }

  // ---------- drawing ----------
  function render(img) {
    const t = tracked.get(img);
    if (!t || !scores.has(t.url)) return;
    const s = scores.get(t.url);
    const keep = s >= cutoff;
    const label = labels.get(t.url);

    if (!t.tag) {
      t.tag = document.createElement("div");
      t.tag.className = "tag";
      root.appendChild(t.tag);
    }
    t.tag.classList.toggle("keep", keep);
    t.tag.classList.toggle("done", !!label);
    t.tag.title = `Predicted chance you'd like this: ${Math.round(s * 100)}%`;
    t.tag.innerHTML = "";
    const score = document.createElement("span");
    score.textContent = label ? { like: "Liked", dislike: "Disliked", maybe: "Maybe" }[label] : Math.round(s * 100);
    t.tag.appendChild(score);

    const actions = document.createElement("span");
    actions.className = "actions";
    for (const [value, text] of [["like", "Like"], ["maybe", "Maybe"], ["dislike", "Dislike"]]) {
      const b = document.createElement("button");
      b.textContent = text;
      b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); sendLabel(img, value); });
      actions.appendChild(b);
    }
    t.tag.appendChild(actions);

    // Fade images below the cutoff; outline the ones worth a look.
    if (!dimmed.has(img)) dimmed.set(img, { opacity: img.style.opacity, outline: img.style.outline, offset: img.style.outlineOffset });
    img.style.opacity = keep ? "" : "0.35";
    img.style.outline = keep ? "3px solid #41583a" : "";
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
      img.style.opacity = orig.opacity; img.style.outline = orig.outline; img.style.outlineOffset = orig.offset;
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
