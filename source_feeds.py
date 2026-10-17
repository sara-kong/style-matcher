"""
source_feeds.py — check your favorite brands for new products and score them with your taste model.

Reads each brand's public Shopify feed (store.com/products.json), skips products it has seen before,
sends the new ones to the scoring server, and writes feeds/digest.html: new arrivals above your
cutoff, with Like / Maybe / Dislike buttons that save labels just like the extension.

Needs the scoring server running (python server.py) in another terminal.
Run:   python source_feeds.py
Then:  open feeds/digest.html
"""
import html
import json
import time
from datetime import date
from pathlib import Path

import requests

BRANDS = {
    "With Jean": "https://withjean.com",
    "Guizio": "https://danielleguiziony.com",
    # Add more Shopify stores here: "Name": "https://store.com",
}
SERVER = "http://127.0.0.1:8000"
MAX_PAGES = 4          # 250 products per page
IMAGE_WIDTH = 600      # Shopify's image CDN resizes on request
ROOT = Path(__file__).resolve().parent
FEEDS = ROOT / "feeds"
SEEN_FILE = FEEDS / "seen.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


def get_json(url, tries=3):
    for attempt in range(tries):
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 429:  # rate limited: wait and try again
            time.sleep(5 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("rate limited (429) — try again later")


def fetch_products(brand, base):
    products = []
    for page in range(1, MAX_PAGES + 1):
        batch = get_json(f"{base}/products.json?limit=250&page={page}").get("products", [])
        if not batch:
            break
        for p in batch:
            if not p.get("images"):
                continue
            src = p["images"][0]["src"]
            price = p["variants"][0].get("price") if p.get("variants") else None
            products.append({
                "key": f"{base}#{p['id']}",
                "brand": brand,
                "title": p.get("title", ""),
                "type": p.get("product_type", ""),
                "price": price,
                "link": f"{base}/products/{p['handle']}",
                "image": src + ("&" if "?" in src else "?") + f"width={IMAGE_WIDTH}",
            })
        time.sleep(1)  # be polite to the store
    return products


def score(products):
    scores, cutoff = {}, None
    for i in range(0, len(products), 40):
        chunk = products[i:i + 40]
        r = requests.post(f"{SERVER}/score", timeout=300, json={
            "items": [{"url": p["image"]} for p in chunk], "page": chunk[0]["link"], "download": True})
        if r.status_code == 409:
            raise SystemExit(r.json()["detail"])
        r.raise_for_status()
        data = r.json()
        scores.update(data["scores"])
        cutoff = data["cutoff"]
        print(f"  scored {min(i + 40, len(products))}/{len(products)}")
    return scores, cutoff


def card(p):
    s = p["score"]
    price = f" · ${p['price']}" if p["price"] else ""
    return f"""<article class="card">
  <a href="{html.escape(p['link'])}" target="_blank" rel="noopener"><img src="{html.escape(p['image'])}" alt="{html.escape(p['title'])}" loading="lazy"></a>
  <div class="meta"><span class="score">{round(s * 100)}</span><span class="brand">{html.escape(p['brand'])}{html.escape(price)}</span></div>
  <h2><a href="{html.escape(p['link'])}" target="_blank" rel="noopener">{html.escape(p['title'])}</a></h2>
  <div class="actions" data-url="{html.escape(p['image'])}" data-page="{html.escape(p['link'])}">
    <button data-label="like">Like</button><button data-label="maybe">Maybe</button><button data-label="dislike">Dislike</button>
  </div>
</article>"""


def write_digest(keep, below, total_new, cutoff, errors):
    notes = "".join(f"<p class='note'>{html.escape(e)}</p>" for e in errors)
    body = "".join(card(p) for p in keep) or "<p class='empty'>Nothing new passed your cutoff today.</p>"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>New arrivals worth a look — {date.today():%b %-d}</title>
<style>
  :root {{ --ink:#2b2a33; --paper:#fdfdfb; --muted:#6d6b75; --line:rgba(43,42,51,.14); --keep:#41583a; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --ink:#ecebef; --paper:#1d1c22; --muted:#a3a1ab; --line:rgba(236,235,239,.16); --keep:#7fa071; }} }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 "Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif; }}
  main {{ max-width:1100px; margin:0 auto; padding:32px 20px 60px; }}
  h1 {{ font-size:26px; font-weight:600; margin:0 0 4px; }}
  .sub, .note, .empty {{ color:var(--muted); margin:0 0 6px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:24px; margin-top:24px; }}
  img {{ width:100%; aspect-ratio:3/4; object-fit:cover; border-radius:6px; display:block; background:var(--line); }}
  .meta {{ display:flex; gap:8px; align-items:center; margin:10px 0 2px; font-size:13px; color:var(--muted); }}
  .score {{ background:var(--keep); color:#fff; font-weight:600; font-variant-numeric:tabular-nums; padding:2px 7px; border-radius:3px 9px 9px 3px; }}
  h2 {{ font-size:14px; font-weight:500; margin:0 0 8px; }}
  h2 a {{ color:inherit; text-decoration:none; }}
  .actions {{ display:flex; gap:6px; }}
  button {{ font:inherit; font-size:12px; padding:5px 9px; border-radius:7px; border:1px solid var(--line); background:transparent; color:var(--ink); cursor:pointer; }}
  button:hover {{ border-color:var(--ink); }}
  button.on {{ background:var(--ink); color:var(--paper); border-color:var(--ink); }}
  button:focus-visible {{ outline:2px solid #4a7bd0; outline-offset:2px; }}
</style></head><body><main>
<h1>New arrivals worth a look</h1>
<p class="sub">{date.today():%A, %B %-d} · {len(keep)} of {total_new} new products passed your cutoff ({cutoff:.2f}); {below} didn’t.</p>
{notes}
<div class="grid">{body}</div>
</main>
<script>
document.querySelectorAll(".actions").forEach(box => box.addEventListener("click", async e => {{
  const b = e.target.closest("button"); if (!b) return;
  try {{
    const r = await fetch("{SERVER}/label", {{ method: "POST", headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ url: box.dataset.url, page: box.dataset.page, label: b.dataset.label }}) }});
    if (!r.ok) throw new Error((await r.json()).detail);
    box.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
  }} catch (err) {{ alert("Couldn’t save the label. Is python server.py running? " + err.message); }}
}}));
</script></body></html>"""
    (FEEDS / "digest.html").write_text(page, encoding="utf-8")


def main():
    try:
        requests.get(f"{SERVER}/status", timeout=3).raise_for_status()
    except Exception:
        raise SystemExit("The scoring server isn't running. Start it in another terminal: python server.py")

    FEEDS.mkdir(exist_ok=True)
    seen = json.loads(SEEN_FILE.read_text()) if SEEN_FILE.exists() else {}
    new, errors = [], []
    for brand, base in BRANDS.items():
        try:
            products = fetch_products(brand, base)
        except Exception as e:
            errors.append(f"{brand}: couldn’t read the feed ({e}).")
            print(f"{brand}: failed — {e}")
            continue
        fresh = [p for p in products if p["key"] not in seen]
        print(f"{brand}: {len(products)} products, {len(fresh)} new")
        new += fresh

    if not new:
        print("Nothing new since the last run, so the previous digest was left as is.")
        return

    scores, cutoff = score(new)
    scored = [dict(p, score=scores[p["image"]]) for p in new if p["image"] in scores]
    keep = sorted((p for p in scored if p["score"] >= cutoff), key=lambda p: -p["score"])
    for p in scored:  # only mark as seen once scored, so failures get retried next run
        seen[p["key"]] = str(date.today())
    SEEN_FILE.write_text(json.dumps(seen, indent=0))

    write_digest(keep, len(scored) - len(keep), len(scored), cutoff, errors)
    print(f"\n{len(keep)} of {len(scored)} new products passed the cutoff. Open feeds/digest.html")


if __name__ == "__main__":
    main()
