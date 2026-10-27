import hashlib
import io
import json
import re
import sys
import time
import webbrowser
from base64 import b64decode, urlsafe_b64decode
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "purchases"
CANDIDATES, DECISIONS, SEEN = DATA / "candidates.json", DATA / "decisions.json", DATA / "seen.json"
LOG_FILE = ROOT / "logs" / "events.jsonl"
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
REVIEW_PORT = 8001

# Store name -> words that appear in its sender address or name. Add stores here.
STORES = {
    "Aritzia": ["aritzia"],
    "With Jean": ["withjean"],
    "Guizio": ["guizio", "danielleguizio"],
}
# Stores that run on Shopify: receipts list items by name, so they can be matched to the store's
# public catalog (store.com/products.json) to find the product photo.
SHOPIFY = {
    "With Jean": "https://withjean.com",
    "Guizio": "https://danielleguiziony.com",
}
SUBJECT_WORDS = "order OR confirmation OR receipt OR purchase OR shipped OR return OR refund OR exchange"
LOOKBACK = "3y"
# Order and return emails must look transactional; marketing emails that merely mention
# ordering ("Order by Dec 22 for holiday delivery") are skipped.
RETURN_WORDS = re.compile(r"\brefund|\byour return|\breturn (has been |is )?(received|processed|confirm|started|"
                          r"request|label|complete)|\bexchange (request|confirm|processed)", re.I)
ORDER_WORDS = re.compile(r"\border\s*(#|no\.?|number|confirm)|\byour (\w+ )?order|\border (is|has)|"
                         r"\bshipped|\bon (its|the) way|out for delivery|\bdelivered|\breceipt|"
                         r"thank you for (your )?(order|purchase|shopping)", re.I)
MARKETING = re.compile(r"\b(sale|% ?off|new arrivals?|last (day|chance)|order (by|now|today|in time)|shop now|"
                       r"gift|holiday|deadline|cutoff|cut-off|just dropped|back in stock|members?|rewards)\b", re.I)
# Image text that marks store furniture rather than a product photo.
JUNK_TEXT = re.compile(r"free (shipping|returns)|^returns?$|pick ?-?up in.?store|shop (the )?app|app store|"
                       r"google play|download|view (in browser|online|order|details)|track (your )?(order|package)|"
                       r"customer (care|service)|^help$|my account|sign in|gift card|^shop$|^menu$", re.I)
NOT_PRODUCT = re.compile(r"logo|icon|social|spacer|pixel|track|facebook|instagram|twitter|pinterest|tiktok|"
                         r"youtube|footer|header|banner|badge|arrow|star|rating|button|divider|apple|google-play", re.I)
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}


# ---------------------------------------------------------------- storage

def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def save(path, data):
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def log(event, **fields):
    LOG_FILE.parent.mkdir(exist_ok=True)
    row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **fields}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- Gmail

def gmail():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    token, creds_file = ROOT / "token.json", ROOT / "credentials.json"
    creds = Credentials.from_authorized_user_file(token, SCOPES) if token.exists() else None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None  # Testing-mode sign-ins expire after 7 days; sign in again
    if not creds or not creds.valid:
        if not creds_file.exists():
            raise SystemExit("credentials.json not found in ~/style-matcher. See the Gmail setup steps.")
        creds = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES).run_local_server(port=0)
        token.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def search(service):
    senders = " OR ".join(f"from:{w}" for words in STORES.values() for w in words)
    query = f"({senders}) subject:({SUBJECT_WORDS}) -category:promotions -category:social newer_than:{LOOKBACK}"
    ids, page = [], None
    while True:
        res = service.users().messages().list(userId="me", q=query, pageToken=page, maxResults=100).execute()
        ids += [m["id"] for m in res.get("messages", [])]
        page = res.get("nextPageToken")
        if not page:
            return ids


def html_body(payload):
    """Find the HTML part of an email, however deeply it's nested."""
    if payload.get("mimeType") == "text/html" and payload.get("body", {}).get("data"):
        return urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")
    for part in payload.get("parts", []) or []:
        found = html_body(part)
        if found:
            return found
    return None


# ---------------------------------------------------------------- parsing

def store_for(sender):
    s = sender.lower()
    return next((name for name, words in STORES.items() if any(w in s for w in words)), None)


def kind_of(subject):
    has_number = re.search(r"#\s*\d|order\s*(no\.?|number)\s*:?\s*\w*\d|\brefund", subject, re.I)
    if MARKETING.search(subject) and not has_number:
        return None
    if RETURN_WORDS.search(subject):
        return "return"
    if ORDER_WORDS.search(subject):
        return "order"
    return None


def image_id(src):
    return hashlib.sha1(src.split("?")[0].encode()).hexdigest()[:16]


def product_images(html, store_name=None):
    """Likely product photos in an email: skips logos, icons, tracking pixels and tiny images."""
    soup = BeautifulSoup(html, "html.parser")
    found = []
    for img in soup.find_all("img"):
        src = (img.get("src") or "").strip()
        if not src.startswith("http") or NOT_PRODUCT.search(src) or NOT_PRODUCT.search(img.get("alt") or ""):
            continue
        sizes = [int(v) for v in (img.get("width"), img.get("height")) if v and str(v).isdigit()]
        if sizes and min(sizes) < 60:
            continue
        alt = " ".join((img.get("alt") or "").split())
        if JUNK_TEXT.search(alt) or (alt and store_name and alt.lower() == store_name.lower()):
            continue
        title = "" if re.fullmatch(r"product( image)?\s*\d*", alt, re.I) else alt  # generic alt: look nearby
        if not title:  # fall back to the text in the same table row, which usually names the product
            row = img.find_parent("tr")
            title = " ".join(row.get_text(" ").split())[:90] if row else ""
        found.append({"image": src, "title": title})
    return found


def text_body(payload):
    """The email's plain-text part (or its HTML converted to text), as non-empty lines."""
    def find(p, mime):
        if p.get("mimeType") == mime and p.get("body", {}).get("data"):
            return urlsafe_b64decode(p["body"]["data"]).decode("utf-8", errors="replace")
        for part in p.get("parts", []) or []:
            got = find(part, mime)
            if got:
                return got
    text = find(payload, "text/plain")
    if not text:
        html = find(payload, "text/html")
        text = BeautifulSoup(html, "html.parser").get_text("\n") if html else ""
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


SHOPIFY_SIZE = re.compile(r"_(?:pico|icon|thumb|small|compact|medium|large|grande|\d+x\d*)(?:_crop_\w+|_cropped)?(?=\.\w+(?:\?|$))")


def full_size(src):
    """Shopify email images are small thumbnails (…_compact_cropped.jpg); dropping the suffix
    asks Shopify's image server for the full-size original."""
    return SHOPIFY_SIZE.sub("", src) if "cdn.shopify.com" in src else src


ITEM_LINE = re.compile(r"^(.+?)\s*[×x]\s*(\d+)$")  # Shopify receipts: "Ara Top × 1"


def receipt_items(lines):
    """Item name, quantity and variant ("White / XS") from a Shopify-style receipt."""
    items = []
    for i, line in enumerate(lines):
        m = ITEM_LINE.match(line)
        if not m:
            continue
        variant = lines[i + 1] if i + 1 < len(lines) and "/" in lines[i + 1] and "$" not in lines[i + 1] else ""
        items.append({"name": m.group(1).strip(), "qty": int(m.group(2)), "variant": variant})
    return items


_catalogs = {}


def catalog(store):
    """A Shopify store's products (fetched once per run, a few pages at most)."""
    if store not in _catalogs:
        products, base = [], SHOPIFY[store]
        for page in range(1, 5):
            for attempt in range(3):
                r = requests.get(f"{base}/products.json?limit=250&page={page}", headers=HEADERS, timeout=20)
                if r.status_code != 429:
                    break
                time.sleep(5 * (attempt + 1))
            r.raise_for_status()
            batch = r.json().get("products", [])
            if not batch:
                break
            products += batch
            time.sleep(1)
        _catalogs[store] = products
    return _catalogs[store]


def norm(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def match_product(store, item):
    """The catalog photo for a receipt item: exact name first, then a close match, preferring the
    photo of the variant bought (e.g. the White one)."""
    products = catalog(store)
    name = norm(item["name"])
    words = set(name.split())
    # Exact title first; otherwise the shortest title that contains the name or all of its words.
    product = next((p for p in products if norm(p["title"]) == name), None) or min(
        (p for p in products if name in norm(p["title"]) or norm(p["title"]) in name
         or words <= set(norm(p["title"]).split())),
        key=lambda p: len(p["title"]), default=None)
    if not product or not product.get("images"):
        return None
    color = norm(item["variant"].split("/")[0]) if item["variant"] else ""
    image = product["images"][0]
    if color:
        for v in product.get("variants", []):
            if color in norm(v.get("title", "")) and v.get("featured_image"):
                image = next((im for im in product["images"] if im["id"] == v["featured_image"]["id"]), image)
                break
    return {"image": image["src"], "title": f"{product['title']} — {item['variant']}".strip(" —"),
            "link": f"{SHOPIFY[store]}/products/{product['handle']}"}


def scan():
    service = gmail()
    candidates = {c["id"]: c for c in load(CANDIDATES, [])}
    seen = set(load(SEEN, []))
    new_ids = [i for i in search(service) if i not in seen]
    print(f"Found {len(new_ids)} new order/return emails to read.")

    per_store_emails, src_counts, fresh, unmatched = {}, {}, [], []
    for n, mid in enumerate(new_ids, 1):
        msg = service.users().messages().get(userId="me", id=mid, format="full").execute()
        headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
        store, kind = store_for(headers.get("from", "")), kind_of(headers.get("subject", ""))
        html = html_body(msg["payload"])
        seen.add(mid)
        if not (store and kind and html):
            continue
        try:
            when = parsedate_to_datetime(headers["date"]).date().isoformat()
        except Exception:
            when = None
        per_store_emails[store] = per_store_emails.get(store, 0) + 1
        found = product_images(html, store)
        if found and store in SHOPIFY:
            # Name the photos from the receipt when it lists the same number of items, in order.
            named = receipt_items(text_body(msg["payload"]))
            if len(named) == len(found):
                for item, r in zip(found, named):
                    item["title"] = item["title"] or f"{r['name']} — {r['variant']}".strip(" —")
        for item in found:
            key = (store, item["image"].split("?")[0])
            src_counts[key] = src_counts.get(key, 0) + 1
            fresh.append(dict(item, id=image_id(item["image"]), image=full_size(item["image"]), store=store,
                              kind=kind, date=when, subject=headers.get("subject", "")[:120]))
        if not found and store in SHOPIFY:
            # No photos in the email: read the item names from the receipt and find them in the
            # store's catalog instead.
            for item in receipt_items(text_body(msg["payload"])):
                try:
                    match = match_product(store, item)
                except Exception as e:
                    print(f"  {store}: couldn't read the catalog ({e}); try again later")
                    seen.discard(mid)  # retry this email next run
                    break
                if match:
                    fresh.append(dict(match, id=image_id(match["image"]), store=store, kind=kind, date=when,
                                      subject=headers.get("subject", "")[:120], source="receipt"))
                    print(f"  matched from receipt: {store} · {match['title']}")
                else:
                    unmatched.append(f"{store} · {item['name']} {item['variant']}".strip())
        if n % 20 == 0:
            print(f"  read {n}/{len(new_ids)}")

    added = 0
    for c in fresh:
        # An image that shows up in most of a store's emails is part of the layout, not a product.
        total, count = per_store_emails[c["store"]], src_counts.get((c["store"], c["image"].split("?")[0]), 0)
        if count >= 3 and count / total > 0.5:
            continue
        old = candidates.get(c["id"])
        if old and old["kind"] == "return":
            continue  # a return email outranks the order email for the same item
        added += old is None
        candidates[c["id"]] = c
    save(CANDIDATES, sorted(candidates.values(), key=lambda c: c["date"] or "", reverse=True))
    save(SEEN, sorted(seen))
    print(f"Added {added} new possible items.")
    for u in unmatched:
        print(f"  Couldn't find in the store's catalog: {u} — use 'I own this' on its product page")


# ---------------------------------------------------------------- review page

def review():
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    candidates = {c["id"]: c for c in load(CANDIDATES, [])}
    decisions = load(DECISIONS, {})
    todo = [c for c in candidates.values() if c["id"] not in decisions]
    if not todo:
        print("Nothing to review. Everything found so far has been sorted.")
        return

    app = FastAPI()
    # A return only counts against your taste if it wasn't your style.
    folders = {"purchased": "catalog/purchased",
               "returned_style": "catalog/returned",       # not my style: a real "no"
               "returned_fit": "catalog/returned_fit",     # fit or quality: not a taste signal
               "returned_need": "catalog/returned_need"}   # loved it, didn't need it: still a "yes"

    class Decision(BaseModel):
        id: str
        decision: str
        data: str | None = None  # the photo itself, when the browser could get it

    @app.post("/decide")
    def decide(d: Decision):
        c = candidates.get(d.id)
        if not c or d.decision not in (*folders, "skip"):
            raise HTTPException(400, "unknown item or decision")
        name = f"mail_{c['id']}.jpg"
        for folder in folders.values():  # changing your mind moves the photo instead of copying it
            (ROOT / folder / name).unlink(missing_ok=True)
        if d.decision != "skip":
            try:
                img = Image.open(io.BytesIO(photo_bytes(c["image"], d.data))).convert("RGB")
            except Exception as e:
                raise HTTPException(422, f"Couldn't get this photo ({e}).")
            dest = ROOT / folders[d.decision]
            dest.mkdir(parents=True, exist_ok=True)
            img.save(dest / name, "JPEG", quality=92)
            log("purchase" if d.decision == "purchased" else "return", store=c["store"], order_date=c["date"],
                item=c["id"], reason=d.decision.split("_", 1)[1] if d.decision.startswith("returned") else None)
        decisions[d.id] = d.decision
        save(DECISIONS, decisions)
        return {"ok": True}

    @app.get("/", response_class=HTMLResponse)
    def page():
        cards = "".join(card(c) for c in todo)
        return PAGE.replace("{{COUNT}}", str(len(todo))).replace("{{CARDS}}", cards)

    url = f"http://127.0.0.1:{REVIEW_PORT}"
    print(f"\nReview page: {url}  (press Ctrl+C here when you're done)")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=REVIEW_PORT, log_level="warning")
    print("Done. Click 'Retrain with new labels' in the extension popup to use what you confirmed.")


def photo_bytes(url, data=None):
    """The photo: from the browser if it sent it, else downloaded here (pretending to be Chrome)."""
    if data:
        return b64decode(data.split(",", 1)[1])
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code == 403:
        try:  # some stores block scripts by their network fingerprint; this mimics Chrome's
            from curl_cffi import requests as chrome
            r = chrome.get(url, impersonate="chrome", timeout=15)
        except ImportError:
            pass
    r.raise_for_status()
    return r.content


def card(c):
    kind = "Return email" if c["kind"] == "return" else "Order email"
    suggested = "returned" if c["kind"] == "return" else "purchased"
    buttons = "".join(
        f'<button data-d="{d}" class="{"suggest" if d == suggested else ""}">{label}</button>'
        for d, label in (("purchased", "Bought it"), ("returned", "Returned it"), ("skip", "Not a product")))
    buttons += ('<div class="why" hidden><span>Why?</span>'
                '<button data-d="returned_style">Not my style</button>'
                '<button data-d="returned_fit">Fit or quality</button>'
                '<button data-d="returned_need">Didn’t need it</button></div>')
    return f"""<article class="card" data-id="{c['id']}">
  <img src="{escape(c['image'])}" alt="" loading="lazy" referrerpolicy="no-referrer">
  <p class="meta">{escape(c['store'])} · {escape(c['date'] or '')} · <span class="kind">{kind}</span></p>
  <h2>{escape(c['title'] or c['subject'])}</h2>
  {f'<p class="meta">Matched from the receipt · <a href="{escape(c["link"])}" target="_blank" rel="noopener">check the product</a></p>' if c.get("source") == "receipt" else ''}
  <div class="actions">{buttons}</div>
  <p class="msg" role="status"></p>
</article>"""


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Confirm your purchases</title>
<style>
  :root { --ink:#2b2a33; --paper:#fdfdfb; --muted:#6d6b75; --line:rgba(43,42,51,.14); --keep:#41583a; --back:#8a4b3a; }
  @media (prefers-color-scheme: dark) { :root { --ink:#ecebef; --paper:#1d1c22; --muted:#a3a1ab; --line:rgba(236,235,239,.16); --keep:#7fa071; --back:#d08c78; } }
  body { margin:0; background:var(--paper); color:var(--ink); font:15px/1.45 "Avenir Next","Helvetica Neue",Helvetica,Arial,sans-serif; }
  main { max-width:1100px; margin:0 auto; padding:32px 20px 60px; }
  h1 { font-size:26px; font-weight:600; margin:0 0 4px; } .sub { color:var(--muted); margin:0; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:24px; margin-top:24px; }
  img { width:100%; aspect-ratio:3/4; object-fit:cover; border-radius:6px; display:block; background:var(--line); }
  .meta { font-size:12px; color:var(--muted); margin:10px 0 2px; } .kind { font-weight:600; }
  h2 { font-size:14px; font-weight:500; margin:0 0 8px; }
  .actions { display:flex; flex-wrap:wrap; gap:6px; }
  button { font:inherit; font-size:12px; padding:5px 9px; border-radius:7px; border:1px solid var(--line); background:transparent; color:var(--ink); cursor:pointer; }
  button.suggest { border-color:var(--ink); }
  button:hover { border-color:var(--ink); } button:focus-visible { outline:2px solid #4a7bd0; outline-offset:2px; }
  .card.purchased button[data-d=purchased] { background:var(--keep); border-color:var(--keep); color:#fff; }
  .card[class*=returned_] button[data-d=returned] { background:var(--back); border-color:var(--back); color:#fff; }
  .card.returned_style button[data-d=returned_style], .card.returned_fit button[data-d=returned_fit],
  .card.returned_need button[data-d=returned_need] { border-color:var(--back); color:var(--back); font-weight:600; }
  .why { display:flex; flex-wrap:wrap; gap:6px; align-items:center; width:100%; margin-top:2px; font-size:12px; color:var(--muted); }
  .why[hidden] { display:none; }
  .card.skip { opacity:.35; }
  .msg { font-size:12px; color:var(--back); margin:6px 0 0; min-height:1em; }
</style></head><body><main>
<h1>Confirm your purchases</h1>
<p class="sub">{{COUNT}} possible items found in your order and return emails. The outlined button is the guess from the email type. Press Ctrl+C in Terminal when you’re done.</p>
<div class="grid">{{CARDS}}</div></main>
<script>
// The photo is fetched by the browser where possible, since some stores (like Aritzia) block
// downloads that don't come from a browser. First choice: the Taste Score extension.
function viaExtension(url) {
  if (!document.documentElement.dataset.tasteBridge) return Promise.resolve(null);
  return new Promise(resolve => {
    const id = Math.random().toString(36).slice(2);
    const done = e => { if (e.source === window && e.data?.type === "taste-image" && e.data.id === id) {
      removeEventListener("message", done); resolve(e.data.data || null); } };
    addEventListener("message", done);
    postMessage({ type: "taste-fetch-image", id, url }, location.origin);
    setTimeout(() => { removeEventListener("message", done); resolve(null); }, 15000);
  });
}
async function viaPage(url) {
  try {
    const blob = await (await fetch(url, { mode: "cors", credentials: "omit" })).blob();
    return await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  } catch { return null; }
}
document.querySelectorAll(".card").forEach(card => card.addEventListener("click", async e => {
  const b = e.target.closest("button"); if (!b) return;
  const why = card.querySelector(".why");
  if (b.dataset.d === "returned") { why.hidden = false; return; }  // ask why before saving
  if (!b.dataset.d.startsWith("returned_")) why.hidden = true;
  const msg = card.querySelector(".msg"); msg.textContent = "Saving…";
  try {
    let data = null;
    if (b.dataset.d !== "skip") {
      const url = card.querySelector("img").src;
      data = (await viaExtension(url)) || (await viaPage(url));
    }
    const r = await fetch("/decide", { method:"POST", headers:{ "Content-Type":"application/json" },
      body: JSON.stringify({ id: card.dataset.id, decision: b.dataset.d, data }) });
    if (!r.ok) throw new Error((await r.json()).detail);
    card.classList.remove("purchased","returned_style","returned_fit","returned_need","skip");
    card.classList.add(b.dataset.d); msg.textContent = "";
  } catch (err) { msg.textContent = err.message; }
}));
</script></body></html>"""


if __name__ == "__main__":
    if "--review-only" not in sys.argv:
        scan()
    review()
