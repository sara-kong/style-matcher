"""
debug_email.py — shows why purchases.py found no product photos in your newest order email.

Prints the email's subject and sender, then every image in it and whether it was kept or
why it was skipped. Prints nothing else from the email.
Run:  python debug_email.py
"""
import re

from bs4 import BeautifulSoup

import purchases as P

service = P.gmail()
ids = P.search(service)
if not ids:
    raise SystemExit("No matching emails found.")
msg = service.users().messages().get(userId="me", id=ids[0], format="full").execute()
headers = {h["name"].lower(): h["value"] for h in msg["payload"].get("headers", [])}
store, kind = P.store_for(headers.get("from", "")), P.kind_of(headers.get("subject", ""))
print(f"Subject: {headers.get('subject')}\nFrom:    {headers.get('from')}\nStore: {store} · Type: {kind}")

html = P.html_body(msg["payload"])
if not html:
    raise SystemExit("This email has no HTML part, so there are no images to find.")

imgs = BeautifulSoup(html, "html.parser").find_all("img")
print(f"\n{len(imgs)} images in the email:\n")
for i, img in enumerate(imgs, 1):
    src = (img.get("src") or "").strip()
    alt = " ".join((img.get("alt") or "").split())
    w, h = img.get("width"), img.get("height")
    sizes = [int(v) for v in (w, h) if v and str(v).isdigit()]
    if not src.startswith("http"):
        why = "skipped: not a web link"
    elif P.NOT_PRODUCT.search(src) or P.NOT_PRODUCT.search(alt):
        hit = (P.NOT_PRODUCT.search(src) or P.NOT_PRODUCT.search(alt)).group(0)
        why = f"skipped: looks like a logo/icon (matched '{hit}')"
    elif sizes and min(sizes) < 60:
        why = f"skipped: too small ({w}x{h})"
    elif P.JUNK_TEXT.search(alt):
        why = "skipped: store graphic text"
    else:
        why = "KEPT"
    print(f"{i:>2}. {why}\n    alt: {alt[:70]!r}  size: {w}x{h}\n    src: {re.sub(r'[?].*', '?…', src)[:110]}\n")
