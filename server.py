"""
server.py — local scoring server for the Taste Score browser extension.

Keeps FashionCLIP and your taste model loaded and answers requests from the extension:
  POST /score    {"urls": [...], "page": "..."}       -> {"scores": {url: 0.68, ...}, "cutoff": 0.54}
  POST /label    {"url": "...", "label": "like|dislike|maybe", "page": "..."}
  POST /retrain                                        -> retrains on catalog/like + catalog/dislike
  GET  /status                                         -> label counts, cutoff, AUC

Put this file in ~/style-matcher next to style_match.py, then run:
  python server.py
Only your own computer can reach it (it listens on 127.0.0.1).
"""
import base64
import hashlib
import io
import threading
import time

import joblib
import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import CLIPModel, CLIPProcessor

from style_match import DEVICE, ROOT, as_tensor, list_images

MODEL_ID = "patrickjohncyh/fashion-clip"
TARGET_PRECISION = 0.8
FALLBACK_CUTOFF = 0.5
MAX_IMAGE_BYTES = 10 * 1024 * 1024
CACHE_FILE = ROOT / ".embedding_cache.joblib"
LABELS = ("like", "dislike", "maybe")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

print("Loading FashionCLIP ...")
clip = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
processor = CLIPProcessor.from_pretrained(MODEL_ID)
lock = threading.Lock()  # one model, one request at a time

state = {"clf": None, "cutoff": FALLBACK_CUTOFF, "auc": None, "counts": {}}
url_vectors = {}  # image URL -> embedding, so retraining can rescore without re-downloading
file_cache = joblib.load(CACHE_FILE) if CACHE_FILE.exists() else {}  # (path, mtime) -> embedding


def embed(images):
    """PIL images -> normalized FashionCLIP vectors (numpy)."""
    out = []
    for i in range(0, len(images), 32):
        inputs = processor(images=images[i:i + 32], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            v = as_tensor(clip.get_image_features(**inputs))
        out.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def embed_files(paths):
    keys = [(str(p), p.stat().st_mtime) for p in paths]
    missing = [p for p, k in zip(paths, keys) if k not in file_cache]
    if missing:
        print(f"  embedding {len(missing)} new catalog images ...")
        vecs = embed([Image.open(p).convert("RGB") for p in missing])
        for p, v in zip(missing, vecs):
            file_cache[(str(p), p.stat().st_mtime)] = v
        joblib.dump(file_cache, CACHE_FILE)
    return np.stack([file_cache[k] for k in keys])


def fetch_image(url, page=None):
    """Download (or decode) an image URL into raw bytes."""
    if url.startswith("data:image/"):
        return base64.b64decode(url.split(",", 1)[1])
    if not url.startswith(("http://", "https://")):
        raise ValueError("unsupported URL")
    headers = dict(HEADERS, Referer=page) if page else HEADERS
    r = requests.get(url, headers=headers, timeout=10, stream=True)
    r.raise_for_status()
    data = r.raw.read(MAX_IMAGE_BYTES + 1, decode_content=True)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image too large")
    return data


def classifier():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def train():
    liked, disliked = list_images("catalog/like"), list_images("catalog/dislike")
    state["counts"] = {"like": len(liked), "dislike": len(disliked),
                       "maybe": len(list_images("catalog/maybe"))}
    if len(liked) < 5 or len(disliked) < 5:
        state["clf"] = None
        print("Need at least 5 liked and 5 disliked images to train.")
        return
    X = embed_files(liked + disliked)
    y = np.array([1] * len(liked) + [0] * len(disliked))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(classifier(), X, y, cv=cv, method="predict_proba")[:, 1]
    cutoff = FALLBACK_CUTOFF
    for t in np.sort(np.unique(oof)):  # lowest cutoff with >= TARGET_PRECISION on held-out items
        above = oof >= t
        if above.sum() >= 5 and y[above].mean() >= TARGET_PRECISION:
            cutoff = float(t)
            break

    state.update(clf=classifier().fit(X, y), cutoff=cutoff, auc=float(roc_auc_score(y, oof)))
    print(f"Trained on {len(liked)} liked / {len(disliked)} disliked. "
          f"Held-out AUC {state['auc']:.2f}, cutoff {cutoff:.2f}")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ScoreRequest(BaseModel):
    urls: list[str]
    page: str | None = None


class LabelRequest(BaseModel):
    url: str
    label: str
    page: str | None = None


@app.get("/status")
def status():
    return {"ready": state["clf"] is not None, "cutoff": state["cutoff"],
            "auc": state["auc"], "counts": state["counts"]}


@app.post("/score")
def score(req: ScoreRequest):
    if state["clf"] is None:
        raise HTTPException(409, "Model not trained yet: add at least 5 liked and 5 disliked images.")
    urls = list(dict.fromkeys(req.urls))[:60]
    todo, images = [], []
    for u in urls:
        if u in url_vectors:
            continue
        try:
            images.append(Image.open(io.BytesIO(fetch_image(u, req.page))).convert("RGB"))
            todo.append(u)
        except Exception:
            pass  # unreachable or not an image: just skip it
    with lock:
        if images:
            for u, v in zip(todo, embed(images)):
                url_vectors[u] = v
        known = [u for u in urls if u in url_vectors]
        scores = {}
        if known:
            proba = state["clf"].predict_proba(np.stack([url_vectors[u] for u in known]))[:, 1]
            scores = {u: round(float(p), 3) for u, p in zip(known, proba)}
    return {"scores": scores, "cutoff": state["cutoff"]}


@app.post("/label")
def label(req: LabelRequest):
    if req.label not in LABELS:
        raise HTTPException(400, f"label must be one of {LABELS}")
    try:
        img = Image.open(io.BytesIO(fetch_image(req.url, req.page))).convert("RGB")
    except Exception as e:
        raise HTTPException(422, f"Couldn't download that image: {e}")
    name = "web_" + hashlib.sha1(req.url.encode()).hexdigest()[:16] + ".jpg"
    for other in LABELS:  # relabeling moves the image rather than duplicating it
        (ROOT / "catalog" / other / name).unlink(missing_ok=True)
    dest = ROOT / "catalog" / req.label
    dest.mkdir(parents=True, exist_ok=True)
    img.save(dest / name, "JPEG", quality=92)
    state["counts"] = {k: len(list_images(f"catalog/{k}")) for k in LABELS}
    return {"saved": f"catalog/{req.label}/{name}", "counts": state["counts"]}


@app.post("/retrain")
def retrain():
    start = time.time()
    with lock:
        train()
    return dict(status(), seconds=round(time.time() - start, 1))


if __name__ == "__main__":
    with lock:
        train()
    print("Scoring server running at http://127.0.0.1:8000 — leave this window open while you browse.")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
