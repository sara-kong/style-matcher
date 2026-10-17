"""
taste_rank.py — Train your taste model on catalog/like + catalog/dislike, then rank new items.

Put new product screenshots in candidates/ (next to this script), then:
  python taste_rank.py
  open ranked.html

Each run retrains on your current labels, so the model improves as you sort more items
into catalog/like and catalog/dislike. The trained model is also saved to taste_model.joblib.
"""
import html
from urllib.parse import quote

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import CLIPModel, CLIPProcessor

from style_match import DEVICE, ROOT, embed_images, list_images

MODEL_ID = "patrickjohncyh/fashion-clip"
TARGET_PRECISION = 0.8  # the cutoff is set so that ~80% of items above it are ones you'd like
FALLBACK_CUTOFF = 0.5


def classifier():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def pick_cutoff(proba, y):
    """Lowest cutoff where, on held-out predictions, at least TARGET_PRECISION of items above it are liked."""
    for t in np.sort(np.unique(proba)):
        above = proba >= t
        if above.sum() >= 5 and y[above].mean() >= TARGET_PRECISION:
            return float(t), int(above.sum()), float(y[above].mean())
    return FALLBACK_CUTOFF, None, None


def main():
    liked = list_images("catalog/like")
    disliked = list_images("catalog/dislike")
    candidates = list_images("candidates")
    if len(liked) < 5 or len(disliked) < 5:
        raise SystemExit("Need at least 5 images in each of catalog/like and catalog/dislike.")
    if not candidates:
        raise SystemExit("Put new product images in a candidates/ folder first.")

    print("Loading FashionCLIP ...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    X = embed_images(model, processor, liked + disliked).numpy()
    y = np.array([1] * len(liked) + [0] * len(disliked))

    # Choose the cutoff from held-out predictions, so it reflects how the model does on new items.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(classifier(), X, y, cv=cv, method="predict_proba")[:, 1]
    cutoff, n_above, prec = pick_cutoff(oof, y)
    if n_above:
        print(f"Cutoff {cutoff:.2f}: on held-out items, {prec:.0%} of the {n_above} above it were ones you like.")
    else:
        print(f"No cutoff reached {TARGET_PRECISION:.0%} precision yet; using {cutoff:.2f}. Label more items.")

    # Final model: trained on every labeled item.
    clf = classifier().fit(X, y)
    joblib.dump({"classifier": clf, "cutoff": cutoff, "model_id": MODEL_ID}, ROOT / "taste_model.joblib")

    proba = clf.predict_proba(embed_images(model, processor, candidates).numpy())[:, 1]
    order = np.argsort(-proba)
    keep = [i for i in order if proba[i] >= cutoff]

    print(f"\n{len(keep)} of {len(candidates)} candidates pass the cutoff:")
    for i in order:
        mark = "✓" if proba[i] >= cutoff else " "
        print(f"  {mark} {proba[i]:.2f}  {candidates[i].name}")

    def fig(i):
        cls = "pass" if proba[i] >= cutoff else "fail"
        return (f"<figure class='{cls}'><img src='{quote(candidates[i].relative_to(ROOT).as_posix())}'>"
                f"<figcaption>{proba[i]:.2f}<br>{html.escape(candidates[i].name)}</figcaption></figure>")

    css = """
      body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }
      .row { display: flex; flex-wrap: wrap; gap: 10px; }
      figure { margin: 0; width: 130px; text-align: center; font-size: 11px; word-break: break-all; }
      img { width: 130px; height: 173px; object-fit: cover; border-radius: 6px; }
      .pass img { border: 3px solid #2e9e5b; }
      .fail { opacity: 0.35; }
    """
    page = (f"<h1>Ranked candidates</h1><p>Number = predicted chance you'd like it. "
            f"Cutoff {cutoff:.2f}: {len(keep)} of {len(candidates)} pass.</p>"
            f"<h2>Worth a look</h2><div class='row'>{''.join(fig(i) for i in keep) or '<p>Nothing passed today.</p>'}</div>"
            f"<h2>Below the cutoff</h2><div class='row'>{''.join(fig(i) for i in order if proba[i] < cutoff)}</div>")
    (ROOT / "ranked.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Ranked candidates</title>"
        f"<style>{css}</style></head><body>{page}</body></html>", encoding="utf-8")
    print("\nOpen ranked.html to see them.")


if __name__ == "__main__":
    main()
