"""
weight_experiment.py — does counting purchases and returns as "strong" labels improve the taste model?

Compares several training setups, always scoring items the model did not train on
(cross-validation, repeated with different shuffles):
  - no purchase data: train on likes and dislikes only
  - purchases/returns counted like ordinary likes/dislikes (weight 1)
  - purchases/returns counted 2x, 3x (the server's current setting) and 5x

It reports held-out AUC on two groups separately:
  - everyday items: your like/dislike labels (does purchase data help the everyday ranking?)
  - purchases/returns: the items you actually bought (does it rank real purchases correctly?)

Run:  python weight_experiment.py
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from style_match import ROOT, list_images

MODEL_ID = "patrickjohncyh/fashion-clip"
CACHE_FILE = ROOT / ".embedding_cache.joblib"
WEIGHTS = [1, 2, 3, 5]
REPEATS = 10


def embeddings(paths):
    """Reuse the server's saved embeddings; compute only what's missing."""
    cache = joblib.load(CACHE_FILE) if CACHE_FILE.exists() else {}
    keys = [(str(p), p.stat().st_mtime) for p in paths]
    missing = [p for p, k in zip(paths, keys) if k not in cache]
    if missing:
        print(f"Embedding {len(missing)} images not seen by the server yet ...")
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
        from style_match import DEVICE, as_tensor
        model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        for i in range(0, len(missing), 32):
            batch = missing[i:i + 32]
            inputs = processor(images=[Image.open(p).convert("RGB") for p in batch], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                v = as_tensor(model.get_image_features(**inputs))
            v = (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()
            for p, vec in zip(batch, v):
                cache[(str(p), p.stat().st_mtime)] = vec
        joblib.dump(cache, CACHE_FILE)
    return np.stack([cache[k] for k in keys])


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def main():
    groups = {g: list_images(f"catalog/{g}") for g in ("like", "dislike", "purchased", "returned")}
    for g, paths in groups.items():
        print(f"{g:>10}: {len(paths)}")
    n_strong = len(groups["purchased"]) + len(groups["returned"])
    if n_strong < 5:
        raise SystemExit("\nNeed at least 5 purchases + returns for a meaningful comparison.")
    if not groups["returned"]:
        print("\nNote: no returns yet, so 'purchases/returns' AUC compares purchases against your dislikes.")

    paths = groups["like"] + groups["purchased"] + groups["dislike"] + groups["returned"]
    X = embeddings(paths)
    y = np.array([1] * (len(groups["like"]) + len(groups["purchased"]))
                 + [0] * (len(groups["dislike"]) + len(groups["returned"])))
    strong = np.array([p in set(groups["purchased"]) | set(groups["returned"]) for p in paths])

    setups = {"no purchase data": None, **{f"weight {w}" + (" (current)" if w == 3 else ""): w for w in WEIGHTS}}
    results = {name: {"everyday": [], "purchases": []} for name in setups}
    for seed in range(REPEATS):
        for name, w in setups.items():
            oof = np.zeros(len(y))
            for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y):
                if w is None:
                    tr = tr[~strong[tr]]  # leave purchase data out of training entirely
                    weights = np.ones(len(tr))
                else:
                    weights = np.where(strong[tr], w, 1.0)
                clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
                oof[te] = clf.fit(X[tr], y[tr], sample_weight=weights).predict_proba(X[te])[:, 1]
            results[name]["everyday"].append(auc(y[~strong], oof[~strong]))
            # Purchases/returns are judged against the opposite everyday labels too, so the AUC is defined.
            buy_mask = strong | (y == 0) if not groups["returned"] else strong
            results[name]["purchases"].append(auc(y[buy_mask], oof[buy_mask]))

    print(f"\nHeld-out AUC, averaged over {REPEATS} shuffles (± = spread). 0.50 = random.\n")
    print(f"  {'setup':<26}{'everyday items':>18}{'purchases/returns':>21}")
    for name, r in results.items():
        e, p = np.array(r["everyday"]), np.array(r["purchases"])
        print(f"  {name:<26}{np.nanmean(e):>12.3f} ± {np.nanstd(e):.3f}{np.nanmean(p):>14.3f} ± {np.nanstd(p):.3f}")
    print("\nDifferences smaller than the ± spread are likely noise. With few purchases, expect wide spreads.")


if __name__ == "__main__":
    main()
