"""
pins_coldstart.py — can a Pinterest import give a brand-new user a useful taste model?

Pretends you're a new user with only your pins (style/) and no labels yet, then scores
each approach against your real labels, which it never trains on for the items it scores.

  Pins only (no labels at all):
    - similarity to average pin   the original Pinterest approach
    - similarity to closest pin   nearest-neighbor version
    - pins-only classifier        pins = yes; other catalog photos, labels hidden, = rough "no"s
                                  (positive-unlabeled learning)
  Pins + a quick onboarding round (the user swipes like/dislike on N items):
    - pins + N swipes,  compared with  N swipes alone
  Reference:
    - full model trained on all your labels

Run:  python pins_coldstart.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from style_match import list_images
from weight_experiment import embeddings

REPEATS = 10
SWIPES = [10, 20, 40]


def clf():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def sample_swipes(rng, idx, y, n):
    """n labeled items from the training part, half likes and half dislikes where possible."""
    pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
    k = min(n // 2, len(pos), len(neg))
    return np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])


def main():
    pins = list_images("style")
    yes = list_images("catalog/like") + list_images("catalog/purchased") + list_images("catalog/returned_need")
    no = list_images("catalog/dislike") + list_images("catalog/returned")
    print(f"{len(pins)} pins · {len(yes)} yes labels · {len(no)} no labels")
    if len(pins) < 5 or min(len(yes), len(no)) < 10:
        raise SystemExit("Need at least 5 pins in style/ and 10 likes and 10 dislikes.")

    P = embeddings(pins)
    X = embeddings(yes + no)
    y = np.array([1] * len(yes) + [0] * len(no))
    mean_pin = P.mean(axis=0) / np.linalg.norm(P.mean(axis=0))

    names = ["similarity to average pin", "similarity to closest pin", "pins-only classifier"]
    names += [f"{n} swipes alone" for n in SWIPES] + [f"pins + {n} swipes" for n in SWIPES]
    names += ["full model (all labels)"]
    results = {n: [] for n in names}

    for seed in range(REPEATS):
        rng = np.random.default_rng(seed)
        scores = {n: np.zeros(len(y)) for n in names}
        for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y):
            scores["similarity to average pin"][te] = X[te] @ mean_pin
            scores["similarity to closest pin"][te] = (X[te] @ P.T).max(axis=1)

            # Positive-unlabeled: pins are "yes"; the training photos, with labels hidden, are "no".
            Xpu = np.vstack([P, X[tr]])
            ypu = np.r_[np.ones(len(P)), np.zeros(len(tr))]
            scores["pins-only classifier"][te] = clf().fit(Xpu, ypu).predict_proba(X[te])[:, 1]

            for n in SWIPES:
                s = sample_swipes(rng, tr, y, n)
                scores[f"{n} swipes alone"][te] = clf().fit(X[s], y[s]).predict_proba(X[te])[:, 1]
                Xb = np.vstack([P, X[s]])
                yb = np.r_[np.ones(len(P)), y[s]]
                scores[f"pins + {n} swipes"][te] = clf().fit(Xb, yb).predict_proba(X[te])[:, 1]

            scores["full model (all labels)"][te] = clf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        for n in names:
            results[n].append(roc_auc_score(y, scores[n]))

    print(f"\nHeld-out AUC on your real labels, averaged over {REPEATS} shuffles. 0.50 = random.\n")
    groups = [("Pins only, no labels", names[:3]),
              ("Quick onboarding", [m for n in SWIPES for m in (f"{n} swipes alone", f"pins + {n} swipes")]),
              ("Reference", names[-1:])]
    for title, group in groups:
        print(f"  {title}")
        for n in group:
            r = np.array(results[n])
            print(f"    {n:<30}{r.mean():.3f} ± {r.std():.3f}")
        print()


if __name__ == "__main__":
    main()
