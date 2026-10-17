"""
style_learn.py — Two tests of whether the embeddings capture your taste.

  A) Nearest-neighbor matching: score each catalog item by its closest single
     Pinterest image (and by the average of its 3 closest), instead of an average style.
  B) Learn from your labels: train a simple classifier on catalog/like vs catalog/dislike,
     always tested on items it did not train on (cross-validation).

Uses the same folders as style_match.py and must sit next to it.
Run:   python style_learn.py
Then:  open learn.html
"""
import html
from urllib.parse import quote

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from transformers import CLIPModel, CLIPProcessor

from style_match import (DEVICE, MODELS, ROOT, TOP_K, embed_images, list_images,
                         precision_at_k)

FOLDS = 5
REPEATS = 5  # cross-validation is repeated with different shuffles, then averaged


def top_k(scores, catalog):
    order = np.argsort(-scores)[:TOP_K]
    return [(catalog[i], float(scores[i])) for i in order]


def classifier():
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


def cross_validate(X, y):
    accs, aucs = [], []
    for seed in range(REPEATS):
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        proba = cross_val_predict(classifier(), X, y, cv=cv, method="predict_proba")[:, 1]
        accs.append(((proba >= 0.5) == y).mean())
        aucs.append(roc_auc_score(y, proba))
    # One out-of-fold run kept for ranking: every score comes from a model that never saw that item.
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)
    proba = cross_val_predict(classifier(), X, y, cv=cv, method="predict_proba")[:, 1]
    return np.mean(accs), np.std(accs), np.mean(aucs), proba


def thumbs(items, liked):
    out = ["<div class='row'>"]
    for path, score in items:
        cls = "like" if path in liked else "dislike"
        out.append(f"<figure class='{cls}'><img src='{quote(path.relative_to(ROOT).as_posix())}'>"
                   f"<figcaption>{score:.3f}</figcaption></figure>")
    out.append("</div>")
    return "".join(out)


def main():
    style = list_images("style")
    liked_list = list_images("catalog/like")
    disliked_list = list_images("catalog/dislike")
    catalog = liked_list + disliked_list
    if not style or len(liked_list) < FOLDS or len(disliked_list) < FOLDS:
        raise SystemExit(f"Need style images and at least {FOLDS} images in each of catalog/like and catalog/dislike.")
    liked = set(liked_list)
    y = np.array([1] * len(liked_list) + [0] * len(disliked_list))
    baseline = y.mean()
    majority = max(baseline, 1 - baseline)
    print(f"{len(style)} style images, {len(catalog)} catalog images "
          f"(random precision baseline {baseline:.0%}, always-guess-majority accuracy {majority:.0%})\n")

    rows, sections = [], []
    for name, model_id in MODELS.items():
        print(f"Loading {name} ...")
        model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        S = embed_images(model, processor, style).numpy()
        C = embed_images(model, processor, catalog).numpy()
        sims = C @ S.T  # catalog x style

        mean = S.mean(axis=0)
        avg = top_k(C @ (mean / np.linalg.norm(mean)), catalog)
        nn1 = top_k(sims.max(axis=1), catalog)
        nn3 = top_k(np.sort(sims, axis=1)[:, -3:].mean(axis=1), catalog)
        acc, acc_sd, auc, proba = cross_validate(C, y)
        clf = top_k(proba, catalog)

        rows.append((name, precision_at_k(avg, liked), precision_at_k(nn1, liked),
                     precision_at_k(nn3, liked), precision_at_k(clf, liked), acc, acc_sd, auc))

        wrong = np.argsort(-np.abs(proba - y))[:TOP_K]
        misses = [(catalog[i], float(proba[i])) for i in wrong]
        sections.append(
            f"<h2>{html.escape(name)}</h2>"
            f"<h3>A) Nearest neighbor, closest single pin: top {TOP_K}</h3>{thumbs(nn1, liked)}"
            f"<h3>B) Classifier: top {TOP_K} by predicted 'like' probability (held-out)</h3>{thumbs(clf, liked)}"
            f"<h3>Classifier's most confident mistakes (number = predicted chance you like it)</h3>{thumbs(misses, liked)}")

    print(f"\nPrecision@{TOP_K} (random baseline {baseline:.0%}):")
    print(f"  {'model':<24}{'average':>9}{'NN-1':>7}{'NN-3':>7}{'classif.':>10}")
    for name, p_avg, p_nn1, p_nn3, p_clf, *_ in rows:
        print(f"  {name:<24}{p_avg:>9.0%}{p_nn1:>7.0%}{p_nn3:>7.0%}{p_clf:>10.0%}")
    print(f"\nClassifier on held-out items (majority-guess accuracy {majority:.0%}, AUC 0.50 = random):")
    for name, *_, acc, acc_sd, auc in rows:
        print(f"  {name:<24}accuracy {acc:.0%} (±{acc_sd:.0%})   AUC {auc:.2f}")

    table = ["<table><tr><th>Model</th><th>Average style</th><th>Nearest pin</th><th>Nearest 3 pins</th>"
             "<th>Classifier</th><th>Classifier accuracy</th><th>Classifier AUC</th></tr>"]
    for name, p_avg, p_nn1, p_nn3, p_clf, acc, acc_sd, auc in rows:
        table.append(f"<tr><td>{html.escape(name)}</td><td>{p_avg:.0%}</td><td>{p_nn1:.0%}</td>"
                     f"<td>{p_nn3:.0%}</td><td>{p_clf:.0%}</td><td>{acc:.0%} ± {acc_sd:.0%}</td><td>{auc:.2f}</td></tr>")
    table.append("</table>")
    css = """
      body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }
      table { border-collapse: collapse; margin: 12px 0; } td, th { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
      td:first-child, th:first-child { text-align: left; }
      h3 { font-size: 13px; font-weight: 600; color: #666; margin: 14px 0 6px; }
      .row { display: flex; flex-wrap: wrap; gap: 8px; }
      figure { margin: 0; width: 100px; text-align: center; font-size: 12px; }
      img { width: 100px; height: 133px; object-fit: cover; border-radius: 6px; border: 3px solid transparent; }
      .like img { border-color: #2e9e5b; } .dislike img { border-color: #d9534f; }
    """
    intro = (f"<h1>Does the model capture your taste?</h1><p>Precision@{TOP_K} columns: random baseline "
             f"{baseline:.0%}. Classifier accuracy: always guessing the bigger folder scores {majority:.0%}. "
             f"AUC: 0.50 is random, 1.00 is perfect.</p>")
    (ROOT / "learn.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Taste test</title><style>{css}</style>"
        f"</head><body>{intro}{''.join(table)}{''.join(sections)}</body></html>", encoding="utf-8")
    print("\nOpen learn.html for the details.")


if __name__ == "__main__":
    main()
