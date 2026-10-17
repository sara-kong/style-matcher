"""
style_clusters.py — Step 6: find the separate aesthetics in your Pinterest images,
and test whether matching per cluster beats matching against one averaged style.

Uses the same folders as style_match.py and must sit next to it.
Run:   python style_clusters.py
Then:  open clusters.html
"""
import html
from urllib.parse import quote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from transformers import CLIPModel, CLIPProcessor

from style_match import (DEVICE, ROOT, TOP_K, embed_images, list_images,
                         normalize, precision_at_k, rank)

MODEL_ID = "patrickjohncyh/fashion-clip"
K = None       # None = pick the number of clusters automatically; or set e.g. K = 3
MAX_K = 6
COLORS = ["#e4572e", "#2e86ab", "#a23b72", "#f1a208", "#3bb273", "#6c4f9c"]


def choose_k(vecs):
    n = len(vecs)
    if K:
        return K, {}
    scores = {}
    for k in range(2, min(MAX_K, n - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(vecs)
        scores[k] = silhouette_score(vecs, labels, metric="cosine")
    return max(scores, key=scores.get), scores


def to_2d(vecs):
    n = len(vecs)
    if n < 10:  # too few points for UMAP
        return PCA(n_components=2).fit_transform(vecs)
    import umap
    reducer = umap.UMAP(n_neighbors=min(15, n - 1), min_dist=0.3, metric="cosine", random_state=0)
    return reducer.fit_transform(vecs)


def plot_map(points, labels, paths, out):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(points[:, 0], points[:, 1], s=0)
    for (x, y), lab, p in zip(points, labels, paths):
        thumb = Image.open(p).convert("RGB")
        thumb.thumbnail((70, 70))
        box = AnnotationBbox(OffsetImage(np.asarray(thumb)), (x, y), frameon=True,
                             bboxprops=dict(edgecolor=COLORS[lab % len(COLORS)], linewidth=3))
        ax.add_artist(box)
    ax.margins(0.12)  # room so edge thumbnails aren't cut off
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Your style map: nearby images look alike to FashionCLIP; border color = cluster")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def thumbs(items, liked=None):
    out = ["<div class='row'>"]
    for path, score in items:
        cls = "" if liked is None else ("like" if path in liked else "dislike")
        cap = "" if score is None else f"<figcaption>{score:.3f}</figcaption>"
        out.append(f"<figure class='{cls}'><img src='{quote(path.relative_to(ROOT).as_posix())}'>{cap}</figure>")
    out.append("</div>")
    return "".join(out)


def main():
    style = list_images("style")
    liked_list = list_images("catalog/like")
    catalog = liked_list + list_images("catalog/dislike")
    if len(style) < 4 or not catalog:
        raise SystemExit("Need at least 4 images in style/ and some in catalog/like + catalog/dislike.")
    liked = set(liked_list)

    print("Loading FashionCLIP ...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    style_vecs = embed_images(model, processor, style)
    catalog_vecs = embed_images(model, processor, catalog)
    sv = style_vecs.numpy()

    k, scores = choose_k(sv)
    if scores:
        print("Cluster fit (silhouette, higher = cleaner split):")
        for kk, s in scores.items():
            print(f"  k={kk}: {s:.3f}{'  <- chosen' if kk == k else ''}")
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(sv)

    plot_map(to_2d(sv), labels, style, ROOT / "clusters.png")

    # Baseline: one averaged style vector (same as style_match.py).
    single = rank(normalize(style_vecs.mean(dim=0)), catalog_vecs, catalog)
    p_single = precision_at_k(single, liked)

    # Multi-cluster: each catalog item scored by its closest cluster centroid.
    centroids = normalize(torch.stack([style_vecs[labels == c].mean(dim=0) for c in range(k)]))
    sims = catalog_vecs @ centroids.T          # catalog x clusters
    best_sim, best_cluster = sims.max(dim=1)
    top = best_sim.argsort(descending=True)[:TOP_K]
    multi = [(catalog[i], best_sim[i].item()) for i in top]
    p_multi = precision_at_k(multi, liked)

    print(f"\nPrecision@{TOP_K}: one averaged style = {p_single:.0%}, "
          f"{k} clusters = {p_multi:.0%}  (random baseline {len(liked) / len(catalog):.0%})")

    parts = [
        "<h1>Style clusters</h1>",
        f"<p>{k} clusters found in {len(style)} style images. Precision@{TOP_K}: "
        f"<b>one averaged style {p_single:.0%}</b> vs <b>{k} clusters {p_multi:.0%}</b> "
        f"(random baseline {len(liked) / len(catalog):.0%}).</p>",
        "<img class='map' src='clusters.png'>",
    ]
    for c in range(k):
        members = [(p, None) for p, lab in zip(style, labels) if lab == c]
        c_top = rank(centroids[c], catalog_vecs, catalog)
        parts.append(f"<h2 style='color:{COLORS[c % len(COLORS)]}'>Cluster {c + 1} "
                     f"({len(members)} images)</h2><h3>Your images</h3>{thumbs(members)}"
                     f"<h3>Top catalog matches for this cluster "
                     f"(precision@{TOP_K}: {precision_at_k(c_top, liked):.0%})</h3>{thumbs(c_top, liked)}")
    css = """
      body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }
      h3 { font-size: 13px; font-weight: 600; color: #666; margin: 12px 0 6px; }
      .map { max-width: 100%; border-radius: 8px; }
      .row { display: flex; flex-wrap: wrap; gap: 8px; }
      figure { margin: 0; width: 100px; text-align: center; font-size: 12px; }
      img:not(.map) { width: 100px; height: 133px; object-fit: cover; border-radius: 6px; border: 3px solid transparent; }
      .like img { border-color: #2e9e5b; } .dislike img { border-color: #d9534f; }
    """
    (ROOT / "clusters.html").write_text(
        f"<!doctype html><html><head><meta charset='utf-8'><title>Style clusters</title>"
        f"<style>{css}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")
    print("Open clusters.html to see your clusters.")


if __name__ == "__main__":
    main()
