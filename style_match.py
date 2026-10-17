"""
style_match.py — Steps 4 & 5: compare CLIP vs FashionCLIP, test cropping, test text queries.

Folder layout (next to this script):
  style/              your Pinterest images
  style_cropped/      (optional) the same images, cropped to just the clothing
  catalog/like/       catalog images you'd actually wear
  catalog/dislike/    catalog images you wouldn't

Run:   python style_match.py
Then:  open report.html
"""
import html
from pathlib import Path
from urllib.parse import quote

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODELS = {
    "CLIP (OpenAI ViT-B/32)": "openai/clip-vit-base-patch32",
    "FashionCLIP": "patrickjohncyh/fashion-clip",
}
# Edit these to describe your own style.
TEXT_QUERIES = [
    "chic outfit inspo",
    "elegant minimalist outfit in neutral tones",
    "elevated form-fitting tops with stylistic feature",
]
TOP_K = 10
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
ROOT = Path(__file__).resolve().parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def list_images(folder):
    """All readable images in a folder (skips broken files with a warning)."""
    folder = ROOT / folder
    if not folder.exists():
        return []
    paths = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        try:
            Image.open(p).verify()
            paths.append(p)
        except Exception:
            print(f"  skipping unreadable image: {p.name}")
    return paths


def as_tensor(out):
    # Older transformers return a tensor; newer versions wrap it in an output object.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def normalize(v):
    return v / v.norm(dim=-1, keepdim=True)


def embed_images(model, processor, paths, batch_size=32):
    vecs = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vecs.append(normalize(as_tensor(model.get_image_features(**inputs))))
    return torch.cat(vecs).cpu()


def embed_text(model, processor, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        return normalize(as_tensor(model.get_text_features(**inputs))).cpu()[0]


def rank(query_vec, catalog_vecs, catalog_paths):
    scores = catalog_vecs @ query_vec  # cosine similarity, since everything is normalized
    top = scores.argsort(descending=True)[:TOP_K]
    return [(catalog_paths[i], scores[i].item()) for i in top]


def precision_at_k(results, liked):
    return sum(path in liked for path, _ in results) / len(results)


def write_report(runs, liked, baseline):
    parts = [
        "<h1>Style match report</h1>",
        f"<p>Green border = in catalog/like, red = in catalog/dislike. "
        f"Random baseline: {baseline:.0%} of the catalog is liked.</p>",
    ]
    for title, p, results in runs:
        score = "" if p is None else f" — precision@{TOP_K}: {p:.0%}"
        parts.append(f"<h2>{html.escape(title)}{score}</h2><div class='row'>")
        for path, s in results:
            cls = "like" if path in liked else "dislike"
            src = quote(path.relative_to(ROOT).as_posix())
            parts.append(f"<figure class='{cls}'><img src='{src}'>"
                         f"<figcaption>{s:.3f}</figcaption></figure>")
        parts.append("</div>")
    css = """
      body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }
      h2 { font-size: 16px; margin-top: 28px; }
      .row { display: flex; flex-wrap: wrap; gap: 8px; }
      figure { margin: 0; width: 120px; text-align: center; font-size: 12px; }
      img { width: 120px; height: 160px; object-fit: cover; border-radius: 6px; border: 3px solid transparent; }
      .like img { border-color: #2e9e5b; }
      .dislike img { border-color: #d9534f; }
    """
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>Style match report</title>"
            f"<style>{css}</style></head><body>{''.join(parts)}</body></html>")
    (ROOT / "report.html").write_text(page, encoding="utf-8")


def main():
    style = list_images("style")
    style_cropped = list_images("style_cropped")
    liked_list = list_images("catalog/like")
    catalog = liked_list + list_images("catalog/dislike")
    if not style or not catalog:
        raise SystemExit("Add images to style/ and to catalog/like + catalog/dislike first.")
    liked = set(liked_list)
    baseline = len(liked) / len(catalog)
    print(f"{len(style)} style images, {len(catalog)} catalog images "
          f"(random baseline: {baseline:.0%} liked)\n")

    runs = []  # (title, precision or None, results)
    for name, model_id in MODELS.items():
        print(f"Loading {name} (first run downloads it) ...")
        model = CLIPModel.from_pretrained(model_id).to(DEVICE).eval()
        processor = CLIPProcessor.from_pretrained(model_id)
        catalog_vecs = embed_images(model, processor, catalog)

        # Step 4 (and 5a): match your averaged style against the catalog, scored.
        variants = {"full Pinterest images": style}
        if style_cropped:
            variants["cropped to clothing"] = style_cropped
        for label, paths in variants.items():
            style_vec = normalize(embed_images(model, processor, paths).mean(dim=0))
            results = rank(style_vec, catalog_vecs, catalog)
            runs.append((f"{name} · {label}", precision_at_k(results, liked), results))

        # Step 5b: text queries. Not scored; check by eye whether results match the words.
        for q in TEXT_QUERIES:
            results = rank(embed_text(model, processor, q), catalog_vecs, catalog)
            runs.append((f'{name} · text: "{q}"', None, results))

    print("\nPrecision@10 (higher is better):")
    for title, p, _ in runs:
        if p is not None:
            print(f"  {p:4.0%}  {title}")
    write_report(runs, liked, baseline)
    print("\nOpen report.html to see the matches.")


if __name__ == "__main__":
    main()
