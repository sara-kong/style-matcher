# Taste Score

A personal fashion recommender built to spend less time window shopping and buy more intentionally.

It learns my taste from my own feedback, finds new pieces from stores I like, and hands over a short list of what's actually worth a look, while flagging anything too similar to what I already own.

> _Add a demo GIF or screenshot here: score tags on a real shopping page._

## What it does

- **Scores products while I browse.** A Chrome extension tags product images on any shopping site with a taste score, highlights what's worth a look, and fades the rest.
- **Hands over new arrivals.** A digest checks my favorite brands' Shopify feeds and shows only new items above my cutoff.
- **Learns from what I actually buy.** Reads order and return emails (read-only Gmail) so purchases and returns become training data. Returns are sorted by reason, because a return isn't the same as a dislike.
- **Checks my wardrobe.** Flags items that I own or that look like something I own, and never highlights them.
- **Measures itself.** Logs every score shown and every label given, and reports the real-world hit rate over time.

## How it works

```
 Sources                      Model                         Outputs
 ───────                      ─────                         ───────
 Browsing (extension)  ─┐
 Shopify feeds          ├──▶  FashionCLIP image embeddings  ──▶  Score tags in the browser
 Gmail orders/returns  ─┘      + personal logistic           ──▶  Digest of new arrivals
                               regression (taste model)      ──▶  Wardrobe check
                                        ▲                     ──▶  Usage report
                                        │
                     Like / Dislike / Maybe / I own this  (feedback → retrain)
```

- **Embeddings:** [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip) turns each product photo into a 512-number vector. It stays frozen.
- **Taste model:** logistic regression (513 parameters) trained on my labels. This is transfer learning with a linear probe.
- **Cutoff:** chosen automatically so that ~80% of highlighted items are hits on held-out data.
- **Everything runs locally.** A FastAPI server on my laptop does the scoring; nothing about my taste leaves my machine.

## What I found

I tested each idea before building on it. Every number below is held-out: scored on items the model didn't train on.

| Experiment | Result | Takeaway |
|---|---|---|
| Match products to my Pinterest saves (average style) | 50% precision@10 vs. 49% random | No better than random |
| Split my pins into style clusters | 50% | Didn't help |
| Nearest single pin | 50–60% | Barely helped |
| **Classifier trained on my likes/dislikes** | **AUC 0.86** (FashionCLIP) vs. 0.80 (CLIP) | My taste is learnable from direct feedback |
| Real-world batches of new items | Middle scores were coin flips at first | My early "dislikes" were too easy; labeling near-misses fixed the ranking |
| Weight purchases 1×–5× more than likes | Everyday AUC fell from 0.851 to 0.825 | Purchase weighting didn't help |
| Returns as strong "no"s | Model ranked my return *above* my keeps | A return isn't a dislike. Mine was my style; I just didn't need it |
| Pinterest as a cold start | Pins only: 0.42–0.52. Pins + 10 swipes: 0.56. **10 swipes alone: 0.71** | Pins hurt (editorial outfits vs. product photos: domain shift). Ten direct labels beat my whole board |

**Results from real use:** _add after a few weeks: hit rate above the cutoff, time spent on shopping sites, and orders/returns per month before vs. after._

## Project structure

| File | Purpose |
|---|---|
| `server.py` | Local scoring server: score, label, retrain, wardrobe check, logging |
| `extension/` | Chrome extension (Manifest V3): score tags and labeling on any site |
| `source_feeds.py` | Pulls new arrivals from Shopify brands and writes the digest |
| `purchases.py` | Finds orders and returns in Gmail; review page to confirm items |
| `usage_report.py` | Real-world scorecard from the event log |
| `style_match.py`, `style_clusters.py`, `style_learn.py` | Early experiments: model comparison, clustering, nearest-neighbor vs. classifier |
| `weight_experiment.py`, `pins_coldstart.py` | Purchase-weighting and Pinterest cold-start experiments |
| `taste_rank.py` | Ranks a folder of new images (pre-extension version) |
| `start.sh` | Runs the digest |

## Setup

Requires macOS or Linux, Python 3.11 or 3.12, and Chrome.

```bash
git clone https://github.com/<you>/style-matcher.git
cd style-matcher
python3 -m venv .venv && source .venv/bin/activate
pip install open_clip_torch transformers torch pillow scikit-learn umap-learn matplotlib \
            fastapi uvicorn requests beautifulsoup4
```

1. **Label a starting set.** Put product photos you like in `catalog/like/` and ones you don't in `catalog/dislike/`. About 20 of each is enough to start, and near-misses teach it the most.
2. **Start the server:** `python server.py`. It runs at `127.0.0.1:8000`.
3. **Load the extension:** `chrome://extensions` → Developer mode → Load unpacked → `extension/`.
4. **Browse.** Hover a tag to Like, Maybe, Dislike, or mark "I own this." Retrain from the popup.
5. **Digest:** add Shopify stores to `BRANDS` in `source_feeds.py`, then run `./start.sh`.
6. **Gmail (optional):** create a Desktop OAuth client with the Gmail API in Google Cloud (testing mode, read-only), save it as `credentials.json`, add stores to `STORES` in `purchases.py`, and run `python purchases.py`.

## Privacy

- Everything runs on your own machine.
- Gmail access is read-only. Only item photos and a few details are kept, never email content.
- `catalog/`, `logs/`, `purchases/`, `credentials.json`, and `token.json` are git-ignored.

## Limitations and next steps

- One user and a few hundred labels, so findings are early signals.
- Image-only: the model judges style, not fit or quality.
- Next: a wanted list (search for specific pieces), automatic weekly delivery, more sources, and a quality signal from fabric descriptions.
