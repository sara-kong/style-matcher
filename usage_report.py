"""
usage_report.py — your real-world scorecard, from what you actually did in the browser.

Reads logs/events.jsonl (written by server.py) and reports how well the model's scores
predicted the labels you gave. Every score here was shown to you *before* you labeled
the item, so these are honest held-out results, not cross-validation.

Run:   python usage_report.py
Then:  open usage.png
"""
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "logs" / "events.jsonl"
WINDOW = 25  # labels per rolling window in the chart


def summarize(rows):
    y = np.array([r["label"] == "like" for r in rows])
    s = np.array([r["score"] for r in rows])
    above = s >= np.array([r["cutoff"] for r in rows])
    return {
        "n": len(rows),
        "liked": y.mean(),
        "auc": roc_auc_score(y, s) if 0 < y.sum() < len(y) else None,
        "n_above": int(above.sum()),
        "precision": y[above].mean() if above.any() else None,   # of items it recommended, share you liked
        "missed": y[~above].mean() if (~above).any() else None,  # of items it faded, share you liked anyway
    }


def pct(x):
    return "  –  " if x is None else f"{x:5.0%}"


def main():
    if not LOG_FILE.exists():
        raise SystemExit("No log yet. Browse and label a few items with the extension first.")
    events = [json.loads(line) for line in LOG_FILE.read_text().splitlines() if line.strip()]
    scored = [e for e in events if e["event"] == "score"]
    labeled = [e for e in events if e["event"] == "label" and e["label"] in ("like", "dislike")
               and e.get("score") is not None]
    retrains = [e for e in events if e["event"] == "retrain"]

    print(f"Items scored: {len(scored)}  ·  passed the cutoff: "
          f"{sum(e['score'] >= e['cutoff'] for e in scored) / max(len(scored), 1):.0%}")
    sites = Counter(e["site"] for e in scored if e.get("site"))
    if sites:
        print("Most-scored sites: " + ", ".join(f"{s} ({n})" for s, n in sites.most_common(5)))
    print(f"Labels given: {len(labeled)} like/dislike  ·  retrains: {len(retrains)}\n")
    if len(labeled) < 5:
        raise SystemExit("Label at least 5 items in the browser to see a scorecard.")

    overall = summarize(labeled)
    print("Overall (scores shown before you labeled each item):")
    print(f"  AUC {overall['auc']:.2f}" if overall["auc"] is not None else "  AUC –")
    print(f"  Of {overall['n_above']} items above the cutoff, you liked {pct(overall['precision']).strip()}")
    print(f"  Of items below the cutoff, you liked {pct(overall['missed']).strip()} anyway")
    print(f"  Share of everything you labeled that you liked: {overall['liked']:.0%}\n")

    weeks = defaultdict(list)
    for r in labeled:
        weeks[datetime.fromisoformat(r["t"]).strftime("%G-W%V")].append(r)
    print(f"  {'week':<10}{'labels':>7}{'AUC':>7}{'hit rate above cutoff':>24}{'liked below':>13}")
    for w in sorted(weeks):
        m = summarize(weeks[w])
        auc = "  –" if m["auc"] is None else f"{m['auc']:.2f}"
        print(f"  {w:<10}{m['n']:>7}{auc:>7}{pct(m['precision']):>24}{pct(m['missed']):>13}")

    # Rolling chart: does the hit rate above the cutoff improve as you keep labeling?
    xs, prec, base = [], [], []
    for i in range(WINDOW, len(labeled) + 1):
        m = summarize(labeled[i - WINDOW:i])
        xs.append(i)
        prec.append(np.nan if m["precision"] is None else m["precision"])
        base.append(m["liked"])
    if xs:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(xs, prec, label="Hit rate above cutoff", color="#41583a", linewidth=2)
        ax.plot(xs, base, label="Share of all labeled items you liked", color="#9a98a3", linestyle="--")
        times = [r["t"] for r in labeled]
        for e in retrains:
            k = sum(t <= e["t"] for t in times)
            if xs[0] <= k <= xs[-1]:
                ax.axvline(k, color="#d4d2da", linewidth=1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(f"Labels given (rolling window of {WINDOW}; grey lines = retrains)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(loc="lower right", frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(ROOT / "usage.png", dpi=130)
        print("\nSaved usage.png")
    else:
        print(f"\nThe chart appears after {WINDOW} labels.")


if __name__ == "__main__":
    main()
