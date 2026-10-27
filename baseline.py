import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAINING_START = date(2026, 10, 17)   # the day you started using the tool
TEST_START = date(2026, 10, 27)       # the day you started using it as intended

# Sites that count as shopping/browsing. A visit counts if the site's address ends with one of these.
SHOPPING_SITES = [
    "aritzia.com", "withjean.com", "danielleguiziony.com", "sezane.com",
    "taobao.com", "tmall.com", "pinterest.com", "natiboutique.com", "cos.com",
    # add more: "thefrankieshop.com", "mirrorpalais.com", ...
]

ROOT = Path(__file__).resolve().parent
CHROME = Path.home() / "Library/Application Support/Google/Chrome"
CHROME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


def period(d):
    return "baseline" if d < TRAINING_START else "training" if d < TEST_START else "test"


def is_shopping(url):
    host = (urlparse(url).hostname or "").lower()
    return any(host == s or host.endswith("." + s) for s in SHOPPING_SITES)


def chrome_visits():
    """Every visit to a shopping site, from every Chrome profile, as local dates."""
    files = [p / "History" for p in CHROME.glob("*") if (p / "History").exists()]
    if not files:
        print("No Chrome history found; skipping browsing.")
        return []
    days, earliest = [], None
    for f in files:
        with tempfile.TemporaryDirectory() as tmp:  # Chrome locks its file, so read a copy
            copy = Path(tmp) / "History"
            shutil.copy2(f, copy)
            con = sqlite3.connect(copy)
            rows = con.execute("SELECT urls.url, visits.visit_time FROM visits "
                               "JOIN urls ON visits.url = urls.id").fetchall()
            con.close()
        for url, t in rows:
            when = (CHROME_EPOCH + timedelta(microseconds=t)).astimezone().date()
            earliest = when if earliest is None or when < earliest else earliest
            if is_shopping(url):
                days.append(when)
    return days, earliest


def purchases():
    """(date, kind) for every item confirmed as bought or returned in purchases.py."""
    cands = {c["id"]: c for c in json.loads((ROOT / "purchases/candidates.json").read_text())} \
        if (ROOT / "purchases/candidates.json").exists() else {}
    decisions = json.loads((ROOT / "purchases/decisions.json").read_text()) \
        if (ROOT / "purchases/decisions.json").exists() else {}
    out = []
    for item_id, d in decisions.items():
        c = cands.get(item_id)
        if not c or not c.get("date") or d == "skip":
            continue
        out.append((date.fromisoformat(c["date"]), "bought" if d == "purchased" else "returned"))
    return out


def main():
    today = date.today()
    visits, first_day = chrome_visits() or ([], None)
    bought = purchases()

    print(f"Periods: baseline < {TRAINING_START} · training {TRAINING_START} to {TEST_START - timedelta(1)} "
          f"· test from {TEST_START}\n")

    # Browsing: average shopping visits per day, and share of days with any shopping visit.
    if visits:
        per_day = Counter(visits)
        print(f"BROWSING (Chrome history from {first_day})")
        print(f"  {'period':<10}{'days':>6}{'visits/day':>12}{'days with shopping':>21}")
        for name in ("baseline", "training", "test"):
            days = [first_day + timedelta(i) for i in range((today - first_day).days + 1)
                    if period(first_day + timedelta(i)) == name]
            if not days:
                print(f"  {name:<10}{'–':>6}")
                continue
            total = sum(per_day[d] for d in days)
            active = sum(1 for d in days if per_day[d])
            print(f"  {name:<10}{len(days):>6}{total / len(days):>12.1f}{active / len(days):>20.0%}")
        print()

    # Buying: items per month in each period.
    if bought:
        print("BUYING (items confirmed in purchases.py)")
        print(f"  {'period':<10}{'months':>8}{'bought/month':>14}{'returned/month':>16}")
        start = min(d for d, _ in bought)
        for name, lo, hi in (("baseline", start, TRAINING_START), ("training", TRAINING_START, TEST_START),
                             ("test", TEST_START, today + timedelta(1))):
            months = max((hi - lo).days / 30.44, 1 / 30.44)
            b = sum(1 for d, k in bought if lo <= d < hi and k == "bought")
            r = sum(1 for d, k in bought if lo <= d < hi and k == "returned")
            print(f"  {name:<10}{months:>8.1f}{b / months:>14.1f}{r / months:>16.1f}")
        print("\n  Note: the test period is short at first, so its monthly rate swings a lot. "
              "Compare after a few weeks.\n")
    else:
        print("No confirmed purchases yet. Run python purchases.py first.\n")

    # Chart: weekly shopping visits, with the periods shaded.
    if visits:
        weeks = Counter(d - timedelta(d.weekday()) for d in visits)
        xs = sorted(weeks)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(xs, [weeks[w] for w in xs], width=5, color="#b89393")
        ax.axvspan(TRAINING_START, TEST_START, color="#dcd5d5", alpha=0.6, label="training week")
        ax.axvspan(TEST_START, today + timedelta(1), color="#c0a5a5", alpha=0.3, label="test period")
        ax.set_ylabel("Shopping-site visits per week")
        ax.set_title("Shopping-site visits per week (Chrome)", loc="left")
        ax.legend(frameon=False, loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(ROOT / "baseline.png", dpi=130)
        print("Saved baseline.png")


if __name__ == "__main__":
    main()
