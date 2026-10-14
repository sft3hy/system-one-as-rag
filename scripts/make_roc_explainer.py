"""ROC/AUROC explainer figure for WEEKLY-REPORT — conceptual, non-technical.

Shows the classic three curves (perfect / real / random) and shades the AREA
under one real classifier's curve, because that area literally IS the AUROC.
Annotated in product language, not statistics language.
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
import sys
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "full-benchmark-v1"
OUT = ROOT / "results" / "report-figures"

plt.rcParams.update({"figure.facecolor": "white", "font.size": 12})

fig, ax = plt.subplots(figsize=(8.6, 6.0))

import json

x = np.linspace(0, 1, 100)
random_y = x
perfect_x = np.array([0.0, 0.0, 1.0])
perfect_y = np.array([0.0, 1.0, 1.0])
# other curves are illustrative power curves; blue is the REAL measured 27B curve
medium_y = x ** 0.45
weak_y = x ** 0.8

recs = [json.loads(l) for l in
        (ROOT / "results" / RUN_NAME / "predictions" / "clef-27b_test.jsonl").read_text().splitlines() if l]
auroc = next(r["auroc"] for r in
             json.loads((ROOT / "results" / RUN_NAME / "reports" / "report.json").read_text())["leaderboard"]
             if r["strategy"] == "clef-27b")
arr = sorted(recs, key=lambda r: -r["p_sufficient"])
pos = np.array([1 if r["case_id"].endswith("_suf") else 0 for r in arr])
n_pos, n_neg = int((pos == 1).sum()), int((pos == 0).sum())
strong_x = np.concatenate([[0.0], np.cumsum(pos == 0) / n_neg])
strong_y = np.concatenate([[0.0], np.cumsum(pos == 1) / n_pos])

# the area under the "our gate" curve is the AUROC — shade it (real measured curve)
ax.fill_between(strong_x, 0, strong_y, color="#1f77b4", alpha=0.18, zorder=1)

ax.plot(perfect_x, perfect_y, color="#9467bd", lw=3.5, label="PERFECT gate  —  AUROC = 1.00")
ax.plot(strong_x, strong_y, color="#1f77b4", lw=3,
        label=f"Our best gate (Clef-27B) — real data, {len(recs):,} questions — AUROC = {auroc:.2f}")
ax.plot(x, medium_y, color="#ff7f0e", lw=2.5, label="Average gate  —  AUROC = 0.75")
ax.plot(x, weak_y, color="#2ca02c", lw=2.5, label="Poor gate  —  AUROC = 0.55")
ax.plot(random_y, x, color="crimson", lw=2.5, ls="--", label="Coin flip  —  AUROC = 0.50")

# annotation: the shaded area
ax.annotate("AUROC = the SHADED AREA under the curve.\nBigger area = better judge.\n0.50 = coin flip · 1.00 = perfect",
            xy=(0.30, 0.50), fontsize=12, fontweight="bold", color="#1f77b4", zorder=6,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#1f77b4", boxstyle="round,pad=0.4"))

# "better" arrow pointing up-left
ax.annotate("", xy=(0.30, 0.87), xytext=(0.62, 0.58),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.2))
ax.text(0.635, 0.855, "BETTER", fontsize=13, fontweight="bold", rotation=32)

# axis labels in product language
ax.set_xlabel("FALSE POSITIVE RATE (the RISK) — % of unanswerable questions wrongly waved through  ·  we cap this at 5%",
              fontsize=11, labelpad=10)
ax.set_ylabel("TRUE POSITIVE RATE (the COVERAGE) — % of answerable questions correctly approved",
              fontsize=11, labelpad=14)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.set_title("What AUROC measures: how far a gate's curve lifts away from the coin-flip line.\n"
             "A judge never needs a fixed threshold to be scored — it's rated over ALL thresholds at once.",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT / "exec_roc_explainer.png", dpi=160, bbox_inches="tight")
print("wrote exec_roc_explainer.png")
