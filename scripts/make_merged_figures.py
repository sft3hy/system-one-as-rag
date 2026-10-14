"""Merged sweep figures — one chart per family question, from merged-stats.json.

    .venv/bin/python scripts/make_merged_figures.py [merged-dir]

Reads results/<merged>/merged-stats.json (written by merge_sweep_leaderboard.py)
plus the underlying prediction jsonl files for the ROC panel. Nothing hand-typed.
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
MERGED = ROOT / "results" / (sys.argv[1] if len(sys.argv) > 1 else "merged-leaderboard-v1")
OUT = ROOT / "results" / "report-figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "font.size": 10.5,
    "axes.titlesize": 12,
})

# family colors: decision-native blue, AR orange, encoder green, reference grays
COLORS = {
    "clef-flash": "#1f77b4", "openjev-flash-9b": "#17becf",
    "clef-27b": "#d62728", "openjev-27b-4bit": "#9467bd",
    "tev1-4b": "#ff7f0e",
    "laya": "#2ca02c", "laya-typed-decisions": "#bafd98", "von-1.3": "#6a9955",
    "opendecider-nano": "#c8e6c9",
    "qwen38-judge": "#8c564b", "cascade": "#7f7f7f", "lexical": "#bcbcbc",
}
MARKERS = {"tev1-4b": "s", "von-1.3": "^", "qwen38-judge": "D", "cascade": "P",
           "lexical": "x"}

stats = json.loads((MERGED / "merged-stats.json").read_text())
rows = stats["rows"]
arms = [r["arm"] for r in rows]
by_arm = {r["arm"]: r for r in rows}

# ------------------------------------------------- 1. AUROC ladder with CIs
fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(arms) + 1.5))
ys = np.arange(len(arms))[::-1]
for y, r in zip(ys, rows):
    c = COLORS.get(r["arm"], "#333333")
    ax.barh(y, r["auroc"], color=c, height=0.62, alpha=0.9)
    if np.isfinite(r["auroc_lo"]):
        ax.plot([r["auroc_lo"], r["auroc_hi"]], [y, y], color="black", lw=1.4,
                solid_capstyle="butt")
    label = f"{r['auroc']:.3f}"
    if r.get("delta_lo") is not None and np.isfinite(r.get("delta_vs_champ", float("nan"))):
        d = r["delta_vs_champ"]
        sig = "*" if (r["delta_lo"] > 0 or r["delta_hi"] < 0) else ""
        label += f"  ({d:+.3f}{sig})"
    ax.annotate(label, (max(r["auroc"], r.get("auroc_hi", r["auroc"])) + 0.012, y),
                va="center", fontsize=9.5, fontweight="bold")
ax.set_yticks(ys)
ax.set_yticklabels([a + ("  ⚠NC" if "NC" in by_arm[a].get("note", "") else "")
                    for a in arms], fontsize=9.5)
ax.set_xlim(min(r["auroc_lo"] for r in rows if np.isfinite(r["auroc_lo"])) - 0.03, 1.02)
champ = by_arm.get("clef-27b")
if champ:
    ax.axvline(champ["auroc"], color="#d62728", ls="--", lw=1, alpha=0.7)
    ax.annotate("incumbent 0.871", (champ["auroc"] + 0.004, -0.85), color="#d62728",
                fontsize=8.5)
ax.set_xlabel("AUROC (test)  ·  bars = point, whiskers = 95% group bootstrap · "
              "(Δ vs clef-27b paired, * = 95% CI excludes 0)")
ax.set_title("System-one sweep merged leaderboard — full SufficiencyBench test "
             f"(n={rows[0]['n_test']} conditions per arm)", fontsize=12)
ax.invert_yaxis()
fig.savefig(OUT / "merged_leaderboard.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- 2. ROC/risk-coverage, one line per family
def roc_curve(pairs):
    arr = sorted(pairs, key=lambda x: -x[0])
    pos = np.array([lab for _, lab in arr])
    n_pos, n_neg = (pos == 1).sum(), (pos == 0).sum()
    tp = np.cumsum(pos == 1)
    fp = np.cumsum(pos == 0)
    return fp / max(n_neg, 1), tp / max(n_pos, 1)

PRED = {}
for r in rows:
    p = ROOT / "results" / r["run"] / "predictions" / f"{r['arm']}_test.jsonl"
    pairs = []
    if p.exists():
        for line in p.open():
            rec = json.loads(line)
            if rec.get("p_sufficient") is not None and rec["p_sufficient"] == rec["p_sufficient"]:
                pairs.append((rec["p_sufficient"], 1 if rec["case_id"].endswith("_suf") else 0))
    PRED[r["arm"]] = pairs

fig, ax = plt.subplots(figsize=(6.4, 5.2))
for r in rows:
    pairs = PRED[r["arm"]]
    if not pairs:
        continue
    fpr, tpr = roc_curve(pairs)
    ax.plot(fpr, tpr, color=COLORS.get(r["arm"], "#333"), lw=1.6,
            marker=MARKERS.get(r["arm"], None), markevery=max(len(fpr)//12, 1),
            markersize=4, label=f"{r['arm']} ({r['auroc']:.3f})")
    # certifiable operating point at the tuned threshold
    t = r["thr"]
    fp5 = sum(1 for s, l in pairs if l == 0 and s >= t) / max(sum(1 for _, l in pairs if l == 0), 1)
    tp5 = sum(1 for s, l in pairs if l == 1 and s >= t) / max(sum(1 for _, l in pairs if l == 1), 1)
    ax.scatter([fp5], [tp5], color=COLORS.get(r["arm"], "#333"), s=38, zorder=5,
               edgecolor="white", linewidth=0.8)
ax.plot([0, 1], [0, 1], ls=":", color="0.6", lw=1)
ax.set_xlabel("false-sufficient rate (FPR) — risk budget 0.05")
ax.set_ylabel("coverage: truly-sufficient passed (%)")
ax.axvline(0.05, color="crimson", lw=1, ls="--", alpha=0.6)
ax.annotate("5% budget", (0.052, 0.08), color="crimson", fontsize=8.5)
ax.set_title("Risk-coverage, all arms on one axis (dots = tuned thresholds)", fontsize=11.5)
ax.legend(fontsize=8.2, loc="lower right", framealpha=0.9)
fig.savefig(OUT / "merged_risk_coverage.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------- 3. separation vs latency: the deploy picture
fig, ax = plt.subplots(figsize=(6.6, 5.0))
for r in rows:
    lat = r["p50_ms"] if np.isfinite(r["p50_ms"]) else 1e3
    ax.scatter(lat, r["auroc"], color=COLORS.get(r["arm"], "#333"), s=70,
               marker=MARKERS.get(r["arm"], "o"), zorder=4,
               edgecolor="white", linewidth=0.8)
    ax.annotate(r["arm"], (lat * 1.12, r["auroc"] + 0.006), fontsize=8.6)
ax.set_xscale("log")
ax.set_xlabel("p50 latency per condition (ms, log)")
ax.set_ylabel("AUROC (test)")
ax.axvspan(100, 250, color="#2ca02c", alpha=0.07)
ax.annotate("fast tier\n(<250 ms)", (120, ax.get_ylim()[0] + 0.02), fontsize=8.5,
            color="#2ca02c")
ax.set_title("The deploy picture: accuracy vs latency, every model we bought", fontsize=11.5)
fig.savefig(OUT / "merged_accuracy_vs_latency.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("[merged figs]", *[p.name for p in sorted(OUT.glob("merged_*.png"))])
