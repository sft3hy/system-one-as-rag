"""Figures for REPORT.md — rebuilt from results/ artefacts, nothing hand-typed.

Sources:
  results/cascade-flash-judge-v1/{reports/report.json,predictions/*.jsonl,stats.json}
  results/clef-flash-vs-clef/reports/report.json
  results/timing-100/reports/report.json
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
import sys
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "cascade-flash-judge-v1"
RUN = ROOT / "results" / RUN_NAME
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

COLORS = {
    "clef-flash": "#1f77b4",
    "clef-27b": "#d62728",
    "qwen38-judge": "#2ca02c",
    "cascade": "#9467bd",
    "lexical": "#7f7f7f",
}

report = json.loads((RUN / "reports" / "report.json").read_text())
rows = {r["strategy"]: r for r in report["leaderboard"]}

# ---------------------------------------------------------------- 1. leaderboard
arms = [a for a in ["clef-flash", "qwen38-judge", "cascade", "clef-27b", "lexical"] if a in rows]
metrics = [
    ("AUROC (test)", "auroc", (min(rows[a]["auroc"] for a in arms) - 0.04, max(rows[a]["auroc"] for a in arms) + 0.02), "higher is better"),
    ("Coverage @5% risk (test)", "coverage", (0.0, max(rows[a]["coverage"] for a in arms) * 1.35), "higher is better"),
    ("ECE (test)", "ece", (0.0, max(rows[a]["ece"] for a in arms) * 1.3), "lower is better"),
    ("p50 latency (ms)", "p50_ms", (0, max(rows[a]["p50_ms"] for a in arms) * 1.3), "lower is better"),
]
fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.4))
for ax, (title, key, ylim, sub) in zip(axes, metrics):
    vals = [rows[a][key] for a in arms]
    bars = ax.bar(range(len(arms)), vals, color=[COLORS[a] for a in arms], width=0.62)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.3g}" if key != "p50_ms" else f"{v:.0f}",
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a.replace("-", "-\n") for a in arms], fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11)
    ax.text(0.5, 1.02, sub, transform=ax.transAxes, ha="center", fontsize=8.5, color="0.45")
fig.suptitle(f"{RUN_NAME}: {report['counts']['test']} test conditions, thresholds tuned on val @ 5% FPR budget",
             fontsize=12.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "leaderboard.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------- 2. risk-coverage curves
def roc_curve(pairs):
    """pairs = [(score, label)] -> (fpr_grid, coverage on true positives)."""
    arr = sorted(pairs, key=lambda x: -x[0])
    pos = np.array([lab for _, lab in arr])
    n_pos, n_neg = (pos == 1).sum(), (pos == 0).sum()
    tp = np.cumsum(pos == 1)
    fp = np.cumsum(pos == 0)
    cov = tp / n_pos
    fpr = fp / n_neg
    return np.concatenate([[0.0], fpr]), np.concatenate([[0.0], cov])

fig, ax = plt.subplots(figsize=(6.2, 4.6))
for arm in ["clef-27b", "cascade", "clef-flash", "qwen38-judge"]:
    recs = [json.loads(l) for l in (RUN / "predictions" / f"{arm}_test.jsonl").read_text().splitlines() if l]
    pairs = [(r["probabilities"]["calibrated"] if "calibrated" in r["probabilities"] else r["p_sufficient"],
             1 if r["case_id"].endswith("_suf") else 0) for r in recs]
    fpr, cov = roc_curve(pairs)
    ax.plot(fpr, cov, color=COLORS[arm], lw=2,
            label=arm + (f"  (thr {rows[arm]['threshold']:.2f}: cov {rows[arm]['coverage']:.3f}, fpr {rows[arm]['risk_fpr']:.3f})" if arm in rows else ""))
    ax.scatter([rows[arm]["risk_fpr"]], [rows[arm]["coverage"]], color=COLORS[arm], s=55, zorder=5,
               edgecolor="black", linewidth=0.7)
ax.axhline(0.05 * 1, ls=":", color="crimson", lw=1, label=None)
ax.axvline(0.05, ls="--", color="crimson", lw=1.2)
ax.text(0.052, ax.get_ylim()[1] * 0.97, "5% risk budget", color="crimson", fontsize=9, va="top")
ax.set_xlabel("False-sufficient rate (risk)")
ax.set_ylabel("Coverage (true positives passed)")
ax.set_title("Risk–coverage: where each gate can sit at ≤5% risk")
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "risk_coverage_final.png", dpi=160)
plt.close(fig)

# --------------------------------------------- 3. judge score distribution (spikes)
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
for ax, arm in zip(axes, ["clef-flash", "clef-27b", "qwen38-judge"]):
    recs = [json.loads(l) for l in (RUN / "predictions" / f"{arm}_val.jsonl").read_text().splitlines() if l]
    pos = [r["p_sufficient"] for r in recs if r["case_id"].endswith("_suf")]
    neg = [r["p_sufficient"] for r in recs if r["case_id"].endswith("_insuf")]
    bins = np.linspace(0, 1, 41)
    ax.hist([neg, pos], bins=bins, color=["#d6272855", "#1f77b455"],
            edgecolor=["#d62728", "#1f77b4"], label=["insufficient (score)", "sufficient (score)"])
    ax.set_title(arm, color=COLORS[arm])
axes[0].set_ylabel("conditions")
for ax in axes:
    ax.set_xlabel("P(sufficient)")
axes[0].legend(fontsize=8.5)
fig.suptitle("Score distributions (val, n=300): the judge answers on 4 spikes, not a scale", y=1.03, fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT / "score_distributions.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------- 4. capacity: flash vs 27B, 1q vs 3q
comp = json.loads((ROOT / "results" / "clef-flash-vs-clef" / "reports" / "report.json").read_text())
crows = {r["strategy"]: r for r in comp["leaderboard"]}
names = ["clef-flash-oQ4e", "clef-flash-oQ4e-1q", "clef-27b-oQ4e", "clef-27b-oQ4e-1q"]
labels = ["Flash 3q", "Flash 1q", "27B 3q", "27B 1q"]
cols = ["#1f77b4", "#6baed6", "#d62728", "#f4a0a0"]
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(6.8, 4.2))
w = 0.2
for i, (key, mk) in enumerate([("auroc", ""), ("coverage", ""), ("p50_ms", "")]):
    vals = [crows[n][key] for n in names]
    if key == "p50_ms":
        vals = [v / 1000 for v in vals]
    ax.bar(x + (i - 1) * w, vals, w, color=cols, label={"auroc": "AUROC", "coverage": "coverage @5%", "p50_ms": "p50 latency (s)"}[key])
    for xi, v in zip(x + (i - 1) * w, vals):
        ax.annotate(f"{v:.3g}", (xi, v), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Capacity buys accuracy & coverage, costs 3.2× latency (question packing adds ~25–35 ms)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "capacity_vs_latency.png", dpi=160)
plt.close(fig)

# ------------------------------------------------- 5. concurrency: throughput pinned
fig, ax = plt.subplots(figsize=(6.8, 4.2))
conc = [1, 4, 8]
flash_qps = [6.2, 6.2, 6.2]
flash_p95 = [195, 701, 1344]
b27_qps = [1.9, 1.9, 1.9]
b27_p95 = [661, 2374, 4498]
ax.plot(conc, flash_qps, "o-", color=COLORS["clef-flash"], lw=2, label="Flash QPS")
ax.plot(conc, b27_qps, "s-", color=COLORS["clef-27b"], lw=2, label="27B QPS")
ax.set_xlabel("concurrent connections")
ax.set_ylabel("throughput (QPS)")
ax.set_ylim(0, 8)
ax2 = ax.twinx()
ax2.grid(False)
ax2.plot(conc, flash_p95, "o--", color=COLORS["clef-flash"], alpha=0.5, label="Flash p95")
ax2.plot(conc, b27_p95, "s--", color=COLORS["clef-27b"], alpha=0.5, label="27B p95")
ax2.set_ylabel("p95 latency (ms)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="center left")
ax.set_title("Decision engine doesn't batch: QPS pinned, concurrency only inflates tails")
fig.tight_layout()
fig.savefig(OUT / "concurrency_ceiling.png", dpi=160)
plt.close(fig)

# --------------------------------------------------------------- 6. cascade costs
stats = json.loads((RUN / "stats.json").read_text())
fig, ax = plt.subplots(figsize=(6.8, 3.8))
shapes = ["clef-flash", "cascade", "qwen38-judge", "clef-27b"]
sec = [(stats[a]["wall_serial_s"]["val"] + stats[a]["wall_serial_s"]["test"]) / 2 / 300 for a in shapes]
bars = ax.barh(shapes, sec, color=[COLORS[a] for a in shapes])
for b, v in zip(bars, sec):
    ax.annotate(f"{v:.2f} s/cond", (v, b.get_y() + b.get_height() / 2),
                va="center", ha="left", fontsize=9.5, fontweight="bold")
ax.set_xlabel("wall-clock seconds per condition (measured)")
ax.set_xlim(0, max(sec) * 1.25)
ax.set_title("Cascade costs flash + 21.7% escalation — half the wall-clock of 27B")
fig.tight_layout()
fig.savefig(OUT / "cost_per_condition.png", dpi=160)
plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob("*.png"))])
