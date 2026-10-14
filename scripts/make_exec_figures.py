"""Leadership-friendly figures for WEEKLY-REPORT — simplified, annotated versions
of the technical charts. Usage: make_exec_figures.py [run-dir-name]
(default: cascade-flash-judge-v1). All numbers read from the run's report.json."""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "cascade-flash-judge-v1"
RUN = ROOT / "results" / RUN_NAME
OUT = ROOT / "results" / "report-figures"
OUT.mkdir(exist_ok=True)
report = json.loads((RUN / "reports" / "report.json").read_text())
rows = {r["strategy"]: r for r in report["leaderboard"]}

plt.rcParams.update({"figure.facecolor": "white", "font.size": 12, "axes.grid": False})

arms = ["clef-flash", "cascade", "clef-27b", "qwen38-judge"]  # heuristics excluded: no latency, floor only
nice = {
    "clef-flash": "Clef-Flash\n(cheap, fast)",
    "cascade": "Two-step combo\n(flash, hard cases to judge)",
    "clef-27b": "Clef-27B\n(strong)\n← recommended",
    "qwen38-judge": "Chat-model judge\n(coverage looks great,\nprobabilities don't)",
}
colors = {"clef-flash": "#1f77b4", "cascade": "#9467bd", "clef-27b": "#d62728", "qwen38-judge": "#9cbd5a"}
cov = [rows[a]["coverage"] * 100 for a in arms]

# ---------------------------------------------- 1. "How much can we safely auto-answer"
fig, ax = plt.subplots(figsize=(9.5, 4.6))
bars = ax.barh(range(len(arms)), cov, color=[colors[a] for a in arms], height=0.6)
for b, v in zip(bars, cov):
    ax.annotate(f"{v:.0f}% of questions", (v + 0.4, b.get_y() + b.get_height() / 2),
                va="center", fontsize=13, fontweight="bold")
ax.set_yticks(range(len(arms)))
ax.set_yticklabels([nice[a] for a in arms], fontsize=11)
ax.invert_yaxis()
ax.set_xlim(0, max(cov) * 1.45)
ax.set_xlabel("Share of user questions the gate lets the AI answer directly — with ≤5% wrong-answers risk")
lo_cov, hi_cov = min(cov), max(cov)
lo_blk = min(rows[a]["false_insufficient_rate"] for a in arms) * 100
hi_blk = max(rows[a]["false_insufficient_rate"] for a in arms) * 100
ax.set_title(f"Every option can only auto-answer {lo_cov:.0f}–{hi_cov:.0f}% of traffic today — and still blocks "
             f"{lo_blk:.0f}–{hi_blk:.0f}% of\nquestions it COULD answer correctly. Not being confident enough is "
             f"the bottleneck, not being wrong.", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "exec_safe_autoanswer.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------- 2. quality vs cost trade-off scatter
labels = {
    "clef-flash": "Clef-Flash\nfast + cheap",
    "cascade": "Two-step combo",
    "qwen38-judge": "Chat judge\ngood judgement,\nbad confidence",
    "clef-27b": "Clef-27B\nRECOMMENDED",
}
fig, ax = plt.subplots(figsize=(9.0, 5.2))
for arm in arms:
    x, y = rows[arm]["p50_ms"], rows[arm]["auroc"]
    big = arm == "clef-27b"
    ax.scatter(x, y, s=380 if big else 240, color=colors[arm], zorder=3,
               edgecolor="black", linewidth=1.4 if big else 0.8)
    ax.annotate(labels[arm], (x, y), textcoords="offset points", xytext=(14, 10), fontsize=11.5,
                fontweight="bold" if big else "normal")
ax.set_xlabel("Typical delay per question (milliseconds — lower is better)")
ax.set_ylabel("Judgement quality (AUROC — higher is better)")
ax.set_xlim(0, max(rows[a]["p50_ms"] for a in arms) * 1.4)
lo_y = min(rows[a]["auroc"] for a in arms)
ax.set_ylim(lo_y - 0.02, max(rows[a]["auroc"] for a in arms) + 0.015)
ax.grid(alpha=0.25)
blk27 = rows["clef-27b"]["false_insufficient_rate"] * 100
blkf = rows["clef-flash"]["false_insufficient_rate"] * 100
ax.set_title("The trade-off: Clef-27B is clearly the most accurate gate; Clef-Flash is clearly the fastest.\n"
             f"At our 5% risk budget the senior still misses {blk27:.0f}% of answerable questions vs the junior's "
             f"{blkf:.0f}% —\nclosing that coverage gap is what we're working on.", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "exec_quality_vs_speed.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"wrote exec_safe_autoanswer.png, exec_quality_vs_speed.png (from {RUN_NAME})")
