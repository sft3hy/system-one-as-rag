"""All-field leadership figures for WEEKLY-REPORT — every model we tested.

    .venv/bin/python scripts/make_exec_merged_figures.py [merged-dir]

Reads results/<merged>/merged-stats.json + the runs' prediction files. Produces:
  exec_field_coverage.png     all arms' safely-auto-answerable traffic, honest
                              disqualification marks on budget-breakers
  exec_field_quality_speed.png  every model on quality x delay (log axis)
  exec_calibration_field.png    how four representative gates spread their
                              confidence (the champion, the hot-tail loser,
                              the spike-y judge, the cheap intern)
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

plt.rcParams.update({"figure.facecolor": "white", "font.size": 12, "axes.grid": False})

stats = json.loads((MERGED / "merged-stats.json").read_text())
rows = {r["arm"]: r for r in stats["rows"]}

NICE = {
    "clef-27b":        "Clef-27B  👑 RECOMMENDED",
    "clef-flash":      "Clef-Flash",
    "qwen38-judge":    "Chat judge",
    "cascade":         "Two-step combo",
    "lexical":         "No-AI checklist",
    "openjev-27b-4bit":     "OpenJev-27B (rival, research-only)",
    "openjev-flash-9b":     "OpenJev-Flash-9B ★ fast-tier upgrade",
    "tev1-4b":                "Tev (chat, 12-vote)",
    "laya-typed-decisions": "Laya-typed",
    "laya":                   "Laya",
    "von-1.3":                "Von",
}
GREEN, RED, GRAY = "#2e8b57", "#c62828", "#9e9e9e"
# merged-stats.json doesn't carry kind; map it here (drives marker/color rules)
KIND = {"lexical": "heuristic", "tev1-4b": "ar_decision",
        "laya": "encoder", "laya-typed-decisions": "encoder", "von-1.3": "encoder",
        "qwen38-judge": "llm_judge", "cascade": "cascade"}
for r in stats["rows"]:
    r["kind"] = KIND.get(r["arm"], "system1")

def certifiable(r):
    return r["fpr_test"] <= 0.05 + 1e-9 and r["coverage_test"] > 0.001

# ------------------------------------ 1. safe auto-answer share, whole field
field = sorted(rows.values(), key=lambda r: -r["coverage_test"])
fig, ax = plt.subplots(figsize=(10.5, 0.52 * len(field) + 1.7))
ys = np.arange(len(field))[::-1]
for y, r in zip(ys, field):
    ok = certifiable(r)
    cov = r["coverage_test"] * 100
    ax.barh(y, cov, color=(GREEN if ok else RED) if r["kind"] != "heuristic" else GRAY,
            height=0.62, alpha=0.92)
    tag = f"{cov:.0f}% of questions"
    if not ok:
        tag += "   ✗ can't keep 5% promise" if r["fpr_test"] > 0.05 else "   ✗ answers nothing"
    ax.annotate(tag, (cov + 0.5, y), va="center", fontsize=12, fontweight="bold",
                color=(GREEN if ok else RED))
    ax.annotate(f"AUROC {r['auroc']:.2f}", (cov + 0.5, y - 0.36), va="center",
                fontsize=9, color="0.35")
ax.set_yticks(ys)
ax.set_yticklabels([NICE[r["arm"]] for r in field], fontsize=11)
ax.set_xlim(0, max(r["coverage_test"] for r in field) * 100 * 1.55)
ax.set_xlabel("Share of user questions answered directly with ≤5% risk kept — GREEN = promise kept, RED = promise broken")
ax.set_title("The whole field (11 models, same 2,224-question exam): the three models that cover most traffic\n"
             "are precisely the three that break our 5% safety budget. The crown — green AND high — still fits.",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT / "exec_field_coverage.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------ 2. quality × speed, whole field
fig, ax = plt.subplots(figsize=(11.6, 6.4))
for r in rows.values():
    ok = certifiable(r)
    x = max(r["p50_ms"], 0.5)
    big = r["arm"] == "clef-27b"
    star = r["arm"] == "openjev-flash-9b"
    col = GRAY if r["kind"] == "heuristic" else (GREEN if ok else RED)
    ax.scatter(x, r["auroc"], s=440 if big else (320 if star else 170),
               color=col, marker=("*" if star else ("s" if r["kind"] == "heuristic" else "o")),
               zorder=(6 if big else (5 if star else 4)),
               edgecolor="black", linewidth=1.6 if (big or star) else 0.8, alpha=0.92)

# labels placed in DATA coordinates in the empty quadrants, arrows to points
LBL = {
    "openjev-27b-4bit":      dict(at=(1700, 0.893), text="OpenJev-27B ✗\n(research-only)", fs=11.5, fw="bold", ha="center"),
    "clef-27b":              dict(at=(42, 0.884),   text="👑 Clef-27B\nRECOMMENDED", fs=12.5, fw="bold", ha="right"),
    "openjev-flash-9b":      dict(at=(1100, 0.856), text="★ OpenJev-Flash-9B\nfast-tier upgrade", fs=11.5, fw="bold", ha="left"),
    "clef-flash":            dict(at=(40, 0.838),    text="Clef-Flash", fs=10.5, fw="normal", ha="left"),
    "cascade":               dict(at=(40, 0.812),     text="Two-step combo", fs=10.5, fw="normal", ha="left"),
    "qwen38-judge":          dict(at=(850, 0.800),    text="Chat judge ✗", fs=10.5, fw="normal", ha="left"),
    "tev1-4b":               dict(at=(1900, 0.826),    text="Tev (chat, 12-vote) ✗", fs=10.5, fw="normal", ha="left"),
    "laya-typed-decisions":  dict(at=(190, 0.694),     text="Laya-typed", fs=10, fw="normal", ha="left"),
    "laya":                  dict(at=(170, 0.620),     text="Laya", fs=10, fw="normal", ha="left"),
    "von-1.3":               dict(at=(260, 0.645),     text="Von ✗ answers nothing", fs=10, fw="normal", ha="left"),
    "lexical":               dict(at=(0.55, 0.672),    text="No-AI checklist", fs=10, fw="normal", ha="left"),
}
for arm, pl in LBL.items():
    r = rows[arm]
    ax.annotate(pl["text"], xy=(max(r["p50_ms"], 0.5), r["auroc"]), xycoords="data",
                xytext=pl["at"], textcoords="data", fontsize=pl["fs"], fontweight=pl["fw"],
                zorder=7, va="center", ha=pl["ha"],
                arrowprops=dict(arrowstyle="-", lw=0.9, color="0.45",
                               shrinkA=0, shrinkB=10, connectionstyle="arc3,rad=0.12"))
ax.set_xscale("log")
ax.set_xlim(0.4, 4200)
ax.set_xlabel("Typical delay per question (milliseconds — note: each step right is 10× slower)")
ax.set_ylabel("Judgement quality (AUROC — higher is better)")
ax.axvspan(30, 200, color=GREEN, alpha=0.06)
ax.grid(alpha=0.25)
ax.set_ylim(0.60, 0.905)
ax.set_title("Every model we tested, plotted honestly: green = keeps its 5% safety promise, red = doesn't.\n"
             "OpenJev-27B judges best of all but is red — disqualified. Clef-27B (👑) stays the only green-and-high.\n"
             "The green star — OpenJev-Flash-9B — beats Clef-Flash decisively and keeps its promise.",
             fontsize=12.5)
fig.tight_layout()
fig.savefig(OUT / "exec_field_quality_speed.png", dpi=160, bbox_inches="tight")
plt.close(fig)

# ------------------------------------ 3. calibration panel — why red is red
def scores(run, arm):
    suff, ins = [], []
    for line in (ROOT / "results" / run / "predictions" / f"{arm}_test.jsonl").open():
        rec = json.loads(line)
        p = rec.get("p_sufficient")
        if p is None or p != p or rec.get("error"):
            continue
        (suff if rec["case_id"].endswith("_suf") else ins).append(p)
    return np.array(suff), np.array(ins)

panel = [("clef-27b", "full-benchmark-v1", "👑 Clef-27B — the champion: scores spread,\ntrust matches behaviour"),
         ("openjev-27b-4bit", "sweep-full-v1", "OpenJev-27B — judges better (0.889), but\ninsufficient cases pile up at the top → 7.3% risk"),
         ("qwen38-judge", "full-benchmark-v1", "Chat judge — fake confidence in 4 spikes\n(0.1 / 0.95 / 1.0), breaks budget too"),
         ("laya-typed-decisions", "sweep-full-v1", "Laya-typed — cheap intern: low energy,\nsafe but tiny coverage")]
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), sharey=True)
hist_bins = np.linspace(0, 1, 26)
for ax, (arm, run, title) in zip(axes, panel):
    s, i = scores(run, arm)
    ax.hist(i, bins=hist_bins, color=RED, alpha=0.75, label="truly UNanswerable")
    ax.hist(s, bins=hist_bins, color="#1f77b4", alpha=0.6, label="truly answerable")
    ax.axvline(rows[arm]["thr"], color="black", ls="--", lw=1.4)
    ax.annotate(f"approved →\n{rows[arm]['coverage_test']*100:.0f}% of traffic",
                (min(rows[arm]["thr"] + 0.02, 0.62), ax.get_ylim()[1] * 0.86),
                fontsize=8.6, color="0.25")
    ax.set_title(title, fontsize=10.5)
axes[1].set_xlabel("gate's confidence that the question is answerable")
axes[0].legend(fontsize=9, loc="upper left")
axes[0].set_ylabel("number of test questions")
fig.suptitle("Why two of the three best judges sit in red: the red masses that leak past the "
             "dashed bar are the 5%-budget breakers", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "exec_calibration_field.png", dpi=160, bbox_inches="tight")
plt.close(fig)

print("wrote exec_field_coverage.png, exec_field_quality_speed.png, exec_calibration_field.png")
