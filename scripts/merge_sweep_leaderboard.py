"""Merge banked full-run prediction dirs into ONE leaderboard, no re-runs.

    .venv/bin/python scripts/merge_sweep_leaderboard.py \
        full-benchmark-v1 sweep-full-v1 [more-run-dirs...]

Reads results/<run>/predictions/<arm>_<split>.jsonl from each run dir, aligns
every prediction by case_id against the canonical dataset cases, tunes each
arm's threshold on val under the same 5% false-sufficient budget, reports test
once, and bootstraps paired Δ-AUROC against the incumbent champion (clef-27b)
with the same seed/reps as full-benchmark-v1 so every CI is comparable.

Emits results/merged-leaderboard-<out>/:
    leaderboard.md / leaderboard.csv / merged-stats.json
"""
from __future__ import annotations

import csv
import json
import sys
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sufficiency_lab.config import load_config  # noqa: E402
from sufficiency_lab.datasets import load_sufficiencybench  # noqa: E402
from sufficiency_lab.metrics import (discriminate, grouped_bootstrap_ci,  # noqa: E402
                                      operating_point)
from sufficiency_lab.thresholds import select_threshold  # noqa: E402

RUN_DIRS = sys.argv[1:2] or ["full-benchmark-v1"]
EXTRA_RUNS = sys.argv[2:] or ["sweep-full-v1"]
BASE = RUN_DIRS[0]
OUT_NAME = "merged-leaderboard-v1"
SEED, REPS, MAX_RISK = 13, 2000, 0.05
CHAMPION = "clef-27b"

# honesty notes stamped into the table — how each arm was actually scored
NOTES = {
    "von-1.3": "noul axis (choice mode dead on rubric)",
    "tev1-4b": "k=12 vote-rate score (no server logprobs)",
    "openjev-27b-4bit": "CC-BY-NC benchmark-only",
    "clef-27b": "incumbent champion",
    "qwen38-judge": "FPR breach at full n — disqualified standalone",
}

cfg = load_config(ROOT / "configs" / "full-benchmark.yaml")
cases = {s: load_sufficiencybench(cfg.data_path / "sufficiencybench", split=s)
         for s in ("val", "test")}


def read_jsonl(path: pathlib.Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_scores(run: str, arm: str, split: str) -> dict[str, float]:
    path = ROOT / "results" / run / "predictions" / f"{arm}_{split}.jsonl"
    if not path.exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        p = r.get("p_sufficient")
        if p is not None and p == p:  # skip NaN
            out[r["case_id"]] = float(p)
    return out


def discover_arms(run: str) -> list[str]:
    pdir = ROOT / "results" / run / "predictions"
    arms = sorted({p.name.rsplit("_", 1)[0] for p in pdir.glob("*_val.jsonl")})
    return arms


runs = [(BASE, discover_arms(BASE))] + [(r, discover_arms(r)) for r in EXTRA_RUNS]
arms = [(run, arm) for run, alist in runs for arm in alist]

# score tables aligned to canonical case order (NaN when an arm lacks a case;
# metrics filter NaN consistently so partial arms compare on their own subset)
def aligned(run: str, arm: str, split: str) -> list[float]:
    smap = load_scores(run, arm, split)
    return [smap.get(c.id, float("nan")) for c in cases[split]]

# champion test scores for paired deltas (aligned to same case order)
champ = {s: aligned(BASE, CHAMPION, s) for s in ("val", "test")}
if not any(np.isfinite(x) for x in champ["test"]):
    champ = {s: [float("nan")] * len(cases[s]) for s in ("val", "test")}

rows = []
for run, arm in arms:
    p_val = aligned(run, arm, "val")
    p_test = aligned(run, arm, "test")
    if not any(np.isfinite(x) for x in p_val):
        continue
    thr = select_threshold(cases["val"], p_val, strategy=arm, max_risk=MAX_RISK)
    d_val = discriminate(cases["val"], p_val)
    d_test = discriminate(cases["test"], p_test)
    op = operating_point(cases["test"], p_test, thr.threshold)
    ci = grouped_bootstrap_ci(cases["test"], p_test, metric="auroc",
                              reps=REPS, seed=SEED)
    champ_ok = [b for b in champ["test"] if np.isfinite(b)]
    if len(champ_ok) == len(cases["test"]) and arm != CHAMPION:
        delta = grouped_bootstrap_ci(cases["test"], p_test, champ["test"],
                                     metric="auroc", reps=REPS, seed=SEED)
        d_lo, d_hi, d_point = delta["lo"], delta["hi"], delta["point"]
    else:
        d_lo = d_hi = d_point = float("nan")
    lats = [r.get("latency_ms") for r in read_jsonl(
        ROOT / "results" / run / "predictions" / f"{arm}_test.jsonl")
        if r.get("latency_ms")]
    rows.append({
        "run": run, "arm": arm, "thr": thr.threshold,
        "n_test": d_test["n"],
        "auroc": d_test["auroc"], "auroc_lo": ci["lo"], "auroc_hi": ci["hi"],
        "auprc": d_test["auprc"], "ece": d_test["ece"],
        "fpr_test": op.risk_fpr, "coverage_test": op.coverage,
        "auroc_val": d_val["auroc"], "thr_fpr_val": thr.risk_fpr,
        "delta_vs_champ": d_point, "delta_lo": d_lo, "delta_hi": d_hi,
        "p50_ms": float(np.percentile(lats, 50)) if lats else float("nan"),
        "note": NOTES.get(arm, ""),
    })

rows.sort(key=lambda r: -(r["auroc"] if np.isfinite(r["auroc"]) else -1))

outdir = ROOT / "results" / OUT_NAME
outdir.mkdir(exist_ok=True)
(outdir / "merged-stats.json").write_text(json.dumps({"rows": rows, "seed": SEED,
    "bootstrap": REPS, "max_risk": MAX_RISK, "champion": CHAMPION}, indent=1))
with (outdir / "leaderboard.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

fmt = lambda x, nd=3: ("—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}")
lines = [
    f"# Merged leaderboard — {OUT_NAME}",
    "",
    f"Thresholds tuned on **val** (≤5% FPR), test reported once; bootstrap seed "
    f"{SEED}, {REPS} reps, grouped by question. Arms from "
    + ", ".join(r for r, _ in runs) + ".",
    "",
    "| # | arm | run | AUROC [95% CI] | Δ vs clef-27b [CI] | thr | test FPR | coverage | ECE | p50 ms | notes |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for i, r in enumerate(rows, 1):
    lines.append(
        f"| {i} | **{r['arm']}** | {r['run']} | "
        f"{fmt(r['auroc'])} [{fmt(r['auroc_lo'])}, {fmt(r['auroc_hi'])}] | "
        f"{fmt(r['delta_vs_champ'], 3)} [{fmt(r['delta_lo'])}, {fmt(r['delta_hi'])}] | "
        f"{fmt(r['thr'], 2)} | {fmt(r['fpr_test'])} | {fmt(r['coverage_test'])} | "
        f"{fmt(r['ece'])} | {fmt(r['p50_ms'], 1)} | {r['note']} |")
(outdir / "leaderboard.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
