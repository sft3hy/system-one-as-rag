"""Certify score-ENSEMBLES from banked predictions — zero GPU, zero re-runs.

    .venv/bin/python scripts/ensemble_certify.py

The motivation (exec_calibration_field.png): OpenJev-27B discriminates best
(AUROC 0.889) but is disqualified by a thin red tail of insufficient cases at
0.95-1.0; the chat judge leaks via four spikes; Clef-27B is clean but slower
and lower-AUROC. Two models blowing up on the SAME cases is rarer than one —
so combine their scores and re-tune the single threshold on val under the
same 5% false-sufficient budget, report test once.

Pool (continuous-score arms, complete on both splits):
    clef-27b, clef-flash, openjev-27b-4bit, openjev-flash-9b,
    qwen38-judge, tev1-4b
Rules:
    rank_avg — mean percentile rank (robust to qwen's spike quantisation)
    prob_avg — mean probability
    prob_min — conservative AND: approve only if the least confident agrees
    prob_max — optimistic OR: any vote of confidence approves
Combinations: all pairs, plus hand-picked triples/quads around the two 27B
brains. Protocol identical to merge_sweep_leaderboard.py: thresholds on val
only (max_risk 0.05), test once, bootstrap seed 13 / 2000 reps / grouped.

Emits results/ensemble-v1/{ensemble-stats.json,ensemble-leaderboard.md}.
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sufficiency_lab.config import load_config  # noqa: E402
from sufficiency_lab.datasets import load_sufficiencybench  # noqa: E402
from sufficiency_lab.metrics import (discriminate, grouped_bootstrap_ci,  # noqa: E402
                                      operating_point)
from sufficiency_lab.thresholds import select_threshold  # noqa: E402

SEED, REPS, MAX_RISK = 13, 2000, 0.05
CHAMPION = "clef-27b"
RUNS = {"full-benchmark-v1": ["clef-27b", "clef-flash", "qwen38-judge"],
        "sweep-full-v1": ["openjev-27b-4bit", "openjev-flash-9b", "tev1-4b"]}
POOL = [a for arms in RUNS.values() for a in arms]

cfg = load_config(ROOT / "configs" / "full-benchmark.yaml")
cases = {s: load_sufficiencybench(cfg.data_path / "sufficiencybench", split=s)
         for s in ("val", "test")}


def read_jsonl(path: pathlib.Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def aligned(run: str, arm: str, split: str) -> np.ndarray:
    smap = {r["case_id"]: float(r["p_sufficient"])
            for r in read_jsonl(ROOT / "results" / run / "predictions"
                                / f"{arm}_{split}.jsonl")
            if r.get("p_sufficient") is not None and r["p_sufficient"] == r["p_sufficient"]}
    out = np.array([smap.get(c.id, float("nan")) for c in cases[split]])
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"{arm}/{split} incomplete — refusing to ensemble partial arms")
    return out


# scores[arm][split] -> aligned vector
scores = {arm: {s: aligned(run, arm, s) for s in ("val", "test")}
          for run, arms in RUNS.items() for arm in arms}
lat_p50 = {arm: 0.0 for arm in POOL}
for run, arms in RUNS.items():
    for arm in arms:
        lats = [r.get("latency_ms") for r in read_jsonl(
            ROOT / "results" / run / "predictions" / f"{arm}_test.jsonl")
            if r.get("latency_ms")]
        lat_p50[arm] = float(np.percentile(lats, 50)) if lats else float("nan")


def pct_rank(x: np.ndarray) -> np.ndarray:
    """Average-tie percentile rank within the split (label-free)."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    s = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks / (len(x) - 1)


RULES = {
    "rank_avg": lambda ms: np.mean([pct_rank(m) for m in ms], axis=0),
    "prob_avg": lambda ms: np.mean(ms, axis=0),
    "prob_min": lambda ms: np.min(ms, axis=0),
    "prob_max": lambda ms: np.max(ms, axis=0),
}

combos = [tuple(c) for c in itertools.combinations(POOL, 2)]
hand = [
    ("clef-27b", "openjev-27b-4bit", "openjev-flash-9b"),
    ("clef-27b", "openjev-27b-4bit", "clef-flash"),
    ("clef-27b", "openjev-27b-4bit", "qwen38-judge"),
    ("clef-27b", "openjev-27b-4bit", "tev1-4b"),
    ("clef-27b", "openjev-27b-4bit", "openjev-flash-9b", "clef-flash"),
    ("clef-27b", "openjev-27b-4bit", "openjev-flash-9b", "qwen38-judge"),
]
combos += [h for h in hand if h not in combos]

champ_test = scores[CHAMPION]["test"]
rows = []
for members in combos:
    for rule, fn in RULES.items():
        name = f"{'+'.join(members)}|{rule}"
        v = fn([scores[m]["val"] for m in members])
        t = fn([scores[m]["test"] for m in members])
        thr = select_threshold(cases["val"], list(v), strategy=name, max_risk=MAX_RISK)
        op_val = operating_point(cases["val"], v, thr.threshold)
        op = operating_point(cases["test"], t, thr.threshold)
        d = discriminate(cases["test"], t)
        ci = grouped_bootstrap_ci(cases["test"], t, metric="auroc", reps=REPS, seed=SEED)
        delta = grouped_bootstrap_ci(cases["test"], t, champ_test,
                                     metric="auroc", reps=REPS, seed=SEED)
        rows.append({
            "name": name, "members": list(members), "rule": rule,
            "thr": thr.threshold, "thr_fpr_val": op_val.risk_fpr,
            "coverage_val": op_val.coverage,
            "auroc": d["auroc"], "auroc_lo": ci["lo"], "auroc_hi": ci["hi"],
            "auprc": d["auprc"], "ece": d["ece"],
            "fpr_test": op.risk_fpr, "coverage_test": op.coverage,
            "delta_vs_champ": delta["point"], "delta_lo": delta["lo"], "delta_hi": delta["hi"],
            "p50_ms_max": max(lat_p50[m] for m in members),
            "certified": bool(op.risk_fpr <= 0.05 + 1e-9 and op.coverage > 0.001),
        })

rows.sort(key=lambda r: (-int(r["certified"]), -r["coverage_test"]))

outdir = ROOT / "results" / "ensemble-v1"
outdir.mkdir(exist_ok=True)
(outdir / "ensemble-stats.json").write_text(json.dumps(
    {"rows": rows, "seed": SEED, "bootstrap": REPS, "max_risk": MAX_RISK,
     "champion_auroc": discriminate(cases["test"], champ_test)["auroc"],
     "champion_coverage": operating_point(cases["test"], champ_test,
        select_threshold(cases["val"], list(scores[CHAMPION]["val"]),
                         strategy=CHAMPION, max_risk=MAX_RISK).threshold).coverage},
    indent=1))

fmt = lambda x, nd=3: ("—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}")
lines = [
    "# Ensemble certification — banked scores, no re-runs",
    "",
    f"{len(rows)} combination scorers from {len(POOL)} banked arms. Thresholds tuned",
    f"on val only (≤{MAX_RISK:.0%} FPR), test reported once; bootstrap seed {SEED},",
    f"{REPS} reps grouped by question. CERTIFIED = test FPR ≤ 5% with coverage > 0.1%.",
    "",
    "| # | members | rule | thr | val FPR | AUROC [CI] | Δ vs champ [CI] | test FPR | coverage | p50 ms (max) | cert |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for i, r in enumerate(rows, 1):
    lines.append(
        f"| {i} | {' + '.join(r['members'])} | {r['rule']} | {fmt(r['thr'], 2)} | "
        f"{fmt(r['thr_fpr_val'])} | {fmt(r['auroc'])} [{fmt(r['auroc_lo'])}, {fmt(r['auroc_hi'])}] | "
        f"{fmt(r['delta_vs_champ'])} [{fmt(r['delta_lo'])}, {fmt(r['delta_hi'])}] | "
        f"{fmt(r['fpr_test'])} | {fmt(r['coverage_test'])} | {fmt(r['p50_ms_max'], 0)} | "
        f"{'✅' if r['certified'] else '❌'} |")

# tail-overlap diagnostic: who leaks together? (descriptive, test split, post-hoc)
insuf = np.array([0 if c.sufficient else 1 for c in cases["test"]])
thr_of = {r["name"]: r["thr"] for r in rows}
champ_thr = select_threshold(cases["val"], list(scores[CHAMPION]["val"]),
                             strategy=CHAMPION, max_risk=MAX_RISK).threshold
oj_thr = select_threshold(cases["val"], list(scores["openjev-27b-4bit"]["val"]),
                          strategy="openjev-27b-4bit", max_risk=MAX_RISK).threshold
leak_c = (insuf == 1) & (champ_test >= champ_thr)
leak_o = (insuf == 1) & (scores["openjev-27b-4bit"]["test"] >= oj_thr)
both = leak_c & leak_o
lines += [
    "",
    "## Leak-set overlap (the ensemble's premise, test split, descriptive)",
    "",
    f"- insufficient-case leaks: clef-27b {int(leak_c.sum())}, openjev-27b {int(leak_o.sum())}, "
    f"**shared {int(both.sum())}** "
    f"(Jaccard {both.sum() / max((leak_c | leak_o).sum(), 1):.2f})",
    f"- an oracle gate union-free of both leaks could cover "
    f"{(np.array([1 if c.sufficient else 0 for c in cases['test']]) & ~(leak_c | leak_o)).sum() / len(cases['test']):.1%} "
    f"of traffic at 0% FPR — that's the headroom ensembles chase.",
]
(outdir / "ensemble-leaderboard.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
