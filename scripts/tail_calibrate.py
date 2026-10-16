"""Hubris-veto: non-monotonic tail calibration from banked predictions — zero GPU.

    .venv/bin/python scripts/tail_calibrate.py

The ensemble result (§15) proved OpenJev's 0.95-1.0 red tail is fixable by a
second opinion. This is the single-model version: if a *band* of a gate's own
scores is empirically dishonest on val (purity below the budget), veto that
band even though it shouts loudest — then lower the normal threshold to spend
the freed budget. Approval rule:

    approve iff  raw >= t   AND   purity_bin(raw) >= c

Bins are fixed by VAL quantiles (40 by default); purity = empirical share of
answerable cases per bin, estimated on VAL ONLY. (t, c) tuned on val to max
coverage subject to val FPR <= 5%; test reported once, bootstrap seed 13 /
2000 reps grouped by question — same protocol as every prior run.

Not monotone-equivalent to plain thresholds: purity() may dip at the top, so
the rule can veto high-score bands that a threshold can only approve.

Arms: every banked arm; licence flags carried through — only clef/cascade/
qwen/lexical combos are shippable if they certify.

Emits results/tail-calibration-v1/{tail-cal-stats.json,tail-cal-report.md}.
"""
from __future__ import annotations

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

SEED, REPS, MAX_RISK = 13, 2000, 0.05
CHAMPION = "clef-27b"
RUNS = {"full-benchmark-v1": ["clef-27b", "clef-flash", "qwen38-judge", "cascade",
                               "lexical"],
        "sweep-full-v1": ["openjev-27b-4bit", "openjev-flash-9b", "tev1-4b",
                          "laya-typed-decisions", "laya"]}
LICENSE = {"openjev-27b-4bit": "CC-BY-NC", "openjev-flash-9b": "CC-BY-NC (HF 2026-10-15)",
           "tev1-4b": "unverified (official card silent)"}
NBINS = 40

cfg = load_config(ROOT / "configs" / "full-benchmark.yaml")
cases = {s: load_sufficiencybench(cfg.data_path / "sufficiencybench", split=s)
         for s in ("val", "test")}
y = {s: np.array([1 if c.sufficient else 0 for c in cases[s]]) for s in ("val", "test")}


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
    return np.array([smap.get(c.id, float("nan")) for c in cases[split]])


def evaluate(name, run, arm, p_val, p_test):
    """Tune (t, c) on val only; report test once."""
    q = np.quantile(p_val, np.linspace(0, 1, NBINS + 1))
    q[0], q[-1] = -np.inf, np.inf
    b_val = np.clip(np.searchsorted(q, p_val, side="right") - 1, 0, NBINS - 1)
    purity = np.full(NBINS, np.nan)
    for b in range(NBINS):
        m = b_val == b
        if m.sum() >= 10:  # honesty: never veto a band we cannot estimate
            purity[b] = y["val"][m].mean()
    pval_bin = np.array([purity[b] if np.isfinite(purity[b]) else 1.0 for b in b_val])

    best = None
    for c in np.round(np.arange(0.80, 1.005, 0.005), 3):
        t_grid = np.unique(np.round(p_val, 2))
        for t in t_grid[t_grid >= 0.5]:
            approve_val = (p_val >= t) & (pval_bin >= c)
            n_pos = int(y["val"].sum())
            fp = int((approve_val & (y["val"] == 0)).sum())
            cov = approve_val.mean()
            if fp / (len(y["val"]) - n_pos) <= MAX_RISK:
                if best is None or cov > best["coverage_val"]:
                    best = dict(t=float(t), c=float(c), coverage_val=float(cov),
                               fpr_val=fp / (len(y["val"]) - n_pos))
    if best is None:
        return None
    # veto band for the report: highest band whose purity < c
    veto_bands = [(round(q[b], 3), round(q[b + 1], 3), round(purity[b], 3))
                  for b in range(NBINS)
                  if np.isfinite(purity[b]) and purity[b] < best["c"] and q[b] >= best["t"] * 0 + 0.5]
    approve_test = (p_test >= best["t"]) & (
        np.array([purity[b] if np.isfinite(purity[b]) else 1.0
                  for b in np.clip(np.searchsorted(q, p_test, side="right") - 1, 0, NBINS - 1)]) >= best["c"])
    # metrics on the veto-gate: binary approval; AUROC uses raw scores (ranking
    # unchanged), risk/coverage use the actual gate
    op = operating_point(cases["test"], np.where(approve_test, 1.0, 0.0), 0.5)
    d = discriminate(cases["test"], p_test)
    ci = grouped_bootstrap_ci(cases["test"], p_test, metric="auroc", reps=REPS, seed=SEED)
    champ = aligned("full-benchmark-v1", CHAMPION, "test")
    delta = grouped_bootstrap_ci(cases["test"], p_test, champ,
                                 metric="auroc", reps=REPS, seed=SEED) if arm != CHAMPION \
        else dict(point=float("nan"), lo=float("nan"), hi=float("nan"))
    return dict(name=name, arm=arm, run=run, t=best["t"], c=best["c"],
                coverage_val=best["coverage_val"], fpr_val=best["fpr_val"],
                veto_bands=veto_bands, n_veto_bands=len(veto_bands),
                coverage_test=float(op.coverage), fpr_test=float(op.risk_fpr),
                auroc_raw=d["auroc"], auroc_lo=ci["lo"], auroc_hi=ci["hi"],
                delta_vs_champ=delta["point"], delta_lo=delta["lo"], delta_hi=delta["hi"],
                certified=bool(op.risk_fpr <= 0.05 + 1e-9 and op.coverage > 0.001),
                licence=LICENSE.get(arm, "ok?"))


# champion baseline under plain threshold for the comparison column
from sufficiency_lab.thresholds import select_threshold  # noqa: E402
champ_scores = {s: aligned("full-benchmark-v1", CHAMPION, s) for s in ("val", "test")}
champ_thr = select_threshold(cases["val"], list(champ_scores["val"]),
                             strategy=CHAMPION, max_risk=MAX_RISK)
champ_op = operating_point(cases["test"], champ_scores["test"], champ_thr.threshold)
print(f"champion plain-threshold baseline: thr {champ_thr.threshold:.2f} "
      f"test FPR {champ_op.risk_fpr:.3f} coverage {champ_op.coverage:.3f}")

rows = []
for run, arms in RUNS.items():
    for arm in arms:
        try:
            p_val = aligned(run, arm, "val")
            p_test = aligned(run, arm, "test")
        except FileNotFoundError:
            continue
        r = evaluate(f"{arm}|hubris-veto", run, arm, p_val, p_test)
        if r:
            rows.append(r)
            print(f"{arm:<24} t {r['t']:.2f} c {r['c']:.3f} veto_bands {r['n_veto_bands']} "
                  f"valCov {r['coverage_val']:.3f} -> test FPR {r['fpr_test']:.3f} "
                  f"cov {r['coverage_test']:.3f} auroc {r['auroc_raw']:.3f} "
                  f"{'CERT' if r['certified'] else 'fail'}")

rows.sort(key=lambda r: (-int(r["certified"]), -r["coverage_test"]))
outdir = ROOT / "results" / "tail-calibration-v1"
outdir.mkdir(exist_ok=True)
(outdir / "tail-cal-stats.json").write_text(json.dumps(
    {"rows": rows, "seed": SEED, "bootstrap": REPS, "max_risk": MAX_RISK,
     "n_bins": NBINS,
     "champion_baseline": {"thr": champ_thr.threshold, "fpr_test": champ_op.risk_fpr,
                           "coverage_test": champ_op.coverage}}, indent=1))
lines = ["# Hubris-veto tail calibration — banked scores, val-only tuning", "",
         "approve iff raw >= t AND purity_band(raw) >= c. Purity bins from val "
         "quantiles (40); (t,c) maximize val coverage @ <=5% val FPR; test once.", "",
         "| arm | t | c | veto bands | val cov | test FPR | test cov | AUROC [CI] | cert | licence |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    bands = "; ".join(f"[{a:.2f},{b:.2f}) pur {p:.2f}" for a, b, p in r["veto_bands"]) or "—"
    lines.append(f"| {r['arm']} | {r['t']:.2f} | {r['c']:.3f} | {bands} | "
                 f"{r['coverage_val']:.3f} | {r['fpr_test']:.3f} | {r['coverage_test']:.3f} | "
                 f"{r['auroc_raw']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] | "
                 f"{'✅' if r['certified'] else '❌'} | {r['licence']} |")
(outdir / "tail-cal-report.md").write_text("\n".join(lines) + "\n")
print("wrote results/tail-calibration-v1/")
