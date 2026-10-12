"""Per-tier thresholds for a cascade run — reproducible version of the
decision that chose 27B over the cascade (FINDINGS §11).

The live cascade (results/cascade-flash-judge-v1) shares one threshold across
two score distributions that barely overlap: the no-think judge answers on
{0.0, 0.1, 0.95, 1.0}, Clef-Flash spreads across (0, 0.94). The tuner picks a
point above the judge's spike to keep its confident-wrong cases out, and the
judge's own correct answers starve with them: coverage 0.030.

This script shows the two honest shapes:
  1. shared threshold, tuned on val (what the harness does today)
  2. per-tier acceptance: decisive cases keep flash's threshold, escalated
     cases get a judge threshold, each capped so the blended false-sufficient
     rate stays inside the risk budget

and transfers each to test with no refitting.

Run: uv run python scripts/tier_thresholds.py results/cascade-flash-judge-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BAND = (0.35, 0.75)


def load(d: Path, arm: str, split: str):
    pred_dir = d / "predictions"
    fp = pred_dir / f"{arm}_{split}.jsonl"
    if not fp.exists():
        return None, None, None
    raw = {json.loads(l)["id"]: bool(json.loads(l)["sufficient"])
           for l in open(d / "raw" / f"sufficiencybench_{split}.jsonl")}
    ids, ps = [], []
    for l in open(fp):
        r = json.loads(l)
        ids.append(r["case_id"])
        ps.append(float(r["p_sufficient"]))
    y = np.array([1 if raw[i] else 0 for i in ids])
    return np.array(ps), y, ids


def operating(y, accepted):
    acc = np.asarray(accepted, dtype=bool)
    cov = float(acc.mean())
    fpr = float(acc[y == 0].mean()) if (y == 0).any() else float("nan")
    return cov, fpr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--band", default=f"{BAND[0]},{BAND[1]}")
    ap.add_argument("--budget", type=float, default=0.05)
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.band.split(","))
    d = Path(a.run_dir)

    fp_val, y_val, ids_val = load(d, "cascade", "val")
    pf_val, _, _ = load(d, "clef-flash", "val")
    if fp_val is None or pf_val is None:
        print("need cascade + clef-flash val predictions")
        return 1
    esc_val = (pf_val >= lo) & (pf_val <= hi)

    print(f"{'shape':<34}{'val cov':>9}{'val fpr':>9}{'test cov':>10}{'test fpr':>10}")
    print("-" * 72)

    # shape 1: shared threshold, tuned on val (today's behaviour). Rounded
    # grid so 0.95 means 0.95 — with an accumulating grid the candidate
    # 0.95000000000000007 silently excludes the judge's 0.95 spike and
    # reports a fake fpr of 0.000 for the exact threshold that keeps 17.
    best = None
    for t in np.round(np.arange(0.50, 0.995, 0.005), 3):
        cov, fpr = operating(y_val, fp_val >= t)
        if fpr <= a.budget and (best is None or cov > best[1]):
            best = (float(t), cov, fpr)
    t_shared = best[0]
    fp_te, y_te, _ = load(d, "cascade", "test")
    if fp_te is not None:
        cov_te, fpr_te = operating(y_te, fp_te >= t_shared)
    else:
        cov_te = fpr_te = float("nan")
    neg_v = int((y_val == 0).sum())
    kept17 = int((fp_val[esc_val & (y_val == 0)] >= 0.95).sum())
    print(f"{'shared threshold (tuned val)':<34}{best[1]:>9.3f}{best[2]:>9.3f}"
          f"{cov_te:>10.3f}{fpr_te:>10.3f}   thr={t_shared:.3f} "
          f"(shared 0.95 alone would keep {kept17} band-neg, fpr ~0.11)")

    # shape 2: per-tier. Both tiers' false-sufficient counts are already
    # fractions of the SAME negative pool, so their blended contribution is a
    # plain sum — dividing the leftover by the escalation rate (first cut at
    # this) inflated the judge tier's allowance ~1/23 and happily certified an
    # FPR of 0.16 against a 0.05 budget. The blended budget is:
    #     (flash_kept_neg + judge_kept_neg) / all_neg <= budget
    #
    # The grid must be ROUNDED, not accumulated: np.arange(0.50, ..., 0.005)
    # lands on 0.95000000000000007, which silently excludes a stored 0.95 and
    # "certifies" a threshold whose entire safety margin is float dust. The
    # judge answers on exactly {0.95, 1.0}, so the only threshold between them
    # is a rounding accident waiting to happen (it happened here, first run).
    esc_rate = float(esc_val.mean())
    print(f"\nescalation rate (val): {esc_rate:.1%}")
    neg = (y_val == 0)
    grid = np.round(np.arange(0.50, 1.001, 0.01), 2)
    best2 = None
    for t_f in grid:
        acc_f = (~esc_val) & (pf_val >= t_f)
        kept_f = int((acc_f & neg).sum())
        if kept_f / max(int(neg.sum()), 1) > a.budget:
            continue
        room_j = int(a.budget * int(neg.sum())) - kept_f   # negatives the judge may keep
        for t_j in grid:
            acc_j = esc_val & (fp_val >= t_j)
            kept_j = int((acc_j & neg).sum())
            if kept_j > room_j:
                continue
            cov = float((acc_f | acc_j).mean())
            fpr = float((kept_f + kept_j) / max(int(neg.sum()), 1))
            if best2 is None or cov > best2[2]:
                best2 = (float(t_f), float(t_j), cov, fpr, kept_f, kept_j)
    if best2 is None:
        print("no per-tier point satisfies the blended budget")
        return 0
    t_f, t_j, cov_v, fpr_v, kf_v, kj_v = best2
    pf_te, _, _ = load(d, "clef-flash", "test")
    if pf_te is not None and fp_te is not None:
        esc_te = (pf_te >= lo) & (pf_te <= hi)
        acc_te = ((~esc_te) & (pf_te >= t_f)) | (esc_te & (fp_te >= t_j))
        cov_te, fpr_te = operating(y_te, acc_te)
    else:
        cov_te = fpr_te = float("nan")
    print(f"{'per-tier (flash | judge)':<34}{cov_v:>9.3f}{fpr_v:>9.3f}"
          f"{cov_te:>10.3f}{fpr_te:>10.3f}   t_f={t_f:.2f} t_j={t_j:.2f} "
          f"(kept neg: flash {kf_v} + judge {kj_v} / {int(neg.sum())})")

    # Margin: the binding risk is band negatives clustering just UNDER the
    # chosen threshold. Report how many re-enter when the bar drops one grid
    # step — a jump from 0 to many means the whole margin is a single
    # quantization spike, not real separation.
    band_neg = fp_val[esc_val & (y_val == 0)]
    if len(band_neg):
        at = int((band_neg >= t_j).sum())
        below = int((band_neg >= t_j - 0.01).sum())
        cliff = below - at
        top = float(band_neg.max())
        print(f"  band negatives at >=t_j={t_j:.2f}: {at}; at >=t_j-0.01: {below} "
              f"(+{cliff}). Highest band-negative score={top:.2f}. "
              + ("MARGIN IS A SPIKE: a one-cent drop re-admits "
                 f"{cliff}/{len(band_neg)} wrong cases, so t_j sits on top of a "
                 "quantization peak (the no-think judge answers on ~{0.95,1.0}). "
                 "Escalation needs an INDEPENDENT signal — thinking, or "
                 "flash+judge agreement — not a higher bar on a saturated score."
                 if cliff > max(2, 0.1 * len(band_neg)) else
                 "margin is spread, threshold looks stable."))

    print("\nRead: per-tier acceptance transfers (test tuned only on val). "
          "If its test coverage still trails the 27B leaderboard row, the "
          "judge is not earning its escalations — deploy the bigger decision "
          "model, or calibrate escalation scores onto one scale before "
          "thresholding (FINDINGS §11 option a).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
