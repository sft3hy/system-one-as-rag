"""Validation-only threshold selection.

Rule: never touch the test split here. The gate target is a ceiling on
false-sufficient rate — P(authorise an answer | context truly insufficient).
We pick the *lowest* threshold meeting it (maximum coverage under the risk
budget), and we require a minimum number of negative validation cases or we
refuse to pick, because a 5% FPR estimated on 20 negatives is noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import operating_point
from .schema import Case


@dataclass(frozen=True)
class Threshold:
    strategy: str
    threshold: float
    risk_fpr: float
    coverage: float
    n_val: int
    n_val_negative: int
    achieved_on: str = "val"
    fallback: bool = False


MIN_NEGATIVES = 40


def select_threshold(
    cases: Sequence[Case],
    scores: Sequence[float],
    *,
    strategy: str,
    max_risk: float = 0.05,
    grid: Sequence[float] | None = None,
    min_negatives: int = MIN_NEGATIVES,
) -> Threshold:
    grid = tuple(grid) if grid else tuple(round(0.01 * i, 2) for i in range(1, 100))
    if len(cases) != len(scores):
        raise ValueError(
            f"cases/scores length mismatch ({len(cases)} vs {len(scores)}): a "
            "threshold must come from the same decisions it will gate."
        )
    ok = ~np.isnan(np.asarray(scores, dtype=float))
    # Filter cases and scores together, exactly once. Filtering p first and
    # then zipping the filtered p against the unfiltered mask silently
    # truncated the sample (and raised once the counts diverged).
    kept = [c for c, k in zip(cases, ok) if k]
    p = np.asarray([s for s, k in zip(scores, ok) if k], dtype=float)
    y = np.array([1 if c.sufficient else 0 for c in kept], dtype=int)
    n_neg = int((y == 0).sum())
    if n_neg < min_negatives:
        # too few negatives to certify the risk budget: default conservative
        return Threshold(strategy, 0.90, float("nan"), float("nan"),
                         int(len(y)), n_neg, achieved_on="val", fallback=True)

    pts = [operating_point(kept, p, t) for t in grid]
    eligible = [pt for pt in pts if pt.risk_fpr <= max_risk and pt.coverage > 0]
    if not eligible:
        safest = min(pts, key=lambda q: (q.risk_fpr, -q.coverage))
        return Threshold(strategy, safest.threshold, safest.risk_fpr, safest.coverage,
                         int(len(y)), n_neg, achieved_on="val", fallback=True)
    best = min(eligible, key=lambda q: (q.threshold, -q.coverage))
    return Threshold(strategy, best.threshold, best.risk_fpr, best.coverage,
                     int(len(y)), n_neg)


def conflict_threshold(
    cases: Sequence[Case],
    scores: Sequence[float],
    *,
    strategy: str,
    max_false_block: float = 0.10,
    grid: Sequence[float] | None = None,
) -> Threshold:
    """Second tier. Blocking on contradiction must not eat too many genuinely
    sufficient cases, so this side is tuned the other way round: the highest
    block rate we accept is bounded by the false-block rate on sufficient data.
    """
    grid = tuple(grid) if grid else tuple(round(0.01 * i, 2) for i in range(1, 100))
    if len(cases) != len(scores):
        raise ValueError(
            f"cases/scores length mismatch ({len(cases)} vs {len(scores)}): a "
            "threshold must come from the same decisions it will gate — pass the "
            "val decisions, not the report-split decisions."
        )
    y = np.array([1 if c.sufficient else 0 for c in cases], dtype=int)
    p = np.asarray([s for s in scores], dtype=float)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return Threshold(strategy, 0.50, float("nan"), float("nan"), int(len(y)), 0,
                         fallback=True)
    best_t, best_fb = 0.50, 1.0
    for t in grid:
        blocked = p >= t
        false_block = float(((blocked) & (y == 1)).sum() / n_pos)
        if false_block <= max_false_block:
            best_t, best_fb = t, false_block
    return Threshold(strategy, best_t, float("nan"), best_fb, int(len(y)), n_pos)
