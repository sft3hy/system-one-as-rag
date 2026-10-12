"""Discrimination, calibration, risk, coverage, and grouped bootstrap CIs.

Label convention, stated once: positive == sufficient. p == p_sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .schema import Case

ECE_BINS = 15


@dataclass
class GatePoint:
    """One operating point on a strategy's risk/coverage curve."""

    threshold: float
    coverage: float
    risk_fpr: float  # P(authorise | truly insufficient)  <- the key risk metric
    false_insufficient_rate: float  # P(block | truly sufficient)
    accuracy: float
    precision: float
    recall: float
    f1: float
    answered: int
    n: int


def _clean(cases: Sequence[Case], scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = np.array([1 if c.sufficient else 0 for c in cases], dtype=int)
    p = np.asarray(scores, dtype=float)
    keep = ~np.isnan(p)
    groups = np.array([c.group for c in cases], dtype=object)
    return y[keep], p[keep], keep, groups[keep]


def discriminate(cases: Sequence[Case], scores: Sequence[float]) -> dict[str, float]:
    y, p, _, _ = _clean(cases, scores)
    out: dict[str, float] = {"n": int(len(y)), "n_pos": int(y.sum())}
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {**out, "auroc": float("nan"), "auprc": float("nan"),
                "brier": float("nan"), "ece": float("nan")}
    return {
        **out,
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
        "mean_pos_score": float(p[y == 1].mean()),
        "mean_neg_score": float(p[y == 0].mean()),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = len(y)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / total) * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def operating_point(cases: Sequence[Case], scores: Sequence[float],
                    threshold: float) -> GatePoint:
    y, p, _, _ = _clean(cases, scores)
    if len(y) == 0:
        return GatePoint(threshold, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    pred = p >= threshold
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    n_neg, n_pos = max(fp + tn, 1), max(tp + fn, 1)
    return GatePoint(
        threshold=float(threshold),
        coverage=(tp + fp) / len(y),
        risk_fpr=fp / n_neg,
        false_insufficient_rate=fn / n_pos,
        accuracy=(tp + tn) / len(y),
        precision=tp / max(tp + fp, 1),
        recall=tp / n_pos,
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        answered=tp + fp,
        n=len(y),
    )


def risk_coverage_curve(cases: Sequence[Case], scores: Sequence[float],
                        grid: Sequence[float]) -> list[GatePoint]:
    return [operating_point(cases, scores, t) for t in grid]


def stratified_auroc(cases: Sequence[Case], scores: Sequence[float],
                     key: str = "quadrant") -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    buckets: dict[str, list[int]] = {}
    for i, c in enumerate(cases):
        val = getattr(c, key, None) if key != "failure_type" else (
            c.failure_type.value if c.failure_type else None
        )
        if val is None:
            continue
        buckets.setdefault(str(val), []).append(i)
    for label, idxs in buckets.items():
        sub = [cases[i] for i in idxs]
        sub_scores = [scores[i] for i in idxs]
        m = discriminate(sub, sub_scores)
        out[label] = m
    return out


def grouped_bootstrap_ci(
    cases: Sequence[Case],
    scores_a: Sequence[float],
    scores_b: Sequence[float] | None = None,
    *,
    metric: str = "auroc",
    reps: int = 2000,
    seed: int = 13,
) -> dict[str, float]:
    """Bootstrap by question group so matched sufficient/insufficient pairs and
    multi-hop chains never split across resamples.

    With ``scores_b`` also returns the paired Δ(a-b) CI, which is the number
    that actually settles "is Clef better than the judge" — unpaired CIs on
    the same 2,224 conditions overlap so widely they answer nothing.
    """
    y_all = np.array([1 if c.sufficient else 0 for c in cases], dtype=int)
    pa = np.asarray(scores_a, dtype=float)
    pb = None if scores_b is None else np.asarray(scores_b, dtype=float)
    groups = np.array([c.group for c in cases], dtype=object)
    uniq, inv = np.unique(groups, return_inverse=True)

    rng = np.random.default_rng(seed)
    point = _metric_vec(metric, y_all, pa) if pb is None else _metric_delta(metric, y_all, pa, pb)
    draws: list[float] = []
    for _ in range(reps):
        picks = rng.integers(0, len(uniq), len(uniq))
        rows = np.flatnonzero(np.isin(inv, picks))
        if len(rows) < 8 or len(np.unique(y_all[rows])) < 2:
            continue
        val = _metric_vec(metric, y_all[rows], pa[rows])
        if pb is not None:
            val = val - _metric_vec(metric, y_all[rows], pb[rows])
        if np.isfinite(val):
            draws.append(float(val))
    if not draws:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "metric": metric}
    arr = np.array(draws)
    return {
        "point": float(point),
        "lo": float(np.percentile(arr, 2.5)),
        "hi": float(np.percentile(arr, 97.5)),
        "n_boot": len(draws),
        "metric": metric,
    }


def _metric_vec(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(y, p))
    if metric == "auprc":
        return float(average_precision_score(y, p))
    if metric == "brier":
        return float(brier_score_loss(y, p))
    if metric == "ece":
        return expected_calibration_error(y, p)
    raise ValueError(f"unknown metric {metric}")


def _metric_delta(metric: str, y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> float:
    return _metric_vec(metric, y, pa) - _metric_vec(metric, y, pb)


def latency_summary(latencies_ms: Sequence[float]) -> dict[str, float]:
    a = np.asarray([x for x in latencies_ms if np.isfinite(x)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
        "throughput_qps_serial": float(1000.0 / a.mean()) if a.mean() > 0 else 0.0,
        "total_s": float(a.sum() / 1000.0),
    }
