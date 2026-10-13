"""Tier calibration: map the judge's scores onto flash's scale before fusion.

The problem (FINDINGS §11, measured): clef and the no-think judge emit scores
on incompatible scales. Flash spreads over (0, 0.94); the judge answers on
~{0.0, 0.1, 0.95, 1.0}. A single threshold over that mixture has no honest
operating point — it must sit above the judge's 0.95 spike to keep its
confident-wrong cases out, and starves coverage to 3%.

The fix calibrates ONLY the judge, onto flash's scale:

  fast (decisive)   passes through raw. Flash is the reference scale — native
                    ECE ~0.11 — and recalibrating it destroys the fine tail its
                    tuned 0.89 threshold depends on. A first attempt that
                    isotonic-ed BOTH tiers collapsed every score into one 0.98
                    plateau whose real accuracy was 0.76, and coverage to zero.
  slow (escalated)  score -> empirical accuracy (isotonic on val band cases),
                    then accuracy -> the flash score that shares that accuracy
                    (inverting flash's own fitted accuracy curve).

One threshold t on the fused score then means one thing everywhere: "as
trustworthy as a flash score >= t". The judge's 0.95 spike — right only ~59%
of the time inside the band — lands wherever flash sits at 59%, far below any
risk-budget threshold. The cliff between the judge's 0.95 and 1.0 survives,
but it is now the HONEST cliff: it reflects the judge actually being that
discrete, and a threshold cannot fall on the wrong side of it by accident.

Fit on val only, ever. Serializes to JSON knots so a live run loads exactly
the mapping the offline simulation produced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression


def _invert(iso: IsotonicRegression, target: float) -> float:
    """The flash score whose fitted accuracy equals `target` (monotone)."""
    x = np.asarray(iso.X_thresholds_, dtype=float)
    y = np.asarray(iso.y_thresholds_, dtype=float)
    if target <= y[0]:
        return float(x[0])
    if target >= y[-1]:
        return float(x[-1])
    # y is nondecreasing with plateaus; np.interp needs strictly increasing
    # xp. For each accuracy level keep the TOP of its x-plateau — the safe,
    # higher reading — so "accuracy a" never maps below the flash scores that
    # actually achieve a.
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if ys and yi == ys[-1]:
            xs[-1] = max(xs[-1], xi)
        elif ys and xi == xs[-1]:
            ys[-1] = max(ys[-1], yi)
        else:
            xs.append(float(xi)); ys.append(float(yi))
    if len(xs) < 2:  # degenerate fit: no usable curve, don't pretend
        return float(xs[0]) if xs else 0.0
    return float(np.interp(target, xs, ys))


class TierCalibrator:
    """Judge -> flash scale via accuracy alignment. Flash passes through."""

    def __init__(self, fast: IsotonicRegression | None = None,
                 slow: IsotonicRegression | None = None,
                 meta: dict | None = None):
        self.fast = fast   # flash score -> accuracy (the reference curve)
        self.slow = slow   # judge score -> accuracy
        self.meta = meta or {}

    # ------------------------------------------------------------------ fit

    @staticmethod
    def _fit(scores: Sequence[float], labels: Sequence[int],
             min_points: int = 16) -> IsotonicRegression:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=float)
        if len(s) < min_points:
            raise ValueError(
                f"tier has n={len(s)} val cases — too few to fit a monotone "
                "reliability curve without memorizing; collect more val data "
                "rather than trusting an underpowered map")
        iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
        iso.fit(s, y)
        return iso

    @classmethod
    def fit(cls, fast_scores: Sequence[float], fast_labels: Sequence[int],
            slow_scores: Sequence[float], slow_labels: Sequence[int],
            *, min_points: int = 16) -> "TierCalibrator":
        fast = cls._fit(fast_scores, fast_labels, min_points)
        slow = cls._fit(slow_scores, slow_labels, min_points)
        ladder = {}
        for v in sorted(set(float(x) for x in slow_scores)):
            ladder[str(v)] = {"accuracy": round(float(slow.predict([v])[0]), 3),
                              "flash_equiv": round(_invert(fast, float(slow.predict([v])[0])), 3)}
        meta = {
            "n_fast": int(len(fast_scores)), "n_slow": int(len(slow_scores)),
            "fast_range": [float(min(fast_scores)), float(max(fast_scores))],
            "slow_range": [float(min(slow_scores)), float(max(slow_scores))],
            # printed at fit time so nobody ships this mapping blind
            "ladder": ladder,
        }
        return cls(fast, slow, meta)

    # ------------------------------------------------------------- transform

    def transform_fast(self, p: float) -> float:
        # flash is the reference scale — raw passthrough, see module docstring
        return p

    def transform_slow(self, p: float) -> float:
        if self.slow is None or self.fast is None:
            return p
        acc = float(self.slow.predict([p])[0])
        return _invert(self.fast, acc)

    # ------------------------------------------------------------- serialize

    def to_dict(self) -> dict:
        def knots(iso: IsotonicRegression | None) -> dict | None:
            if iso is None:
                return None
            return {"x": [float(x) for x in iso.X_thresholds_],
                    "y": [float(y) for y in iso.y_thresholds_]}
        return {"kind": "isotonic-tier-v2-flashscale", "fast": knots(self.fast),
                "slow": knots(self.slow), "meta": self.meta}

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "TierCalibrator":
        d = json.loads(Path(path).read_text())
        if d.get("kind") != "isotonic-tier-v2-flashscale":
            raise ValueError(f"not a supported calibration file: {path}")

        def build(block: dict | None) -> IsotonicRegression | None:
            if not block:
                return None
            iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            x = np.asarray(block["x"], dtype=float)
            y = np.asarray(block["y"], dtype=float)
            iso.fit(x, y, sample_weight=np.full(len(x), 1000))
            return iso

        return cls(build(d["fast"]), build(d["slow"]), d.get("meta", {}))
