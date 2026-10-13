"""Cascade: cheap decision tier, autoregressive judge only when unsure.

The economics of gating are a band, not a point. Cases whose p_sufficient
lands inside ``[low, high]`` are ambiguous, and that band is where a slow
judge can pay for itself. Everything outside it is settled at the fast tier.

Thresholds come from thresholds.py (validation only). This module never tunes.

One honest caveat, recorded per escalated case: outside the band the score is
Clef's, inside the band it is the judge's. Those are different distributions —
the no-think judge answers on ~{0.0, 0.1, 0.95, 1.0} while Clef spreads over
(0, 0.94) — so a single threshold over the raw mixture measurably collapses
coverage (FINDINGS §11). `calibrator` fixes this before fusion: each tier's
score becomes empirical P(sufficient | score, tier) fitted on val, and one
threshold then means the same thing on both sides of the band. `tier` is
persisted regardless, so any downstream analysis can stratify by the tier that
actually produced the number.
"""

from __future__ import annotations

from time import perf_counter

from ..schema import Case
from .base import BaseStrategy, _clamp


class CascadeStrategy(BaseStrategy):
    kind = "cascade"

    def __init__(self, fast: BaseStrategy, slow: BaseStrategy,
                 low: float = 0.35, high: float = 0.75, name: str = "cascade",
                 calibrator=None):
        """`calibrator` (sufficiency_lab.calibration.TierCalibrator, optional)
        maps each tier's raw score onto empirical reliability fitted on val.
        Without it the cascade emits raw mixed scores and the threshold tuner
        must thread the needle between the judge's quantization spikes — which
        measurably collapses coverage (FINDINGS §11). With it, one threshold
        means one thing on both sides of the band.
        """
        super().__init__(fast.questions)
        self.fast, self.slow = fast, slow
        self.low, self.high = low, high
        self.name = name
        self.calibrator = calibrator
        self.escalations = 0
        self.total = 0

    @property
    def escalation_rate(self) -> float:
        return self.escalations / self.total if self.total else 0.0

    def _score(self, case: Case):
        self.total += 1
        start = perf_counter()
        fast = self.fast.predict(case)
        if fast.error or not (self.low <= fast.p_sufficient <= self.high):
            probs = dict(fast.probabilities)
            probs["escalated"] = 0.0
            probs["raw_p_sufficient"] = fast.p_sufficient
            p = fast.p_sufficient
            if self.calibrator is not None:
                p = self.calibrator.transform_fast(p)
                probs["calibrated"] = 1.0
            meta = {"label": "fast_tier", "ttft_ms": fast.latency_ms,
                   "input_tokens": fast.input_tokens,
                   "output_tokens": fast.output_tokens}
            return p, fast.p_conflicting, probs, meta

        self.escalations += 1
        slow = self.slow.predict(case)
        elapsed = (perf_counter() - start) * 1000.0
        if slow.error:  # never lose the fast signal
            probs = dict(fast.probabilities)
            probs["escalated"] = 1.0
            probs["raw_p_sufficient"] = fast.p_sufficient
            return fast.p_sufficient, fast.p_conflicting, probs, {
                "label": "fast_fallback", "ttft_ms": elapsed,
                "escalation_error": slow.error,
            }
        probs = dict(slow.probabilities)
        probs["escalated"] = 1.0
        probs["fast_p_sufficient"] = fast.p_sufficient
        probs["raw_p_sufficient"] = slow.p_sufficient
        meta = {
            "label": "judge_tier",
            "ttft_ms": elapsed,
            "input_tokens": (fast.input_tokens or 0) + (slow.input_tokens or 0),
            "output_tokens": slow.output_tokens,
        }
        p = _clamp(slow.p_sufficient)
        if self.calibrator is not None:
            p = self.calibrator.transform_slow(p)
            probs["calibrated"] = 1.0
        con = slow.p_conflicting if slow.p_conflicting is not None else fast.p_conflicting
        return p, con, probs, meta

    def close(self) -> None:
        self.fast.close()
        self.slow.close()
