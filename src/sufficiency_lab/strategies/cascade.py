"""Cascade: cheap decision tier, autoregressive judge only when unsure.

The economics of gating are a band, not a point. Cases whose p_sufficient
lands inside ``[low, high]`` are ambiguous, and that band is where a slow
judge can pay for itself. Everything outside it is settled at the fast tier.

Thresholds come from thresholds.py (validation only). This module never tunes.

One honest caveat, recorded per escalated case: outside the band the score is
Clef's calibrated choice probability, inside the band it is the judge's. Those
are different distributions, so a single threshold over a cascade is tuned on a
*mixture*. That is normal for cascades and the only way to buy coverage without
paying decode on every request — but it is why `tier` is persisted, so any
downstream calibration can be stratified by the tier that actually produced the
number.
"""

from __future__ import annotations

from time import perf_counter

from ..schema import Case
from .base import BaseStrategy, _clamp


class CascadeStrategy(BaseStrategy):
    kind = "cascade"

    def __init__(self, fast: BaseStrategy, slow: BaseStrategy,
                 low: float = 0.35, high: float = 0.75, name: str = "cascade"):
        super().__init__(fast.questions)
        self.fast, self.slow = fast, slow
        self.low, self.high = low, high
        self.name = name
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
            meta = {"label": "fast_tier", "ttft_ms": fast.latency_ms,
                   "input_tokens": fast.input_tokens,
                   "output_tokens": fast.output_tokens}
            return fast.p_sufficient, fast.p_conflicting, probs, meta

        self.escalations += 1
        slow = self.slow.predict(case)
        elapsed = (perf_counter() - start) * 1000.0
        if slow.error:  # never lose the fast signal
            probs = dict(fast.probabilities)
            probs["escalated"] = 1.0
            return fast.p_sufficient, fast.p_conflicting, probs, {
                "label": "fast_fallback", "ttft_ms": elapsed,
                "escalation_error": slow.error,
            }
        probs = dict(slow.probabilities)
        probs["escalated"] = 1.0
        probs["fast_p_sufficient"] = fast.p_sufficient
        meta = {
            "label": "judge_tier",
            "ttft_ms": elapsed,
            "input_tokens": (fast.input_tokens or 0) + (slow.input_tokens or 0),
            "output_tokens": slow.output_tokens,
        }
        p = _clamp(slow.p_sufficient)
        con = slow.p_conflicting if slow.p_conflicting is not None else fast.p_conflicting
        return p, con, probs, meta

    def close(self) -> None:
        self.fast.close()
        self.slow.close()
