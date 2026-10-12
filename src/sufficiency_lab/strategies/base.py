"""Strategy protocol. One input, one probability, one latency.

Adapters stay thin on purpose: no strategy may carry its own threshold or its
own rubric (see contract.py), and none may mutate the case it is given.
"""

from __future__ import annotations

from time import perf_counter
from typing import Protocol, Sequence

from ..contract import DEFAULT_QUESTIONS, Questions
from ..schema import Case, Decision


class Strategy(Protocol):
    name: str
    kind: str

    def warmup(self, cases: Sequence[Case]) -> None: ...
    def predict(self, case: Case) -> Decision: ...
    def close(self) -> None: ...


class BaseStrategy:
    """Shared plumbing: latency timing, error capture, rubric ownership."""

    name: str = "base"
    kind: str = "abstract"

    def __init__(self, questions: Questions = DEFAULT_QUESTIONS):
        self.questions = questions
        self._errors = 0

    # overridden by adapters
    def _score(self, case: Case) -> tuple[float, float | None, dict[str, float], dict]:
        raise NotImplementedError

    def predict(self, case: Case) -> Decision:
        start = perf_counter()
        try:
            p_suf, p_conf, probs, meta = self._score(case)
            latency = (perf_counter() - start) * 1000.0
            return Decision(
                case_id=case.id,
                strategy=self.name,
                p_sufficient=_clamp(p_suf),
                p_conflicting=None if p_conf is None else _clamp(p_conf),
                label=meta.get("label"),
                probabilities=probs,
                latency_ms=latency,
                ttft_ms=meta.get("ttft_ms"),
                input_tokens=meta.get("input_tokens"),
                output_tokens=meta.get("output_tokens"),
                attempts=meta.get("attempts", 1),
                contract_version=self.questions.version,
            )
        except Exception as exc:  # a broken arm must not sink the run
            self._errors += 1
            return Decision(
                case_id=case.id,
                strategy=self.name,
                p_sufficient=float("nan"),
                latency_ms=(perf_counter() - start) * 1000.0,
                attempts=0,
                error=f"{type(exc).__name__}: {exc}"[:400],
            )

    def warmup(self, cases: Sequence[Case]) -> None:
        for case in cases:
            self.predict(case)

    def close(self) -> None:
        return None


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
