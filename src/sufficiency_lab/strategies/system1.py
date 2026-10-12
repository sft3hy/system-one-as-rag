"""Clef / OpenJev adapter over omlx POST /v1/systemone.

Two operating points, both benchmarked:

``batch_questions=True``
    sufficiency + contradiction (+ optional diagnosis) as separate typed
    questions in ONE request. On Clef the backbone runs once over the state
    and the joint schema head scores every option, so extra questions should
    cost ~nothing. That is a hypothesis, and it is measured (Δp vs split).

``batch_questions=False``
    one noul per request. The round-trip delta is the price of full
    independence between the two scores.

``truncate`` matters: Clef cuts state to its 16384-token context by default;
with truncate=False an oversized prompt is a 413, which we record rather than
silently drop.
"""

from __future__ import annotations

from typing import Sequence

from ..contract import ANSWER_OPTION, CONFLICT_OPTION, Questions
from ..omlx_client import OmlxClient, SystemOneAnswer
from ..schema import Case
from .base import BaseStrategy


def _positive(ans: SystemOneAnswer, option: str) -> float | None:
    """Read the gate probability out of whichever primitive was answered.

    ``noul`` -> its yes-probability. ``choice`` -> the named option's
    probability. Anything else is None, which the caller turns into a recorded
    adapter error rather than a silent 0.0 (a fake 'insufficient' would
    otherwise inflate apparent caution and corrupt the risk estimate).
    """
    if ans is None:
        return None
    if ans.noul is not None:
        return float(ans.noul)
    if ans.probabilities:
        for key, val in ans.probabilities.items():
            if key == option:
                return float(val)
    return None


class SystemOneStrategy(BaseStrategy):
    kind = "system1"

    def __init__(
        self,
        client: OmlxClient,
        model: str,
        questions: Questions,
        *,
        name: str | None = None,
        batch_questions: bool = True,
        include_diagnostic: bool = True,
        truncate: bool = True,
        quantization: str | None = None,
    ):
        super().__init__(questions)
        self.client = client
        self.model = model
        self.name = name or model
        self.batch_questions = batch_questions
        self.include_diagnostic = include_diagnostic
        self.truncate = truncate
        self.quantization = quantization

    def _score(self, case: Case):
        state = self.questions.state(case.question, case.context)

        if self.batch_questions:
            qs = self.questions.questions(include_diagnostic=self.include_diagnostic)
            resp = self.client.systemone(
                self.model, state, qs, truncate=self.truncate
            )
            return self._from_batch(resp)
        return self._from_split(state)

    # ------------------------------------------------------------------ paths

    def _from_batch(self, resp) -> tuple[float, float | None, dict, dict]:
        suf = resp.answers.get("sufficient")
        if suf is None:
            raise RuntimeError(f"no 'sufficient' answer in {list(resp.answers)}")
        p_suf = _positive(suf, ANSWER_OPTION)
        if p_suf is None:
            raise RuntimeError(
                f"'sufficient' answer has no readable probability (type={suf.type})")
        con = resp.answers.get("conflicting")
        p_con = _positive(con, CONFLICT_OPTION) if con is not None else None
        probs: dict[str, float] = {"sufficient": p_suf}
        if p_con is not None:
            probs["conflicting"] = p_con
        diag = resp.answers.get("diagnosis")
        label = diag.choice if diag is not None else None
        if diag is not None and diag.probabilities:
            for k, v in diag.probabilities.items():
                probs[f"diagnosis_{k}"] = v
        meta = {
            "label": label,
            "input_tokens": resp.usage.get("prompt_tokens")
            or resp.usage.get("input_tokens"),
            "output_tokens": resp.usage.get("completion_tokens")
            or resp.usage.get("output_tokens"),
            "ttft_ms": resp.latency_ms,  # no decode: full latency == ttft
            "questions_in_request": len(resp.answers),
            "confidence": diag.confidence if diag else None,
        }
        return p_suf, p_con, probs, meta

    def _from_split(self, state) -> tuple[float, float | None, dict, dict]:
        all_q = self.questions.questions(include_diagnostic=False)
        r1 = self.client.systemone(self.model, state, {"sufficient": all_q["sufficient"]},
                                    truncate=self.truncate)
        r2 = self.client.systemone(self.model, state, {"conflicting": all_q["conflicting"]},
                                    truncate=self.truncate)
        p_suf = _positive(r1.answers["sufficient"], ANSWER_OPTION)
        p_con = _positive(r2.answers["conflicting"], CONFLICT_OPTION)
        if p_suf is None:
            raise RuntimeError("no readable probability on split sufficiency request")
        probs = {"sufficient": p_suf}
        if p_con is not None:
            probs["conflicting"] = p_con
        meta = {
            "input_tokens": (
                (r1.usage.get("prompt_tokens") or r1.usage.get("input_tokens") or 0)
                + (r2.usage.get("prompt_tokens") or r2.usage.get("input_tokens") or 0)
            ),
            "ttft_ms": r1.latency_ms,
            "questions_in_request": 1,
            "requests": 2,
        }
        return p_suf, p_con, probs, meta

    def warmup(self, cases: Sequence[Case]) -> None:
        super().warmup(cases)
