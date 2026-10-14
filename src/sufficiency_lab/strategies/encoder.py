"""Encoder decision models (Tier B) via a local sidecar that speaks the
TypeSafe SystemOne subset: POST /system_one {state, questions} -> {answers}.

Laya (`laya` Router) and Von (`von-sdk`) both implement exactly this typed
protocol natively, so ONE strategy serves the whole Tier B family — the
sidecar (`scripts/encoder_sidecar.py`) is the only model-specific piece.

Same rubric as every other arm: the questions payload comes from
DEFAULT_QUESTIONS verbatim (contract fairness rule), and the gate score is
the `sufficient` option probability of the `sufficiency` choice question —
identical semantics to SystemOneStrategy. Runs off-GPU in its own venv, so
encoder decisions never occupy the decision-engine slot (§6/§10 hygiene).
"""

from __future__ import annotations

from time import perf_counter

import httpx

from ..schema import Case
from .base import BaseStrategy


class EncoderSidecarStrategy(BaseStrategy):
    kind = "encoder"

    def __init__(self, url: str, questions, name: str = "encoder",
                 timeout_s: float = 120.0, include_diagnostic: bool = True):
        super().__init__(questions)
        self.name = name
        self.include_diagnostic = include_diagnostic
        self._http = httpx.Client(base_url=url, timeout=timeout_s)

    def _score(self, case: Case):
        start = perf_counter()
        state = self.questions.state(case.question, case.context)
        r = self._http.post("/systemone", json={
            "state": state,
            "questions": self.questions.questions(
                include_diagnostic=self.include_diagnostic),
        })
        r.raise_for_status()
        body = r.json()
        answers = body.get("answers", {})
        suf = answers.get("sufficient") or {}
        # a choice answer carries per-option probabilities; a noul answer is one prob
        probs_map = suf.get("probabilities") or {}
        p = float(probs_map.get("sufficient", suf.get("probability", suf.get("noul", 0.0))))
        con = answers.get("conflicting") or {}
        p_conf = (float((con.get("probabilities") or {}).get("contradictory",
                  con.get("probability", con.get("noul"))))
                  if con else None)
        probs = {"sufficient": p}
        if p_conf is not None:
            probs["conflicting"] = p_conf
        for qid, ans in answers.items():
            for k, v in (ans.get("probabilities") or {}).items():
                probs[f"{qid}_{k}"] = float(v)
        elapsed = (perf_counter() - start) * 1000.0
        meta = {"label": suf.get("label") or suf.get("choice"),
                "ttft_ms": elapsed,
                "truncated": bool(body.get("truncated", False))}
        return p, p_conf, probs, meta

    def close(self) -> None:
        self._http.close()
