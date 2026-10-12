"""The calibrated sufficiency gate: two independent scores, two thresholds,
explicit terminal states, and a latency budget that decides whether the slow
tier is even allowed to run.

    p_sufficient >= t_answer          -> candidate ANSWERED
    p_conflicting >= t_conflict       -> ABSTAINED_CONTRADICTION (overrides)
    inside the ambiguous band         -> escalate to judge, if the budget allows
    otherwise                         -> ABSTAINED_INSUFFICIENT_CONTEXT

Budget honesty: if the fast tier has already spent the latency budget, the
judge never runs and the case resolves at the fast tier under
``over_budget_policy``. That policy is a product decision, recorded per case.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .schema import Case, Decision, GateAction, TerminalState


@dataclass(frozen=True)
class GateConfig:
    """Three numbers, one shape:

        p < escalate_low   -> block (clearly insufficient)
        t_answer <= p      -> answer
        escalate_low..high -> ambiguous: the only band a slow judge can enter

    ``t_answer`` comes from thresholds.py (risk-capped, val-only). The band is
    a latency/economics choice, not a statistical one.
    """

    t_answer: float = 0.50
    t_conflict: float = 0.50
    escalate_low: float = 0.35
    escalate_high: float = 0.75
    fast_budget_ms: float = 80.0
    judge_budget_ms: float = 1200.0
    over_budget_policy: str = "abstain"  # "abstain" | "answer" | "escalate_anyway"

    def __post_init__(self) -> None:
        if self.over_budget_policy not in {"abstain", "answer", "escalate_anyway"}:
            raise ValueError(f"bad over_budget_policy {self.over_budget_policy}")
        if not (self.escalate_low <= self.t_answer <= self.escalate_high):
            raise ValueError(
                f"band [{self.escalate_low}, {self.escalate_high}] must bracket "
                f"t_answer={self.t_answer}"
            )


class Gate:
    def __init__(self, cfg: GateConfig, name: str = "gate"):
        self.cfg = cfg
        self.name = name
        self.counts: dict[str, int] = {}

    def decide(self, case: Case, decision: Decision, *,
               elapsed_ms: float | None = None,
               judge=None) -> GateAction:
        cfg = self.cfg
        spent = elapsed_ms if elapsed_ms is not None else decision.latency_ms
        start = perf_counter()

        # 1. contradiction overrides everything except an explicit opt-out
        if decision.p_conflicting is not None and decision.p_conflicting >= cfg.t_conflict:
            return self._emit(case.id, decision, TerminalState.ABSTAINED_CONTRADICTION,
                              "conflict_tier", spent, start)

        if decision.p_sufficient >= cfg.escalate_high:
            return self._emit(case.id, decision, TerminalState.ANSWERED, "fast", spent, start)

        if decision.p_sufficient < cfg.escalate_low:
            return self._emit(case.id, decision,
                              TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT, "fast",
                              spent, start)

        # 2. ambiguous band -> the only place the slow tier can earn its keep
        if judge is None:
            state = (TerminalState.ANSWERED if decision.p_sufficient >= cfg.t_answer
                    else TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT)
            return self._emit(case.id, decision, state, "fast_no_judge", spent, start)

        if spent >= cfg.fast_budget_ms and cfg.over_budget_policy != "escalate_anyway":
            state = (TerminalState.ANSWERED
                     if cfg.over_budget_policy == "answer" and decision.p_sufficient >= cfg.t_answer
                     else TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT)
            return self._emit(case.id, decision, state, "fast_over_budget", spent, start)

        slow = judge.predict(case)
        total = spent + slow.latency_ms
        if slow.error:
            state = (TerminalState.ANSWERED
                    if decision.p_sufficient >= cfg.t_answer
                    else TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT)
            return self._emit(case.id, decision, state, "judge_error_fallback", total,
                              start)
        if slow.latency_ms > cfg.judge_budget_ms:
            # judge ran but blew its own budget: record, still use the verdict
            return self._emit(case.id, decision,
                              TerminalState.ANSWERED if slow.p_sufficient >= cfg.t_answer
                              else TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT,
                              "judge_over_budget", total, start)
        state = (TerminalState.ANSWERED if slow.p_sufficient >= cfg.t_answer
                 else TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT)
        return self._emit(case.id, decision, state, "judge", total, start)

    def _emit(self, case_id: str, d: Decision, state: TerminalState, tier: str,
              spent_ms: float, start: float) -> GateAction:
        total = spent_ms + (perf_counter() - start) * 1000.0
        self.counts[state.value] = self.counts.get(state.value, 0) + 1
        return GateAction(
            case_id=case_id,
            strategy=self.name,
            action=state,
            p_sufficient=d.p_sufficient,
            p_conflicting=d.p_conflicting,
            tier=tier,
            latency_ms=total,
        )
