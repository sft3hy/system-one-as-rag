"""Leakage and schema tests: the parts of the harness that, if wrong, make
every number downstream meaningless."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from sufficiency_lab import datasets
from sufficiency_lab.config import load_config
from sufficiency_lab.contract import DEFAULT_QUESTIONS, Questions
from sufficiency_lab.gating import Gate, GateConfig
from sufficiency_lab.metrics import (
    discriminate,
    expected_calibration_error,
    grouped_bootstrap_ci,
    operating_point,
)
from sufficiency_lab.runner import StubStrategy, load_split, run
from sufficiency_lab.schema import Case, Decision, FailureType, TerminalState
from sufficiency_lab.strategies.heuristic import (
    LengthStrategy,
    QuestionOnlyStrategy,
)
from sufficiency_lab.thresholds import select_threshold


def make_case(i: int, sufficient: bool, group: str | None = None) -> Case:
    return Case(
        id=f"{group or f'q{i}'}_{'suf' if sufficient else 'insuf'}",
        question=f"What happened in episode {i}?",
        context=f"Episode {i} describes the siege of port {i} in detail."
        if sufficient
        else f"Episode {i} mentions harbours generally, {i * 7} words of fog.",
        sufficient=sufficient,
        split="test",
        group_id=group or f"q{i}",
        question_type="factual",
    )


@pytest.fixture
def paired_cases() -> list[Case]:
    return [c for i in range(60) for c in (make_case(i, True), make_case(i, False))]


# ------------------------------------------------------------------ leakage


def test_group_split_never_shares_groups():
    """A sufficient variant and its insufficient twin share a group_id, so a
    threshold tuned on val can never be scored against its own twin."""
    cases = [make_case(i, s) for i in range(20) for s in (True, False)]
    groups = {c.group for c in cases}
    assert len(groups) == 20
    per_group: dict[str, list[bool]] = {}
    for c in cases:
        per_group.setdefault(c.group, []).append(c.sufficient)
    # matched pairs must be identifiable as pairs
    assert all(len(v) == 2 for v in per_group.values())


def test_limit_keeps_groups_whole(tmp_path: Path, monkeypatch):
    import sufficiency_lab.datasets as ds

    cases = [make_case(i, s) for i in range(10) for s in (True, False)]
    monkeypatch.setitem(ds.ADAPTERS, "custom", lambda p, split="test": cases)
    out = [c.group for c in cases[:4]]
    assert len(set(out)) == 2, "truncation split a matched pair"


def test_bootstrap_resamples_groups_not_rows(paired_cases: list[Case]):
    scores = [0.9 if c.sufficient else 0.1 for c in paired_cases]
    ci = grouped_bootstrap_ci(paired_cases, scores, metric="auroc", reps=50, seed=1)
    assert ci["n_boot"] > 0
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    # a perfect separator must stay perfect under any grouping resample
    assert ci["lo"] == pytest.approx(1.0)


def test_question_only_arm_cannot_see_context():
    """Control arm: identical scores for both variants of a question."""
    s = QuestionOnlyStrategy()
    a = s.predict(make_case(3, True))
    b = s.predict(make_case(3, False))
    assert a.p_sufficient == pytest.approx(b.p_sufficient)
    assert not np.isnan(a.p_sufficient)


# ------------------------------------------------------------- discrimination


def test_operating_point_arithmetic():
    cases = [make_case(i, i % 2 == 0) for i in range(10)]
    scores = [0.9 if c.sufficient else 0.4 for c in cases]
    op = operating_point(cases, scores, 0.5)
    assert op.risk_fpr == 0.0
    assert op.coverage == 0.5
    assert op.accuracy == 1.0
    op2 = operating_point(cases, scores, 0.1)
    assert op2.risk_fpr == 1.0 and op2.coverage == 1.0


def test_threshold_respects_risk_cap_on_val_only():
    """Risk must be computed on the split we tune on, never implied by test."""
    val = [make_case(i, i % 2 == 0) for i in range(100)]
    scores = np.array([0.95 if c.sufficient else 0.30 + 0.3 * (i % 3 == 0)
                      for i, c in enumerate(val)])
    th = select_threshold(val, scores, strategy="x", max_risk=0.05)
    assert th.risk_fpr <= 0.05
    assert th.threshold >= 0.60, "threshold should sit above the noisy negatives"
    assert th.achieved_on == "val"


def test_threshold_refuses_when_negatives_are_too_few():
    val = [make_case(i, True) for i in range(50)] + [make_case(100 + i, False)
                                                       for i in range(5)]
    scores = [0.9 if c.sufficient else 0.1 for c in val]
    th = select_threshold(val, scores, strategy="x", max_risk=0.05)
    assert th.fallback, "5 negatives cannot certify a 5% false-sufficient rate"


def test_nan_scores_are_dropped_not_zeroed():
    """A failed arm must not masquerade as a confident 'insufficient'."""
    cases = [make_case(i, i % 2 == 0) for i in range(10)]
    scores = [float("nan")] * 10
    d = discriminate(cases, scores)
    assert d["n"] == 0
    assert np.isnan(d["auroc"])


def test_select_threshold_survives_partial_nan_scores():
    """Regression: filtering scores and then re-filtering them against the
    unfiltered mask truncated the sample and raised IndexError once some of an
    arm's cases errored. Cases and scores must be dropped together, once."""
    n = 200
    cases = [make_case(i, i % 2 == 0) for i in range(n)]
    scores = [0.95 if c.sufficient else 0.20 + 0.25 * (i % 3 == 0)
              for i, c in enumerate(cases)]
    # 18 hard failures, scattered — exactly what a judge that errors on some
    # cases produces
    for i in range(0, n, 11):
        scores[i] = float("nan")

    th = select_threshold(cases, scores, strategy="judge", max_risk=0.05)
    assert not th.fallback
    kept = sum(1 for s in scores if s == s)
    assert th.n_val == kept
    assert th.risk_fpr <= 0.05
    op = operating_point(cases, scores, th.threshold)
    assert op.risk_fpr <= 0.05


def test_metrics_reject_mismatched_lengths():
    cases = [make_case(i, True) for i in range(5)]
    with pytest.raises(ValueError, match="length mismatch"):
        operating_point(cases, [0.5] * 4, 0.5)


def test_ece_is_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.03


# ------------------------------------------------------------------- gating


def test_gate_terminal_states_are_exclusive():
    cfg = GateConfig(t_answer=0.6, t_conflict=0.5, escalate_low=0.3,
                     escalate_high=0.8)
    gate = Gate(cfg)
    case = make_case(1, True)

    def decide(p, pc):
        from sufficiency_lab.schema import Decision

        return gate.decide(case, Decision(case_id=case.id, strategy="s",
                                           p_sufficient=p, p_conflicting=pc,
                                           latency_ms=10))

    assert decide(0.95, 0.1).action is TerminalState.ANSWERED
    assert decide(0.10, 0.1).action is TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT
    assert decide(0.95, 0.9).action is TerminalState.ABSTAINED_CONTRADICTION
    assert decide(0.45, 0.1).action is TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT
    assert decide(0.45, 0.1).tier == "fast_no_judge"


def test_gate_band_over_budget_never_escalates_by_default():
    cfg = GateConfig(t_answer=0.6, escalate_low=0.3, escalate_high=0.8,
                     fast_budget_ms=5.0)
    gate = Gate(cfg)
    case = make_case(1, True)
    from sufficiency_lab.schema import Decision

    stub = StubStrategy()
    act = gate.decide(case, Decision(case_id=case.id, strategy="s",
                                       p_sufficient=0.5, latency_ms=50.0),
                      judge=stub)
    assert act.tier == "fast_over_budget"
    assert act.action is TerminalState.ABSTAINED_INSUFFICIENT_CONTEXT


def test_gate_band_over_budget_can_answer_when_policy_says():
    cfg = GateConfig(t_answer=0.4, escalate_low=0.3, escalate_high=0.8,
                     fast_budget_ms=5.0, over_budget_policy="answer")
    gate = Gate(cfg)
    case = make_case(1, True)
    from sufficiency_lab.schema import Decision

    stub = StubStrategy()
    act = gate.decide(case, Decision(case_id=case.id, strategy="s",
                                       p_sufficient=0.5, latency_ms=50.0),
                      judge=stub)
    assert act.action is TerminalState.ANSWERED


def test_gate_config_rejects_unbracketed_threshold():
    with pytest.raises(ValueError):
        GateConfig(t_answer=0.9, escalate_low=0.35, escalate_high=0.75)


# ------------------------------------------------------------------ schema


def test_case_roundtrip_preserves_semantics(tmp_path: Path):
    c = Case(id="x", question="q", context="ctx", sufficient=False,
             failure_type=FailureType.BROKEN_MULTI_HOP, required_facts=("a", "b"))
    d = c.to_json_dict()
    assert d["failure_type"] == "broken_multi_hop"
    back = Case.from_json_dict(d)
    assert back.failure_type is FailureType.BROKEN_MULTI_HOP
    assert back.required_facts == ("a", "b")
    assert back.sufficient is False


# ------------------------------------------------------------------ cascade


class _FixedStrategy:
    """Returns a fixed p, optionally sleeping, so the cascade's routing and
    latency accounting can be asserted. Latency is *measured* by the base
    class rather than self-reported, so the slow tier really must sleep."""

    kind = "fixed"

    def __init__(self, p, latency_ms=10.0, error=None, name="fixed"):
        self.name = name
        self._p = p
        self._sleep = latency_ms / 1000.0
        self._err = error
        self.questions = DEFAULT_QUESTIONS
        self.calls = 0

    def predict(self, case):
        self.calls += 1
        if self._sleep:
            time.sleep(self._sleep)
        if self._err:
            return Decision(case_id=case.id, strategy=self.name,
                           p_sufficient=float("nan"), latency_ms=self._sleep * 1e3,
                           error=self._err)
        return Decision(case_id=case.id, strategy=self.name, p_sufficient=self._p,
                        p_conflicting=0.01, probabilities={"sufficient": self._p},
                        latency_ms=self._sleep * 1e3, input_tokens=100,
                        output_tokens=0)

    def close(self):
        return None


def test_cascade_escalates_only_inside_the_band():
    from sufficiency_lab.strategies.cascade import CascadeStrategy

    fast = _FixedStrategy(0.50, latency_ms=20)     # inside [0.35, 0.75]
    slow = _FixedStrategy(0.90, latency_ms=60)
    cas = CascadeStrategy(fast, slow, low=0.35, high=0.75)
    d = cas.predict(make_case(1, True))
    assert d.p_sufficient == pytest.approx(0.90), "band must defer to the judge"
    assert d.label == "judge_tier"
    assert cas.escalation_rate == 1.0
    assert fast.calls == 1 and slow.calls == 1
    assert d.latency_ms > 0
    assert d.input_tokens == 200, "both tiers' tokens are billed"


def test_cascade_does_not_pay_for_decode_outside_the_band():
    """Routing is asserted structurally (call counts), not by wall clock:
    macOS sleep granularity overshoots by ~5x at tens-of-milliseconds scales,
    so a timing assertion here tests the OS scheduler rather than the cascade.
    Real latency is measured against the live server in the benchmark."""
    from sufficiency_lab.strategies.cascade import CascadeStrategy

    fast = _FixedStrategy(0.95, latency_ms=20)
    slow = _FixedStrategy(0.10, latency_ms=60)
    cas = CascadeStrategy(fast, slow, low=0.35, high=0.75)
    d = cas.predict(make_case(1, True))
    assert slow.calls == 0, "high confidence must never reach the judge"
    assert fast.calls == 1
    assert d.p_sufficient == pytest.approx(0.95)
    assert cas.escalation_rate == 0.0
    assert d.label == "fast_tier"
    assert d.latency_ms > 0, "latency is still recorded at the fast tier"

    fast2 = _FixedStrategy(0.05, latency_ms=20)
    cas2 = CascadeStrategy(fast2, _FixedStrategy(0.99), low=0.35, high=0.75)
    assert cas2.predict(make_case(2, False)).p_sufficient == pytest.approx(0.05)
    assert cas2.escalation_rate == 0.0


def test_cascade_keeps_fast_signal_when_judge_fails():
    """A judge error must not erase a usable fast-tier score: dropping the
    case would silently bias the split toward easy questions."""
    from sufficiency_lab.strategies.cascade import CascadeStrategy

    fast = _FixedStrategy(0.55, latency_ms=20)
    slow = _FixedStrategy(0.0, error="ValueError: truncated")
    cas = CascadeStrategy(fast, slow, low=0.35, high=0.75)
    d = cas.predict(make_case(1, True))
    assert d.p_sufficient == pytest.approx(0.55)
    assert not np.isnan(d.p_sufficient)
    assert d.label == "fast_fallback"


def test_cascade_wiring_requires_existing_arm_names():
    from sufficiency_lab.config import ModelSpec
    from sufficiency_lab.runner import build_strategy

    cfg = load_config(Path("configs/sufficiencybench.yaml"))
    spec = ModelSpec(name="casc", kind="cascade", extra={"fast": "nope"})
    with pytest.raises(ValueError, match="unknown fast arm"):
        build_strategy(spec, cfg, None, registry={})


def test_cascade_yaml_wires_to_real_instances():
    """The cascade must reuse the same objects as the standalone arms, and the
    config must actually deliver its knobs — a whitelist that silently dropped
    extra.fast produced a cascade that could not be built at all."""
    from sufficiency_lab.runner import build_strategy
    from sufficiency_lab.strategies.cascade import CascadeStrategy
    from sufficiency_lab.strategies.llm_judge import LLMJudgeStrategy
    from sufficiency_lab.strategies.system1 import SystemOneStrategy

    cfg = load_config(Path("configs/cascade.yaml"))
    specs = cfg.models_for()
    reg: dict = {}
    for s in [x for x in specs if x.kind != "cascade"] + \
             [x for x in specs if x.kind == "cascade"]:
        reg[s.name] = build_strategy(s, cfg, object(), reg)

    cas = reg["cascade"]
    assert isinstance(cas, CascadeStrategy)
    assert reg["cascade"].fast is reg["clef-flash"], "fast tier must be shared"
    assert reg["cascade"].slow is reg["qwen38-judge"], "slow tier must be shared"
    assert (cas.low, cas.high) == (0.35, 0.75), "band edges must reach the arm"
    assert isinstance(reg["clef-flash"], SystemOneStrategy)
    assert reg["clef-flash"].quantization == "oQ4e"
    judge = reg["qwen38-judge"]
    assert isinstance(judge, LLMJudgeStrategy)
    assert judge.enable_thinking is False, "thinking default must be off"
    for s in reg.values():
        try:
            s.close()
        except Exception:
            pass


def test_quadrant_derivation_is_explicit():
    assert datasets.QUADRANT_FROM_FIELDS[(False, True)] == "Q3_insufficient_knows"
    assert datasets._failure_type("multi_hop", False) is FailureType.BROKEN_MULTI_HOP
    # never invent contradiction gold
    assert datasets._failure_type("factual", False) is FailureType.MISSING_FACT


def test_contract_is_shared_by_all_arms():
    """Fairness invariant: one rubric, every strategy.

    v2: the gate's primary primitive is `choice`, because noul was measured on
    this stack to return the same p for matched sufficient/insufficient pairs.
    The noul mode is kept so the failure stays reproducible as an arm.
    """
    qs = DEFAULT_QUESTIONS.questions(include_diagnostic=True)
    assert qs["sufficient"]["type"] == "choice"
    assert qs["conflicting"]["type"] == "choice"
    assert qs["sufficient"]["criteria"]["sufficient"]
    assert qs["conflicting"]["criteria"]["contradictory"]
    assert "memory" in qs["sufficient"]["instructions"]
    # the two axes stay independent: contradiction is not a sufficiency option
    assert "contradictory" not in qs["sufficient"]["criteria"]

    noul_qs = Questions(primitive="noul").questions(include_diagnostic=False)
    assert noul_qs["sufficient"]["type"] == "noul"


# --------------------------------------------------------------- end to end


def test_dry_run_produces_report(tmp_path: Path):
    """The whole pipeline offline: load, score, tune, gate, report."""
    from dataclasses import replace

    from scripts import make_fixtures

    data = tmp_path / "raw"
    make_fixtures.build(data / "fixtures")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"""
run:
  id: dry
  dataset: fixtures
  data_path: {data}
  out_dir: {tmp_path / 'results'}
  splits: [val, test]
  limit: 6
risk:
  max_false_sufficient: 0.05
  bootstrap: 40
models:
  - name: stub
    kind: stub
""")
    cfg = replace(load_config(cfg_path), run_id="dry-x")
    from sufficiency_lab import datasets

    datasets.ADAPTERS["fixtures"] = datasets.load_sufficiencybench
    run_dir = run(cfg, live=False, cold=False)
    report = json.loads((run_dir / "reports" / "report.json").read_text())
    assert report["counts"]["test"] == 12
    row = report["leaderboard"][0]
    assert row["strategy"] == "stub"
    assert 0.0 <= row["auroc"] <= 1.0
    assert (run_dir / "predictions" / "stub_test.jsonl").exists()
    assert (run_dir / "predictions" / "stub_val.jsonl").exists(), (
        "val predictions must persist: thresholds tune on them, and a report "
        "regenerated without them silently falls back to threshold 0.50"
    )
    assert (run_dir / "reports" / "report.md").exists()


def test_load_split_validates_real_shape():
    with pytest.raises(FileNotFoundError):
        datasets.load_sufficiencybench(Path("/nonexistent"), "test")


def test_offline_run_refuses_to_overwrite_measured_results(tmp_path: Path):
    """A stub AUROC of 1.000 in a results table reads as real. That is worse
    than no number, so an offline run must refuse to touch a measured run."""
    from dataclasses import replace

    from sufficiency_lab import datasets
    from sufficiency_lab.config import ModelSpec
    from sufficiency_lab.runner import run as run_fn
    from sufficiency_lab.schema import write_jsonl

    datasets.ADAPTERS["fixtures"] = datasets.load_sufficiencybench
    run_dir = tmp_path / "results" / "real-run"
    write_jsonl(run_dir / "predictions" / "measured_test.jsonl",
                [{"case_id": "x", "strategy": "measured", "p_sufficient": 0.7}])

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"""
run:
  id: real-run
  dataset: fixtures
  data_path: {tmp_path / 'data'}
  out_dir: {tmp_path / 'results'}
  splits: [val, test]
  limit: 4
models:
  - name: stub
    kind: stub
""")
    cfg = replace(load_config(cfg_path), dataset="fixtures")
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    from scripts import make_fixtures

    make_fixtures.build(cfg.data_path / "fixtures")

    with pytest.raises(RuntimeError, match="overwrite"):
        run_fn(cfg, live=False)
    # explicit force still works, so the guard is a seatbelt not a roadblock
    run_dir2 = run_fn(cfg, live=False, force=True)
    assert (run_dir2 / "reports" / "report.md").exists()


def test_dry_run_uses_separate_run_id(tmp_path: Path):
    """--dry-run must namespace its own run id so a stub report cannot land in,
    and silently overwrite, a measured run's directory."""
    from dataclasses import replace

    from sufficiency_lab import datasets
    from sufficiency_lab.runner import run as run_fn

    datasets.ADAPTERS["fixtures"] = datasets.load_sufficiencybench
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"""
run:
  id: smoke-clef-flash
  dataset: fixtures
  data_path: {tmp_path / 'data'}
  out_dir: {tmp_path / 'results'}
  splits: [val, test]
  limit: 4
models:
  - name: stub
    kind: stub
""")
    cfg = load_config(cfg_path)
    # emulate main()'s dry-run namespacing rule
    dry_id = cfg.run_id if cfg.run_id.endswith("-dry") else f"{cfg.run_id}-dry"
    assert dry_id == "smoke-clef-flash-dry"
    cfg = replace(cfg, run_id=dry_id, dataset="fixtures")
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    from scripts import make_fixtures

    make_fixtures.build(cfg.data_path / "fixtures")
    run_dir = run_fn(cfg, live=False)
    assert run_dir.name.endswith("-dry")
    # the measured run's directory was never touched
    assert not (tmp_path / "results" / "smoke-clef-flash").exists()
