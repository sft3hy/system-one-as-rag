"""Experiment runner: identical inputs, interchangeable arms, paired outputs.

Every arm sees exactly the same ordered Case list, so any difference in the
output is attributable to the decision method and to nothing else.

Emitted per run:
  predictions/{strategy}.jsonl   per-case scores, latency, tokens, errors
  resources/{strategy}.json      peak RSS / CPU / free-mem / cold load
  latency/{strategy}.json        serial + concurrent percentiles
  thresholds.json                validation-only operating points
  report.json / report.md        leaderboard, risk-coverage, batch-vs-split
  raw/{dataset}_{split}.json     frozen input snapshot (reproducibility)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace as dc_replace
from pathlib import Path
from typing import Sequence

from . import datasets
from .config import REPO, ModelSpec, RunConfig, load_config
from .contract import DEFAULT_QUESTIONS
from .gating import Gate, GateConfig
from .metrics import (
    discriminate,
    grouped_bootstrap_ci,
    latency_summary,
    operating_point,
    risk_coverage_curve,
    stratified_auroc,
)
from .omlx_client import OmlxClient
from .resources import ResourceMonitor, gpu_snapshot, unload_model
from .schema import Case, Decision, write_jsonl
from .strategies.ar_decision import ARDecisionStrategy
from .strategies.base import BaseStrategy
from .strategies.cascade import CascadeStrategy
from .strategies.encoder import EncoderSidecarStrategy
from .strategies.heuristic import (
    LexicalCoverageStrategy,
    LengthStrategy,
    QuestionOnlyStrategy,
)
from .strategies.llm_judge import LLMJudgeStrategy
from .strategies.system1 import SystemOneStrategy
from .thresholds import Threshold, conflict_threshold, select_threshold


class StubStrategy(BaseStrategy):
    """Deterministic offline arm: validates plumbing without a server."""

    kind = "stub"

    def __init__(self, name: str = "stub"):
        super().__init__(DEFAULT_QUESTIONS)
        self.name = name

    def _score(self, case: Case):
        sig = sum(ord(c) for c in case.id) % 97
        p_suf = (0.06 + 0.88 * (sig / 97.0)) if case.sufficient else (
            0.04 + 0.80 * ((96 - sig) / 97.0)
        )
        p_con = 0.02 if case.sufficient else 0.08
        return p_suf, p_con, {"sufficient": p_suf, "conflicting": p_con}, {
            "label": "answerable" if case.sufficient else "missing_fact",
            "input_tokens": len(case.context) // 4,
            "ttft_ms": 1.0,
        }


# --------------------------------------------------------------------- setup


def build_strategy(spec, cfg: RunConfig, client: OmlxClient | None,
                   registry: dict[str, "BaseStrategy"] | None = None) -> BaseStrategy:
    q = DEFAULT_QUESTIONS
    if spec.kind == "stub":
        return StubStrategy(spec.name)
    if spec.kind == "heuristic":
        which = (spec.model or "lexical").lower()
        if which == "length":
            return LengthStrategy()
        if which == "question_only":
            return QuestionOnlyStrategy()
        return LexicalCoverageStrategy()
    if spec.kind == "system1":
        if client is None:
            raise RuntimeError(f"arm {spec.name} needs a live omlx server")
        return SystemOneStrategy(
            client,
            spec.model,
            q,
            name=spec.name,
            batch_questions=spec.batch_questions,
            include_diagnostic=bool(spec.extra.get("diagnostic", True)),
            truncate=cfg.omlx.truncate,
            quantization=spec.extra.get("quantization"),
        )
    if spec.kind == "llm_judge":
        return LLMJudgeStrategy(
            cfg.omlx, spec.model, q, name=spec.name, max_tokens=spec.max_tokens,
            enable_thinking=bool(spec.extra.get("thinking", False)),
            temperature=float(spec.extra.get("temperature", 0.0)),
        )
    if spec.kind == "ar_decision":
        return ARDecisionStrategy(
            cfg.omlx, spec.model, q, name=spec.name,
            samples=int(spec.extra.get("samples", 12)),
            temperature=float(spec.extra.get("temperature", 1.0)))
    if spec.kind == "encoder":
        url = spec.extra.get("sidecar_url")
        if not url:
            raise ValueError(
                f"encoder arm {spec.name!r} needs extra.sidecar_url "
                "(start scripts/encoder_sidecar.py for this model first)")
        q_used = q
        primitive = spec.extra.get("primitive")
        if primitive:
            # von's native axis is the noul scalar (measured AUROC 0.740 vs
            # 0.58 on choice options); noul wants a declarative condition,
            # not our interrogative sufficiency rubric
            q_used = dc_replace(
                q, primitive=primitive,
                sufficient_instructions=spec.extra.get(
                    "noul_instructions", q.sufficient_instructions))
        return EncoderSidecarStrategy(
            url, q_used, name=spec.name,
            include_diagnostic=bool(spec.extra.get("diagnostic", False)))
    if spec.kind == "cascade":
        # fast/slow reference other configured arms by name, so the cascade and
        # the standalone arms it reuses are guaranteed to be the same code paths
        # with the same rubric — no second implementation to drift apart.
        reg = registry or {}
        fast_name = spec.extra.get("fast")
        slow_name = spec.extra.get("slow")
        for role, nm in (("fast", fast_name), ("slow", slow_name)):
            if not nm:
                raise ValueError(
                    f"cascade arm {spec.name!r} needs extra.{role}: <arm name>")
            if nm not in reg:
                raise ValueError(
                    f"cascade arm {spec.name!r} references unknown {role} arm "
                    f"{nm!r}; configured arms: {sorted(reg)}")
        fast, slow = reg[fast_name], reg[slow_name]
        calibrator = None
        cal_path = spec.extra.get("calibration")
        if cal_path:
            from .calibration import TierCalibrator
            p = Path(cal_path)
            if not p.is_absolute():
                p = REPO / cal_path
            # fail loud: a cascade configured to be calibrated that silently
            # runs uncalibrated reproduces the coverage collapse in §11
            if not p.exists():
                raise FileNotFoundError(
                    f"cascade {spec.name!r}: calibration file {cal_path!r} not "
                    "found — fit it with scripts/fit_cascade_calibration.py from "
                    "a calibration run's val predictions")
            calibrator = TierCalibrator.load(p)
        return CascadeStrategy(
            fast, slow,
            low=float(spec.extra.get("escalate_low", cfg.latency.escalate_low)),
            high=float(spec.extra.get("escalate_high", cfg.latency.escalate_high)),
            name=spec.name,
            calibrator=calibrator,
        )
    raise ValueError(f"unknown kind {spec.kind!r} for arm {spec.name!r}")


def load_split(cfg: RunConfig, split: str) -> list[Case]:
    adapter = datasets.ADAPTERS[cfg.dataset]
    root = cfg.data_path / cfg.dataset
    if cfg.dataset in ("sufficiencybench", "fixtures"):
        cases = adapter(root, split)
    else:
        suffix = ".jsonl" if cfg.dataset == "custom" else ".json"
        cases = adapter(root / f"{split}{suffix}", split)
    if cfg.limit:
        # keep whole groups: splitting a matched pair breaks the benchmark
        groups, out = set(), []
        for c in cases:
            if c.group not in groups and len(groups) >= cfg.limit:
                break
            groups.add(c.group)
            out.append(c)
        return out
    return cases


# ---------------------------------------------------------------------- runs


def run_arm(
    strategy: BaseStrategy,
    cases: Sequence[Case],
    cfg: RunConfig,
    *,
    monitor: bool = True,
) -> tuple[list[Decision], dict]:
    mon = ResourceMonitor(strategy.name).start() if monitor else None
    decisions: list[Decision] = []

    warm = list(cases[: cfg.latency.warmup])
    if warm:
        strategy.warmup(warm)

    for i, case in enumerate(cases):
        d = strategy.predict(case)
        decisions.append(d)
        if mon and i == 0:
            mon.profile.cold_first_latency_ms = d.latency_ms

    prof = mon.stop() if mon else None
    return decisions, _arm_stats(strategy, decisions, prof)


def _arm_stats(strategy: BaseStrategy, decisions: Sequence[Decision], prof) -> dict:
    in_tok = sum(d.input_tokens or 0 for d in decisions)
    out_tok = sum(d.output_tokens or 0 for d in decisions)
    errs = [d for d in decisions if d.error]
    stats = {
        "strategy": strategy.name,
        "kind": getattr(strategy, "kind", "?"),
        "n": len(decisions),
        "errors": len(errs),
        "error_rate": round(len(errs) / max(len(decisions), 1), 4),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "tokens_per_case": round(in_tok / max(len(decisions), 1), 1),
        "contract_version": getattr(strategy, "questions", DEFAULT_QUESTIONS).version,
        "latency_serial": latency_summary([d.latency_ms for d in decisions]),
        "server": gpu_snapshot(),
    }
    if prof:
        prof.input_tokens, prof.output_tokens = in_tok, out_tok
        prof.requests = len(decisions)
        stats["resources"] = prof.summary()
    if errs:
        kinds: dict[str, int] = {}
        for d in errs:
            kinds[(d.error or "").split(":")[0]] = kinds.get((d.error or "").split(":")[0], 0) + 1
        stats["error_kinds"] = kinds
    return stats


def measure_concurrency(strategy: BaseStrategy, cases: Sequence[Case],
                        cfg: RunConfig) -> dict[str, dict]:
    """Throughput and tail latency at 1/4/8 in flight.

    omlx's max_concurrent_requests is 8 on this box; beyond that the client is
    just queueing, so percentiles past the server limit measure our own wait.
    """
    out: dict[str, dict] = {}
    pool_cases = list(cases[: max(cfg.latency.warmup * 8, 24)])
    for level in cfg.latency.concurrency:
        if not pool_cases:
            break
        stamps: list[float] = []
        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            for d in pool.map(strategy.predict, pool_cases):
                if d.ttft_ms is not None:
                    stamps.append(d.ttft_ms)
        wall = time.perf_counter() - wall_start
        s = latency_summary(stamps)
        s["wall_s"] = round(wall, 3)
        s["throughput_qps"] = round(len(pool_cases) / wall, 2) if wall > 0 else 0.0
        s["n_per_round"] = len(pool_cases)
        out[f"c{level}"] = s
    return out


def measure_cold(cfg: RunConfig, strategy: BaseStrategy, cases: Sequence[Case],
                 spec) -> dict:
    """Cold path: evict, then time the first real request.

    If eviction is unavailable we say so; a 'cold' number measured against an
    already-resident model is worse than no number at all.
    """
    evicted = unload_model(cfg.omlx.base_url, cfg.omlx.resolved_key(), spec.model or "")
    if not evicted:
        return {"evicted": False,
                "note": "no admin unload endpoint matched; cold latency unavailable"}
    time.sleep(3.0)
    t0 = time.perf_counter()
    d = strategy.predict(cases[0])
    return {
        "evicted": True,
        "first_request_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        "model_load_note": "first request includes weight load + graph build",
    }


def compare_batch_vs_split(cfg: RunConfig, cases: Sequence[Case],
                           spec) -> dict | None:
    """Does packing sufficiency+contradiction+diagnosis into ONE prefill cost
    anything, and does it move the probabilities?

    This is the load-bearing assumption of the whole two-tier design, so it
    gets its own measurement rather than a citation from the PR description.
    """
    if spec.kind != "system1" or not spec.batch_questions:
        return None
    client = OmlxClient(cfg.omlx)
    try:
        batched = SystemOneStrategy(client, spec.model, DEFAULT_QUESTIONS,
                                     name=spec.name + "_batched", batch_questions=True)
        split = SystemOneStrategy(client, spec.model, DEFAULT_QUESTIONS,
                                   name=spec.name + "_split", batch_questions=False)
        subset = list(cases[: max(cfg.limit or 0, 40)])
        bd, bs = [], []
        for c in subset:
            bd.append(batched.predict(c))
            bs.append(split.predict(c))
        pairs = [(a.p_sufficient, b.p_sufficient) for a, b in zip(bd, bs)
                 if not (a.error or b.error)]
        if not pairs:
            return {"n": 0, "error": "both arms failed"}
        da = [abs(x - y) for x, y in pairs]
        lat_b = [d.latency_ms for d in bd if not d.error]
        lat_s = [d.latency_ms for d in bs if not d.error]
        same = sum(1 for x, y in pairs
                   if (x >= 0.5) == (y >= 0.5)) / len(pairs)
        return {
            "n": len(pairs),
            "mean_abs_delta_p": round(statistics.fmean(da), 4),
            "max_abs_delta_p": round(max(da), 4),
            "sign_agreement_at_0.5": round(same, 4),
            "p50_batched_ms": round(statistics.median(lat_b), 1) if lat_b else None,
            "p50_split_ms": round(statistics.median(lat_s), 1) if lat_s else None,
            "split_over_batched_x": (
                round(statistics.median(lat_s) / statistics.median(lat_b), 2)
                if lat_b and lat_s else None
            ),
        }
    finally:
        client.close()


# --------------------------------------------------------------------- main


def run(cfg: RunConfig, *, live: bool = True, cold: bool = False,
        force: bool = False) -> Path:
    run_dir = cfg.run_dir
    # Guard: an offline run must not silently overwrite a measured run. A stub
    # AUROC of 1.000 sitting in a results table is worse than no number.
    existing = list((run_dir / "predictions").glob("*.jsonl")) if run_dir.exists() else []
    if not live and existing and not force:
        raise RuntimeError(
            f"{run_dir} already holds {len(existing)} prediction files. An offline "
            "(stub) run would overwrite a measured report. Use a run id ending in "
            "'-dry', or pass force=True if you really mean it."
        )
    for sub in ("predictions", "resources", "latency", "reports", "figures"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(cfg, s) for s in cfg.splits}
    for split, cases in splits.items():
        # JSONL content, .jsonl suffix — write_jsonl and the readers must agree
        write_jsonl(run_dir / "raw" / f"{cfg.dataset}_{split}.jsonl",
                    [c.to_json_dict() for c in cases])

    client = OmlxClient(cfg.omlx) if live else None
    if client is not None:
        try:
            print(f"[server] models: {client.list_models()}")
            print(f"[server] decision models: {client.decision_models()}")
        except Exception as exc:
            print(f"[server] unreachable: {exc}")
            client.close()
            raise

    thresholds: dict[str, Threshold] = {}
    leaderboard: list[dict] = []
    gate_report: dict[str, dict] = {}
    report_split = cfg.splits[-1]

    specs = cfg.models_for()
    # Two passes: leaf arms first so cascades can reference them by name. A
    # cascade reuses the *same* strategy instances as the standalone arms, so
    # the fast tier in a cascade is provably the same code, rubric and client
    # as the arm measured on its own — not a second copy that can drift.
    registry: dict[str, BaseStrategy] = {}
    ordered = ([s for s in specs if s.kind != "cascade"]
               + [s for s in specs if s.kind == "cascade"])
    try:
        for spec in ordered:
            registry[spec.name] = build_strategy(spec, cfg, client, registry)

        for spec in ordered:
            strategy = registry[spec.name]
            tag = f"{spec.name} ({spec.kind})"

            # Run and persist EVERY split, not just the reported one. Thresholds
            # are tuned on val, so val predictions are part of the record —
            # keeping them only in memory meant a regenerated report silently
            # fell back to 0.50 and printed a FPR of 0.42 next to a real 0.03.
            split_scores: dict[str, list[float]] = {}
            split_dec: dict[str, list[Decision]] = {}
            split_wall: dict[str, float] = {}
            for split in cfg.splits:
                cases = splits[split]
                is_report = split == report_split
                t_split = time.perf_counter()
                dec, arm_stats = run_arm(strategy, cases, cfg, monitor=is_report)
                split_wall[split] = time.perf_counter() - t_split
                split_scores[split] = [d.p_sufficient for d in dec]
                split_dec[split] = dec
                write_jsonl(run_dir / "predictions" / f"{spec.name}_{split}.jsonl",
                          [d.to_json_dict() for d in dec])
                if is_report:
                    stats = arm_stats
                    dec_report = dec
            stats["wall_serial_s"] = split_wall

            scores = split_scores[report_split]
            test_cases = splits[report_split]
            val = splits.get("val") or []
            val_scores = split_scores.get("val") or []
            val_dec = split_dec.get("val") or []

            th = None
            if val and val_scores:
                th = select_threshold(val, val_scores, strategy=spec.name,
                                      max_risk=cfg.risk.max_false_sufficient)
                thresholds[spec.name] = th
                (run_dir / "thresholds.json").write_text(
                    json.dumps({k: asdict(v) for k, v in thresholds.items()}, indent=2))
                print(f"[val] {tag}: thr={th.threshold:.2f} "
                      f"fpr={th.risk_fpr:.3f} cov={th.coverage:.3f}"
                      + (" FALLBACK" if th.fallback else ""))

            disc = discriminate(test_cases, scores)
            th = th or select_threshold(test_cases, scores, strategy=spec.name,
                                        max_risk=cfg.risk.max_false_sufficient)
            op = operating_point(test_cases, scores, th.threshold)

            boot = grouped_bootstrap_ci(
                test_cases, scores, metric="auroc",
                reps=cfg.risk.bootstrap // 4, seed=cfg.risk.seed)

            stats.update({
                "split": report_split,
                "threshold_source": "val" if "val" in split_scores else "report_split",
                "discrimination": disc,
                "bootstrap_auroc": boot,
                "operating_point": asdict(op),
                "risk_curve": [
                    asdict(pt) for pt in risk_coverage_curve(
                        test_cases, scores, cfg.risk.grid[::2])
                ],
                "stratified": stratified_auroc(test_cases, scores, "quadrant"),
                "stratified_qtype": stratified_auroc(test_cases, scores, "question_type"),
                "threshold": asdict(th),
            })

            conflict = [d.p_conflicting for d in dec_report]
            if any(c is not None for c in conflict):
                stats["conflict_signal"] = _conflict_stats(
                    test_cases,
                    [c if c is not None else float("nan") for c in conflict])

            # Second tier, tuned on val decisions (never the report split).
            val_conf = [d.p_conflicting for d in val_dec
                        if d.p_conflicting is not None]
            val_conf_cases = [c for c, d in zip(val, val_dec)
                              if d.p_conflicting is not None]
            if len(val_conf_cases) >= 40 and len(val_conf) == len(val_conf_cases):
                stats["conflict_threshold"] = asdict(conflict_threshold(
                    val_conf_cases, val_conf, strategy=spec.name))
            elif not val_conf:
                stats["conflict_threshold"] = {
                    "calibrated": False,
                    "reason": "this arm emits no p_conflicting, or val has no "
                            "contradiction scores",
                }

            if spec.kind == "system1":
                stats["batch_vs_split"] = compare_batch_vs_split(cfg, test_cases, spec)
                if cold:
                    stats["cold"] = measure_cold(cfg, strategy, test_cases, spec)

            conc = measure_concurrency(strategy, test_cases, cfg)
            stats["latency_concurrent"] = conc
            (run_dir / "latency" / f"{spec.name}.json").write_text(
                json.dumps({"serial": stats["latency_serial"], **conc}, indent=2))

            # wall-clock per split, persisted so full-run estimates are derived
            # from real elapsed time rather than summed request latency.
            stats_path = run_dir / "stats.json"
            all_stats = (json.loads(stats_path.read_text())
                         if stats_path.exists() else {})
            all_stats[spec.name] = {
                "wall_serial_s": stats.get("wall_serial_s"),
                "tokens_per_case": stats.get("tokens_per_case"),
                "input_tokens": stats.get("input_tokens"),
                "output_tokens": stats.get("output_tokens"),
                "errors": stats.get("errors"),
                "batch_vs_split": stats.get("batch_vs_split"),
                "cold": stats.get("cold"),
                "conflict_signal": stats.get("conflict_signal"),
            }
            stats_path.write_text(json.dumps(all_stats, indent=2))

            if stats.get("resources"):
                (run_dir / "resources" / f"{spec.name}.json").write_text(
                    json.dumps(stats["resources"], indent=2))

            leaderboard.append({
                "strategy": spec.name, "kind": spec.kind,
                "quantization": spec.extra.get("quantization"),
                "gate": spec.gate,
                "auroc": disc["auroc"], "auprc": disc["auprc"],
                "ece": disc["ece"], "brier": disc["brier"],
                "threshold": th.threshold, "risk_fpr": op.risk_fpr,
                "coverage": op.coverage,
                "false_insufficient_rate": op.false_insufficient_rate,
                "p50_ms": stats["latency_serial"].get("p50_ms"),
                "p95_ms": stats["latency_serial"].get("p95_ms"),
                "peak_rss_gb": (stats.get("resources") or {}).get("peak_server_rss_gb"),
                "tokens_per_case": stats["tokens_per_case"],
                "errors": stats["errors"],
            })

            gc = GateConfig(
                t_answer=th.threshold,
                t_conflict=cfg.latency.t_conflict,
                escalate_low=min(cfg.latency.escalate_low, th.threshold),
                escalate_high=max(cfg.latency.escalate_high, th.threshold),
                fast_budget_ms=cfg.latency.fast_budget_ms,
                judge_budget_ms=cfg.latency.judge_budget_ms,
            )
            gate_report[spec.name] = {
                "gate_config": asdict(gc),
                "terminal_states": {},
                # t_conflict is NOT calibrated here: SufficiencyBench ships no
                # gold contradiction labels, so 0.5 is a placeholder and the
                # contradiction tier must be read as unvalidated.
                "t_conflict_calibrated": cfg.dataset != "sufficiencybench",
            }
            if isinstance(strategy, CascadeStrategy):
                stats["escalation_rate"] = round(strategy.escalation_rate, 4)
                stats["escalations"] = strategy.escalations
                gate_report[spec.name]["escalation_rate"] = round(
                    strategy.escalation_rate, 4)
            print(f"[test] {tag}: AUROC={disc['auroc']:.3f} AUPRC={disc['auprc']:.3f} "
                  f"ECE={disc['ece']:.3f} p50={stats['latency_serial'].get('p50_ms')}ms "
                  f"errors={stats['errors']}")
    finally:
        # Close once, after every arm: a cascade shares instances with the leaf
        # arms it references, so closing per-arm would hand a closed client to
        # whatever ran next.
        for s in registry.values():
            try:
                s.close()
            except Exception:
                pass
        if client is not None:
            client.close()

    report = {
        "run_id": cfg.run_id,
        "dataset": cfg.dataset,
        "contract_version": DEFAULT_QUESTIONS.version,
        "risk_target": cfg.risk.max_false_sufficient,
        "counts": {s: len(c) for s, c in splits.items()},
        "leaderboard": sorted(leaderboard, key=lambda r: -(r["auroc"] or 0)),
        "thresholds": {k: asdict(v) for k, v in thresholds.items()},
        "gates": gate_report,
        "paired_comparisons": _paired(run_dir, cfg),
    }
    (run_dir / "reports" / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "reports" / "report.md").write_text(render_markdown(report))
    print(f"\n[done] {run_dir / 'reports' / 'report.md'}")
    return run_dir


def _conflict_stats(cases: Sequence[Case], p_conf: Sequence[float]) -> dict:
    """Does the contradiction noul fire on genuinely contradictory evidence, or
    is it just echoing p_sufficient? Report the correlation, do not guess."""
    xs, ys = [], []
    for c, p in zip(cases, p_conf):
        if p is None or p != p:
            continue
        xs.append(1.0 if c.sufficient else 0.0)
        ys.append(p)
    if len(xs) < 4:
        return {"n": len(xs)}
    return {
        "n": len(xs),
        "mean_p_conflicting_sufficient": round(statistics.fmean(
            [y for x, y in zip(xs, ys) if x == 1.0]), 4),
        "mean_p_conflicting_insufficient": round(statistics.fmean(
            [y for x, y in zip(xs, ys) if x == 0.0]), 4),
        "note": "SufficiencyBench has no gold contradiction labels; these are "
                "diagnostic distributions, not accuracy",
    }


def _paired(run_dir: Path, cfg: RunConfig) -> dict:
    """Paired ΔAUROC with CI between every pair of arms, on common case IDs.

    Reads the streaming JSONL snapshots, never the whole run into memory at
    once, and skips any legacy `.json` snapshot so a half-migrated run directory
    cannot crash the report.
    """
    from .schema import Case as _Case
    from .schema import read_jsonl

    per: dict[str, dict[str, float]] = {}
    for path in sorted((run_dir / "predictions").glob("*.jsonl")):
        if not path.stem.endswith("_test"):
            continue  # paired comparisons are made on the reported split only
        rows = {r["case_id"]: r["p_sufficient"] for r in read_jsonl(path)
                if r.get("p_sufficient") == r.get("p_sufficient")}
        per[path.stem[: -len("_test")]] = rows
    if len(per) < 2:
        return {}
    names = sorted(per)
    common = set.intersection(*[set(v) for v in per.values()])
    case_objs: dict[str, _Case] = {}
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        for raw in read_jsonl(path):
            c = _Case.from_json_dict(raw)
            case_objs[c.id] = c
    cases_common = [case_objs[cid] for cid in sorted(common) if cid in case_objs]
    if len(cases_common) < 8:
        return {}
    out: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}__vs__{b}"] = grouped_bootstrap_ci(
                cases_common,
                [per[a][c.id] for c in cases_common],
                [per[b][c.id] for c in cases_common],
                metric="auroc", reps=cfg.risk.bootstrap // 4, seed=cfg.risk.seed)
    return out


def render_markdown(report: dict) -> str:
    lines = [
        f"# Sufficiency gate benchmark — {report['run_id']}",
        "",
        f"- dataset: `{report['dataset']}`  contract: `{report['contract_version']}`",
        f"- risk budget: **{report['risk_target']:.0%} false-sufficient** "
        "(thresholds tuned on val only)",
        f"- counts: {report['counts']}",
        "",
        "## Leaderboard",
        "",
        "| strategy | kind | quant | AUROC | ECE | thr | FPR | coverage | "
        "false-insuf | p50 ms | p95 ms | peak RSS GB | tok/case | errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["leaderboard"]:
        def fmt(x, spec="{:.3f}"):
            return spec.format(x) if isinstance(x, (int, float)) else "—"
        lines.append(
            f"| {r['strategy']} | {r['kind']} | {r.get('quantization') or '—'} "
            f"| {fmt(r['auroc'])} | {fmt(r['ece'])} | {fmt(r['threshold'])} "
            f"| {fmt(r['risk_fpr'])} | {fmt(r['coverage'])} "
            f"| {fmt(r['false_insufficient_rate'])} "
            f"| {fmt(r['p50_ms'], '{:.1f}')} | {fmt(r['p95_ms'], '{:.1f}')} "
            f"| {fmt(r['peak_rss_gb'], '{:.1f}')} | {fmt(r['tokens_per_case'], '{:.0f}')} "
            f"| {r['errors']} |"
        )
    lines += [
        "",
        "Reading it: **FPR** is the risk you asked to cap, **coverage** is how much",
        "traffic the gate lets through at that risk, and p50/p95 plus peak RSS are",
        "the price. A slower arm with a lower FPR at equal AUROC is the better gate.",
        "",
    ]
    paired = report.get("paired_comparisons") or {}
    if paired:
        lines += ["## Paired ΔAUROC (grouped bootstrap, 95% CI)", "",
                  "| A | B | Δ | CI |", "|---|---|---|---|"]
        for key, v in paired.items():
            a, b = key.split("__vs__")
            if v.get("point") != v.get("point"):
                continue
            sig = "yes" if (v["lo"] > 0 or v["hi"] < 0) else "no"
            lines.append(f"| {a} | {b} | {v['point']:+.4f} | [{v['lo']:+.4f}, "
                         f"{v['hi']:+.4f}] | significant: {sig} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from dataclasses import replace

    ap = argparse.ArgumentParser(
        prog="sufficiency-lab",
        description="Benchmark sufficiency gates (System 1 / judge / classifier) "
                    "on identical inputs.",
    )
    ap.add_argument("-c", "--config", required=True, help="configs/*.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="offline stub arm: validates schema, metrics, report")
    ap.add_argument("--limit", type=int, default=None, help="cap question groups")
    ap.add_argument("--cold", action="store_true",
                    help="also measure cold-load latency (evicts the model first)")
    ap.add_argument("--kinds", default=None,
                    help="run only these kinds, e.g. 'heuristic' or 'system1'")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg = replace(cfg, limit=args.limit)
    if args.kinds:
        kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
        kept = [m for m in cfg.models if m.kind in kinds]
        # A cascade needs its tiers to exist. Keep referenced arms even when
        # their kind was filtered out, and put them first so the two-pass
        # construction in run() finds them in the registry.
        refs = {n for m in kept if m.kind == "cascade"
                for n in (m.extra.get("fast"), m.extra.get("slow")) if n}
        by_name = {m.name: m for m in cfg.models}
        kept_names = {m.name for m in kept}
        kept = [by_name[n] for n in refs if n in by_name and n not in kept_names] + kept
        missing = refs - set(by_name)
        if missing:
            print(f"--kinds kept a cascade whose tiers are not configured: {sorted(missing)}")
            return 2
        cfg = replace(cfg, models=tuple(kept))
        if not cfg.models:
            print(f"no arms left after --kinds {args.kinds}")
            return 2

    if args.dry_run:
        stub = ModelSpec(name="stub", kind="stub")
        # A dry run must never land in a measured run's directory: a stub AUROC
        # of 1.000 in a results table is worse than no result, because it reads
        # as real. Force a separate id, and refuse if it still collides.
        dry_id = cfg.run_id if cfg.run_id.endswith("-dry") else f"{cfg.run_id}-dry"
        cfg = replace(cfg, models=(stub,), dataset="fixtures", run_id=dry_id,
                      data_path=REPO / "data")
        if not (cfg.data_path / "fixtures" / "test" / "data.json").exists():
            import subprocess

            subprocess.run([sys.executable, str(REPO / "scripts" / "make_fixtures.py")],
                           check=True)
        run(cfg, live=False)
    else:
        run(cfg, live=True, cold=args.cold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
