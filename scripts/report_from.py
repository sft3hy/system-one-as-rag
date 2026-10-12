"""Regenerate report.json / report.md from an existing run directory,
without re-running any model.

A long run must never be lost to a report bug. This recomputes every number
from the frozen raw + per-split prediction JSONL using the same metric and
threshold code the runner uses, so a regenerated report equals a fresh one —
including the val-tuned threshold, which is why val predictions are persisted.

Run: uv run python scripts/report_from.py results/smoke-clef-flash
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.contract import DEFAULT_QUESTIONS  # noqa: E402
from sufficiency_lab.metrics import (  # noqa: E402
    discriminate,
    grouped_bootstrap_ci,
    latency_summary,
    operating_point,
    risk_coverage_curve,
    stratified_auroc,
)
from sufficiency_lab.runner import render_markdown  # noqa: E402
from sufficiency_lab.schema import Case, read_jsonl  # noqa: E402
from sufficiency_lab.thresholds import select_threshold  # noqa: E402

SPLITS = ("train", "val", "test")
MIN_NEG = 40


def parse_name(stem: str) -> tuple[str, str | None]:
    """`clef-flash-choice-3q_test` -> (strategy, 'test'); legacy -> (name, None)."""
    for s in SPLITS:
        if stem.endswith(f"_{s}"):
            return stem[: -(len(s) + 1)], s
    return stem, None


def load_jsonl_dir(d: Path, pattern: str = "*.jsonl") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for path in sorted(d.glob(pattern)):
        out[path.stem] = list(read_jsonl(path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--risk", type=float, default=0.05)
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    gold: dict[str, Case] = {}
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        for raw in read_jsonl(path):
            c = Case.from_json_dict(raw)
            gold[c.id] = c
    if not gold:
        print(f"no raw/*.jsonl in {run_dir} — nothing to report")
        return 1

    # group predictions by strategy, keeping split separation intact
    arms: dict[str, dict[str, dict[str, dict]]] = {}
    for stem, rows in load_jsonl_dir(run_dir / "predictions").items():
        strategy, split = parse_name(stem)
        if split is None:
            split = "test"  # legacy single-file layout
        arms.setdefault(strategy, {})[split] = {
            r["case_id"]: r for r in rows if gold.get(r["case_id"])
        }
    if not arms:
        print(f"no predictions in {run_dir}")
        return 1

    report_split = "test" if any("test" in v for v in arms.values()) else \
        next(iter(next(iter(arms.values()))))
    tune_split = "val" if any("val" in v for v in arms.values()) else None

    leaderboard: list[dict] = []
    thresholds: dict[str, dict] = {}
    per_strategy: dict[str, dict[str, float]] = {}

    for strategy, by_split in sorted(arms.items()):
        rep = by_split.get(report_split) or {}
        if not rep:
            continue
        scores = {cid: float(r["p_sufficient"]) for cid, r in rep.items()
                  if r.get("p_sufficient") == r.get("p_sufficient")}
        if not scores:
            continue
        per_strategy[strategy] = scores
        use = [gold[cid] for cid in sorted(scores)]
        all_scores = [scores[c.id] for c in use]

        th = None
        tune_map = by_split.get(tune_split or "") or {}
        if tune_map:
            tune_scores = {cid: float(r["p_sufficient"]) for cid, r in tune_map.items()
                           if r.get("p_sufficient") == r.get("p_sufficient")}
            tune_cases = [gold[cid] for cid in sorted(tune_scores) if cid in gold]
            if sum(1 for c in tune_cases if not c.sufficient) >= MIN_NEG:
                th = select_threshold(tune_cases, [tune_scores[c.id] for c in tune_cases],
                                      strategy=strategy, max_risk=a.risk)
                thresholds[strategy] = asdict(th)
        thr = th.threshold if th else 0.50

        disc = discriminate(use, all_scores)
        op = operating_point(use, all_scores, thr)
        boot = grouped_bootstrap_ci(use, all_scores, metric="auroc",
                                     reps=max(a.bootstrap // 4, 100))
        lat = latency_summary([rep[c.id]["latency_ms"] for c in use])

        res_path = run_dir / "resources" / f"{strategy}.json"
        peak_rss = (json.loads(res_path.read_text()).get("peak_server_rss_gb")
                    if res_path.exists() else None)
        lat_path = run_dir / "latency" / f"{strategy}.json"
        conc = json.loads(lat_path.read_text()).get("c8", {}) if lat_path.exists() else {}

        rows = list(rep.values())
        toks = [r.get("input_tokens") or 0 for r in rows]
        kind = "heuristic" if not any(toks) else "system1"

        leaderboard.append({
            "strategy": strategy, "kind": kind, "quantization": None,
            "gate": "fast", "n": disc["n"],
            "auroc": disc["auroc"], "auprc": disc["auprc"], "ece": disc["ece"],
            "brier": disc["brier"], "threshold": thr,
            "threshold_source": "val" if th else "default_0.5",
            "risk_fpr": op.risk_fpr, "coverage": op.coverage,
            "false_insufficient_rate": op.false_insufficient_rate,
            "p50_ms": lat.get("p50_ms"), "p95_ms": lat.get("p95_ms"),
            "p95_c8_ms": conc.get("p95_ms"),
            "qps_c8": conc.get("throughput_qps"),
            "peak_rss_gb": peak_rss,
            "tokens_per_case": sum(toks) / max(len(toks), 1),
            "errors": sum(1 for r in rows if r.get("error")),
            "bootstrap_auroc": boot,
            "stratified_quadrant": stratified_auroc(use, all_scores, "quadrant"),
            "stratified_qtype": stratified_auroc(use, all_scores, "question_type"),
            "risk_curve": [asdict(p) for p in risk_coverage_curve(
                use, all_scores, [round(0.01 * i, 2) for i in range(1, 100, 2)])],
        })

    paired: dict[str, dict] = {}
    names = sorted(per_strategy)
    if len(names) > 1:
        common = set.intersection(*[set(v) for v in per_strategy.values()])
        shared = [gold[cid] for cid in sorted(common)]
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                paired[f"{x}__vs__{y}"] = grouped_bootstrap_ci(
                    shared, [per_strategy[x][c.id] for c in shared],
                    [per_strategy[y][c.id] for c in shared],
                    metric="auroc", reps=max(a.bootstrap // 4, 100))

    report = {
        "run_id": run_dir.name,
        "dataset": next(iter(gold.values())).source,
        "contract_version": DEFAULT_QUESTIONS.version,
        "risk_target": a.risk,
        "counts": {"gold": len(gold), "report_split": report_split,
                   "tune_split": tune_split or "none"},
        "leaderboard": sorted(leaderboard, key=lambda r: -(r["auroc"] or 0)),
        "thresholds": thresholds,
        "paired_comparisons": paired,
        "regenerated_from": "predictions + raw snapshots",
    }
    (run_dir / "reports").mkdir(exist_ok=True)
    (run_dir / "reports" / "report.json").write_text(json.dumps(report, indent=2))
    (run_dir / "reports" / "report.md").write_text(render_markdown(report))
    print(f"wrote {run_dir / 'reports' / 'report.md'}\n")
    for r in report["leaderboard"]:
        ci = r["bootstrap_auroc"]
        print(f"  {r['strategy']:<24} AUROC={r['auroc']:.3f} "
              f"[{ci['lo']:.3f},{ci['hi']:.3f}] thr={r['threshold']:.2f}"
              f"({r['threshold_source']}) FPR={r['risk_fpr']:.3f} "
              f"cov={r['coverage']:.3f} p50={r['p50_ms']:.0f}ms "
              f"p95={r['p95_ms']:.0f}ms n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
