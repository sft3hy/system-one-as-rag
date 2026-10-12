"""Risk/coverage figures from a completed run's predictions.

Two plots, because the two decisions you actually make are different:
  1. risk_coverage.png — at what threshold does an arm meet the 5% cap, and
     what coverage does that cost?
  2. risk_vs_latency.png — the same risk, priced in milliseconds. The arm
     that is leftmost-and-lowest is the gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.metrics import risk_coverage_curve  # noqa: E402
from sufficiency_lab.schema import Case, read_jsonl  # noqa: E402


def load(run_dir: Path) -> tuple[list[Case], dict[str, dict[str, float]]]:
    from sufficiency_lab.schema import read_jsonl

    cases: list[Case] = []
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        for raw in read_jsonl(path):
            cases.append(Case.from_json_dict(raw))
    by_case = {c.id: c for c in cases}
    scores: dict[str, dict[str, float]] = {}
    for path in sorted((run_dir / "predictions").glob("*_test.jsonl")):
        rows = {}
        for r in read_jsonl(path):
            p = r.get("p_sufficient")
            if p == p and r["case_id"] in by_case:  # drop NaN
                rows[r["case_id"]] = float(p)
        scores[path.stem[: -len("_test")]] = rows
    common = set.intersection(*[set(v) for v in scores.values()]) if scores else set()
    ordered = [by_case[cid] for cid in sorted(common)]
    return ordered, {k: {c.id: v[c.id] for c in ordered} for k, v in scores.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--risk", type=float, default=0.05)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    cases, scores = load(run_dir)
    if not cases or not scores:
        print("nothing to plot: no predictions with gold labels in", run_dir)
        return 1

    grid = [round(0.01 * i, 3) for i in range(1, 100)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    for name, per_case in sorted(scores.items()):
        pts = risk_coverage_curve(cases, [per_case[c.id] for c in cases], grid)
        cov = [p.coverage for p in pts]
        risk = [p.risk_fpr for p in pts]
        lat_path = run_dir / "latency" / f"{name}.json"
        p95 = None
        if lat_path.exists():
            p95 = json.loads(lat_path.read_text()).get("serial", {}).get("p95_ms")
        ax1.plot(risk, cov, marker="o", ms=3, lw=1.4, label=name)
        if p95 is not None:
            ax2.plot(risk, [p95] * len(risk), lw=0, label=name)
            ax2.scatter(risk, [p95] * len(risk), s=18, alpha=0.85)
        best = min(pts, key=lambda p: (abs(p.risk_fpr - a.risk), -p.coverage))
        ax1.scatter([best.risk_fpr], [best.coverage], marker="*", s=150, zorder=5,
                    edgecolors="black", linewidths=0.6)

    ax1.axvline(a.risk, ls="--", c="crimson", lw=1,
                label=f"risk budget {a.risk:.0%}")
    ax1.set_xlabel("false-sufficient rate (risk)")
    ax1.set_ylabel("coverage (share authorised)")
    ax1.set_title("Risk vs coverage — stars sit at each arm's budget")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8)

    ax2.set_xlabel("false-sufficient rate (risk)")
    ax2.set_ylabel("p95 latency (ms)")
    ax2.set_title("Same risk, priced in latency")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    fig.suptitle(f"Sufficiency gates · {run_dir.name} · {len(cases)} cases", fontsize=11)
    fig.tight_layout()
    out = run_dir / "figures" / "risk_coverage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
