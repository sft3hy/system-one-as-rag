"""Extrapolate full-run cost from a calibration run.

Everything derives from measured wall-clock actually observed on this hardware:
per-arm summed request latency (the honest serial proxy) plus the throughput
plateau measured at 4 and 8 concurrent connections. Nothing here is projected
from vendor latency claims — the reason omlx's batched-prefill numbers are
excluded is that they do not transfer to /v1/systemone (measured: QPS is flat
at the serial reciprocal).

Run: uv run python scripts/estimate_full_run.py results/timing-100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.schema import read_jsonl  # noqa: E402

# SufficiencyBench, verified against the downloaded splits:
#   train 1390 questions / 2780 conditions
#   val    278 / 556      test 1112 / 2224
SIZES = {
    "val+test (report set)": (278 + 1112) * 2,
    "train+val+test (all)": (1390 + 278 + 1112) * 2,
}


def fmt_dur(s: float) -> str:
    if s < 90:
        return f"{s:5.1f}s"
    if s < 5400:
        return f"{s/60:5.1f}m"
    return f"{s/3600:5.1f}h"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="serving concurrency assumed for the projection")
    ap.add_argument("--judge-share", type=float, default=0.0,
                    help="fraction of traffic a cascade escalates to the judge")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    if not run_dir.exists():
        print(f"no such run: {run_dir}")
        return 1

    calib_groups: set[str] = set()
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        calib_groups |= {r["group_id"] for r in read_jsonl(path)}

    stats_all: dict = {}
    if (run_dir / "stats.json").exists():
        stats_all = json.loads((run_dir / "stats.json").read_text())

    # kind per arm: the leaderboard recorded it at run time. Inferring kind
    # from token counts mislabels a no-think judge as a decision model (both
    # can have small outputs), and the latency advice differs completely.
    kinds: dict[str, str] = {}
    report_path = run_dir / "reports" / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        kinds = {row["strategy"]: row.get("kind", "?")
                 for row in report.get("leaderboard", [])}

    rows = []
    for pred in sorted((run_dir / "predictions").glob("*_test.jsonl")):
        arm = pred.stem[: -len("_test")]
        conc = {}
        lat_path = run_dir / "latency" / f"{arm}.json"
        if lat_path.exists():
            conc = json.loads(lat_path.read_text())
        rows.append(_arm_row(arm, pred, conc, stats_all.get(arm, {}),
                              kinds.get(arm, "?")))

    if not rows:
        print(f"no predictions in {run_dir}")
        return 1

    print(f"calibration: {len(calib_groups)} question groups from {run_dir.name}")
    print("(sufficiency is 2 conditions per question: sufficient + insufficient)\n")
    print(f"{'arm':<22}{'kind':<11}{'p50ms':>7}{'p95ms':>7}{'s/cond':>8}"
          f"{'fail':>6}{'QPS':>7}{'val+test':>10}{'all':>8}")
    print("-" * 85)
    for r in sorted(rows, key=lambda x: -x["per_cond_s"]):
        full = SIZES["val+test (report set)"] * r["per_cond_s"]
        alls = SIZES["train+val+test (all)"] * r["per_cond_s"]
        qps = f"{r['qps']:.1f}" if r["qps"] else "—"
        if r["qps"] > 1000:
            qps = f"{r['qps']/1000:.0f}k"
        print(f"{r['arm']:<22}{r['kind']:<11}{r['p50']:>7.0f}{r['p95']:>7.0f}"
              f"{r['per_cond_s']:>8.3f}{r['fail_pct']:>5.0f}%{qps:>7}"
              f"{fmt_dur(full):>10}{fmt_dur(alls):>8}")

    by = {r["arm"]: r for r in rows}
    dec = [r for r in rows if r["kind"] == "system1"]
    judge = [r for r in rows if r["kind"] == "llm_judge"]
    full_n = SIZES["val+test (report set)"]

    print("\ntotal wall-clock for the report set (val+test, 2,780 conditions):")
    serial_all = sum(r["per_cond_s"] for r in rows if r["kind"] != "heuristic")
    print(f"  all arms, serial          {fmt_dur(serial_all * full_n)}")
    if judge:
        j = min(judge, key=lambda r: r["per_cond_s"])
        print(f"  judge alone             {fmt_dur(j['per_cond_s'] * full_n)}"
              f"   ({j['arm']})")
    if dec:
        d = max(dec, key=lambda r: r["per_cond_s"])
        print(f"  slowest decision arm    {fmt_dur(d['per_cond_s'] * full_n)}"
              f"   ({d['arm']})")

    if judge and dec and a.judge_share:
        d = min(dec, key=lambda r: r["per_cond_s"])
        j = min(judge, key=lambda r: r["per_cond_s"])
        cascade = full_n * (d["per_cond_s"] + a.judge_share * j["per_cond_s"])
        print(f"\n  cascade @ {a.judge_share:.0%} escalation  {fmt_dur(cascade)}"
              f"   ({d['arm']} + {j['arm']})")
        print(f"    vs judge alone            {fmt_dur(j['per_cond_s'] * full_n)}"
              " — the cascade pays decode only on the band,")
        print("    so judge-grade escalation comes at near-flash cost.")

    print("\nthroughput plateau (concurrency does not help /v1/systemone):")
    for r in sorted(rows, key=lambda x: -x["per_cond_s"]):
        if r["c4"] and r["c8"]:
            print(f"  {r['arm']:<22} c1 {r['qps']:.1f} qps -> c4 {r['c4']:.1f} -> "
                  f"c8 {r['c8']:.1f} qps   "
                  f"{'FLAT (no cross-request batching)' if abs(r['c8']-r['c4'])<0.15*r['c4'] else 'scales'}")

    print("\ncaveats:")
    print("  - 'fail%' is cases the arm could not score. Anything above ~1% is a")
    print("    reliability bug, not noise, and disqualifies the arm as a gate.")
    print("  - projections are serial: a single client. Add a replica per ~6 QPS")
    print("    for Clef-Flash, ~2 QPS for Clef-27B.")
    return 0


def _arm_row(arm: str, pred: Path, conc: dict, stats: dict, kind: str) -> dict:
    """Per-condition cost, preferring measured wall-clock over summed latency.

    wall_serial_s is elapsed time around a split: it includes serialization,
    retries and client overhead. Summed request latency excludes them, so it
    quietly under-projects a full run. Use it only when the run predates the
    wall-clock accounting.
    """
    lat, errs, tot = [], 0, 0
    for r in read_jsonl(pred):
        tot += 1
        if r.get("error"):
            errs += 1
        v = r.get("latency_ms")
        if v == v:
            lat.append(float(v))
    arr = np.array(lat) if lat else np.array([float("nan")])

    wall = stats.get("wall_serial_s") or {}
    n_conditions = tot
    if wall:
        wall_s = sum(float(w) for w in wall.values())
        # each split's wall covers the same number of conditions as test
        n_conditions = tot * len(wall)
    else:
        has_val = pred.with_name(pred.stem[: -len("_test")] + "_val" + pred.suffix
                                  ).exists()
        wall_s = (sum(lat) / 1000.0) * (2 if has_val else 1)
        n_conditions = tot * (2 if has_val else 1)

    return {
        "arm": arm,
        "kind": kind,
        "p50": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "per_cond_s": wall_s / max(n_conditions, 1),
        "fail_pct": 100.0 * errs / max(tot, 1),
        "qps": float(conc.get("serial", {}).get("throughput_qps_serial")
                    or conc.get("serial", {}).get("throughput_qps")
                    or 1000.0 / max(np.median(arr), 1e-6)),
        "c4": float(conc.get("c4", {}).get("throughput_qps") or 0),
        "c8": float(conc.get("c8", {}).get("throughput_qps") or 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
