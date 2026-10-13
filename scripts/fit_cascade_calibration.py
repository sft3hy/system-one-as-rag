"""Fit cascade tier calibration from a run's val predictions.

Takes a run dir (e.g. results/cascade-flash-judge-v1), reads clef-flash and
qwen38-judge val predictions, labels them from the run's raw snapshot, and
writes calibration/<run_id>.json: two isotonic maps score -> empirical
accuracy. The live cascade loads that file, so the threshold tuner sees ONE
scale and no longer has to thread the needle between the judge's 0.95/1.0
quantization spikes (FINDINGS §11).

Fit consumes val ONLY — test is never touched here, which is what makes the
transfer numbers in tier_thresholds.py honest.

Run:
  uv run python scripts/fit_cascade_calibration.py results/cascade-flash-judge-v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.calibration import TierCalibrator  # noqa: E402
from sufficiency_lab.schema import read_jsonl  # noqa: E402

BAND = (0.35, 0.75)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--fast", default="clef-flash")
    ap.add_argument("--slow", default="qwen38-judge")
    ap.add_argument("--band", default=f"{BAND[0]},{BAND[1]}")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lo, hi = (float(x) for x in a.band.split(","))
    d = Path(a.run_dir)

    gold = {r["id"]: bool(r["sufficient"])
            for r in read_jsonl(d / "raw" / "sufficiencybench_val.jsonl")}

    def read(arm: str):
        fp = d / "predictions" / f"{arm}_val.jsonl"
        if not fp.exists():
            raise FileNotFoundError(f"{fp} — calibration needs {arm}'s val run")
        out = []
        for r in read_jsonl(fp):
            if r.get("error") or r["p_sufficient"] != r["p_sufficient"]:
                continue  # unscored cases carry no calibration information
            out.append((float(r["p_sufficient"]), int(gold[r["case_id"]])))
        return out

    fast = read(a.fast)
    slow_all = read(a.slow)
    fmap = dict(zip([r["case_id"] for r in read_jsonl(
        d / "predictions" / f"{a.fast}_val.jsonl")], [x[0] for x in fast]))

    # fast tier = decisive cases (outside the band, per flash's own score);
    # slow tier = band cases. Splitting by the ROUTING rule, not by score
    # overlap, is what makes each map describe the population that tier
    # actually serves online.
    slow_ids = [cid for cid, p in fmap.items() if lo <= p <= hi]
    fast_pts = [(p, gold[cid]) for cid, p in fmap.items()
                if not (lo <= p <= hi)]
    slow_map = {cid: p for cid, p in
                zip([r["case_id"] for r in read_jsonl(
                    d / "predictions" / f"{a.slow}_val.jsonl")],
                    [x[0] for x in slow_all])}
    slow_pts = [(slow_map[cid], gold[cid]) for cid in slow_ids
                if cid in slow_map]

    cal = TierCalibrator.fit([p for p, _ in fast_pts], [y for _, y in fast_pts],
                             [p for p, _ in slow_pts], [y for _, y in slow_pts])
    # calibrators are model artifacts, not experiment output: they live in a
    # tracked dir so a live run reproducibly loads the exact mapping a report
    # was produced with. results/ is gitignored precisely because it holds
    # per-run, regenerable output.
    out = Path(a.out) if a.out else (Path(__file__).resolve().parents[1] /
                                       "calibration" / f"{d.name}.json")
    cal.save(out)
    print(f"fit fast n={cal.meta['n_fast']} slow n={cal.meta['n_slow']}")
    print("judge -> flash-equivalent ladder:")
    for v, l in cal.meta["ladder"].items():
        print(f"  judge {v}: accuracy {l['accuracy']} -> flash-equiv {l['flash_equiv']}")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
