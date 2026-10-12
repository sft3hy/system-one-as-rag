"""Stability + batching-interference probe.

Two anomalies from the first live contact with clef-flash-oQ4e:

1. `noul` and `choice` on the SAME state give opposite verdicts
   (noul sufficient=0.93 while choice diagnosis='missing' is plausible), and
   `noul` returns the identical 0.932 for a sufficient and an insufficient
   context, and 0.96 for a control question about cities. That pattern says the
   noul head is not conditioning on the instruction text.

2. p_sufficient moved between runs of the identical request
   (0.9019 in one probe, 0.932 in another). If a fixed input is not
   reproducible, no threshold can be tuned on it.

So, before any benchmark number is trusted:
  A. repeat one identical request N times -> intra-config spread
  B. same question asked alone vs packed with 2 others -> interference
  C. choice-only contract vs noul-only contract -> which primitive binds

If A's spread is comparable to the sufficient/insufficient gap, the benchmark
must average over repeats (report mean p and its SE) instead of one pass.

Run: uv run python scripts/probe_stability.py [--reps 12]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.config import OmlxConfig  # noqa: E402
from sufficiency_lab.contract import (  # noqa: E402
    CONFLICTING_INSTRUCTIONS,
    DIAGNOSTIC_CHOICES,
    SUFFICIENT_INSTRUCTIONS,
)
from sufficiency_lab.omlx_client import OmlxClient  # noqa: E402

Q = "Which city hosted the conference?"
SU = ("The conference was held in Denver in 2025, drawing four thousand attendees, "
      "and the next edition was awarded to Lisbon.")
IN = ("The organisers met in Vienna in 2019 to agree the criteria for selecting "
      "future venues, and debated budget caps for three days before adjourning.")

Noul = lambda instr: {"type": "noul", "instructions": instr}  # noqa: E731


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clef-flash-oQ4e")
    ap.add_argument("--reps", type=int, default=12)
    a = ap.parse_args()
    client = OmlxClient(OmlxConfig())

    st_suf = {"question": Q, "retrieved_context": SU}
    st_ins = {"question": Q, "retrieved_context": IN}

    print("A. reproducibility: identical request, N passes\n")
    print(f"{'arm':<34} {'mean':>7} {'sd':>7} {'min':>7} {'max':>7} {'spread':>8}")
    print("-" * 76)
    for label, state, q in (
        ("noul(sufficient) | sufficient ctx", st_suf, Noul(SUFFICIENT_INSTRUCTIONS)),
        ("noul(sufficient) | insufficient ctx", st_ins, Noul(SUFFICIENT_INSTRUCTIONS)),
        ("choice(evidence) | sufficient ctx", st_suf,
         {"type": "choice", "instructions": "What is the evidence state?",
          "criteria": DIAGNOSTIC_CHOICES}),
        ("choice(evidence) | insufficient ctx", st_ins,
         {"type": "choice", "instructions": "What is the evidence state?",
          "criteria": DIAGNOSTIC_CHOICES}),
    ):
        vals, lat = [], []
        for _ in range(a.reps):
            r = client.systemone(a.model, state, {"q": q})
            ans = r.answers["q"]
            v = ans.noul if ans.noul is not None else (ans.probabilities or {}).get("answerable")
            if v is not None:
                vals.append(float(v))
            lat.append(r.latency_ms)
        if not vals:
            print(f"{label:<34} no values")
            continue
        print(f"{label:<34} {statistics.fmean(vals):>7.3f} "
              f"{(statistics.pstdev(vals) if len(vals) > 1 else 0):>7.4f} "
              f"{min(vals):>7.3f} {max(vals):>7.3f} "
              f"{max(vals) - min(vals):>8.3f}   p50={statistics.median(lat):.0f}ms")

    print("\nB. interference: same question alone vs packed with two others\n")
    one = {"sufficient": Noul(SUFFICIENT_INSTRUCTIONS)}
    three = {"sufficient": Noul(SUFFICIENT_INSTRUCTIONS),
             "conflicting": Noul(CONFLICTING_INSTRUCTIONS),
             "diagnosis": {"type": "choice", "instructions": "What is the evidence state?",
                          "criteria": DIAGNOSTIC_CHOICES}}
    for label, state in (("sufficient ctx", st_suf), ("insufficient ctx", st_ins)):
        solo, packed = [], []
        for _ in range(max(a.reps // 2, 4)):
            solo.append(client.systemone(a.model, state, one).answers["sufficient"].noul)
            packed.append(client.systemone(a.model, state, three).answers["sufficient"].noul)
        s = statistics.fmean([v for v in solo if v is not None])
        p = statistics.fmean([v for v in packed if v is not None])
        print(f"  {label:<18} alone={s:.3f}  packed={p:.3f}  Δ={p - s:+.3f}")

    print("\nC. noul vs choice, direction test (does EITHER separate the pair?)\n")
    for label, state in (("sufficient", st_suf), ("insufficient", st_ins)):
        rn = client.systemone(a.model, state, {"q": Noul(SUFFICIENT_INSTRUCTIONS)})
        rc = client.systemone(a.model, state,
                              {"q": {"type": "choice",
                                     "instructions": "What is the evidence state?",
                                     "criteria": DIAGNOSTIC_CHOICES}})
        probs = rc.answers["q"].probabilities or {}
        print(f"  {label:<13} noul={rn.answers['q'].noul:.3f}  "
              f"choice={rc.answers['q'].choice:<14} "
              f"p(answerable)={probs.get('answerable', 0):.3f} "
              f"p(missing_fact)={probs.get('missing_fact', 0):.3f}")
    client.close()
    print("\nVerdict rule: a usable primitive must separate the matched pair by")
    print("materially more than its own repeat-spread. Anything else is noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
