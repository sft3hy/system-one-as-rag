"""Instruction-sensitivity probe: does Clef's noul actually read the question?

Triggered by a live observation: on a context that does NOT contain the answer
("organisers met in Vienna in 2019 to agree criteria") Clef-Flash-oQ4e returned
sufficient=0.90 AND conflicting=0.76 AND diagnosis='answerable'=0.86 — three
mutually inconsistent statements about the same record.

Either the model is uncalibrated for this task, or the free-text `instructions`
are not binding the way the docs imply and the head is scoring topic affinity.
This distinguishes them: the sufficiency question should flip between matched
contexts, while a control question about something else in the SAME state should
not move with the sufficiency label.

If p_sufficient is invariant to wording but tracks whether the gold answer
string appears in the context, the adapter contract is fine and the model is
doing lexical grounding. If p barely moves at all, the instructions are not
binding and the whole two-tier contract must be re-specified as a `choice`
question instead.

Run: uv run python scripts/probe_instructions.py [--model clef-flash-oQ4e]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.config import OmlxConfig  # noqa: E402
from sufficiency_lab.contract import SUFFICIENT_INSTRUCTIONS  # noqa: E402
from sufficiency_lab.omlx_client import OmlxClient  # noqa: E402

CTX_SUF = ("The conference was held in Denver in 2025, drawing four thousand "
           "attendees to the downtown convention centre, and the next edition "
           "was awarded to Lisbon.")
CTX_INSUF = ("The organisers met in Vienna in 2019 to agree the criteria for "
             "selecting future venues, and debated budget caps for three days "
             "before adjourning.")
CTX_TRUE_CONFLICT = ("The conference was held in Denver in 2025. One internal "
                     "memo states the same conference took place in Seattle in "
                     "2025, and both memos are dated after the event closed.")

Q = "Which city hosted the conference?"
GOLD_IN_CTX = {"sufficient": True, "insufficient": False, "conflict": None}

# Wording controls: same state, different question about the state.
INSTR = {
    "sufficiency(doc)": SUFFICIENT_INSTRUCTIONS,
    "answer-present": ("The exact answer string to the question appears "
                       "verbatim in the retrieved context."),
    "topic-match": ("The retrieved context is about the same subject as the "
                    "question, whatever the answer is."),
    "unrelated-control": ("The retrieved context mentions a city name "
                          "anywhere in the text."),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="clef-flash-oQ4e")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()

    client = OmlxClient(OmlxConfig())
    state_of = {
        "sufficient": {"question": Q, "retrieved_context": CTX_SUF},
        "insufficient": {"question": Q, "retrieved_context": CTX_INSUF},
        "conflict": {"question": Q, "retrieved_context": CTX_TRUE_CONFLICT},
    }

    print(f"model: {a.model}\n")
    print(f"{'instruction':<20} {'sufficient ctx':>15} {'insufficient ctx':>17} "
          f"{'true conflict':>14} {'Δ suf-insuf':>12}")
    print("-" * 82)
    rows = {}
    for label, instr in INSTR.items():
        vals = []
        for key in ("sufficient", "insufficient", "conflict"):
            lat = []
            for _ in range(a.reps):
                r = client.systemone(a.model, state_of[key],
                                     {"q": {"type": "noul", "instructions": instr}})
                lat.append(r.answers["q"].noul)
            vals.append(statistics.median([v for v in lat if v is not None]))
        rows[label] = vals
        print(f"{label:<20} {vals[0]:>15.3f} {vals[1]:>17.3f} {vals[2]:>14.3f} "
              f"{vals[0] - vals[1]:>+12.3f}")

    print("\ninterpretation:")
    d_doc = rows["sufficiency(doc)"][0] - rows["sufficiency(doc)"][1]
    d_lex = rows["answer-present"][0] - rows["answer-present"][1]
    print(f"  sufficiency question separates matched pair by {d_doc:+.3f}")
    print(f"  'answer string present' separates by          {d_lex:+.3f}")
    print("  - if both are large, the model IS grounding on the question and the")
    print("    earlier 0.90-on-insufficient was this fixture being too easy/short.")
    print("  - if sufficiency is small and answer-present is large, the head is")
    print("    lexical and 'sufficient' instructions are not binding.")
    ctrl = rows["unrelated-control"]
    print(f"  control (city mentioned anywhere): {ctrl[0]:.3f} / {ctrl[1]:.3f} / {ctrl[2]:.3f}")
    print("    this should be ~1 for sufficient+conflict (both name cities) and")
    print("    lower for insufficient if instructions bind at all.")

    # 3-way choice alternative, as a possible replacement contract
    print("\n--- alternative: 3-way choice instead of 2 noul ---")
    choice_q = {
        "type": "choice",
        "instructions": ("Does the retrieved context alone support a complete "
                         "correct answer?"),
        "criteria": {
            "answerable": "contains every fact needed to answer",
            "missing": "a needed fact is absent",
            "contradictory": "states incompatible things about the answer",
        },
    }
    for key in ("sufficient", "insufficient", "conflict"):
        r = client.systemone(a.model, state_of[key], {"c": choice_q})
        ans = r.answers["c"]
        print(f"  {key:<13} -> {ans.choice:<14} "
              f"{ {k: round(v, 3) for k, v in (ans.probabilities or {}).items()} }")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
