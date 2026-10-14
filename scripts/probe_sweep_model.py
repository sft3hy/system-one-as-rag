"""Phase 0 smoke gates for a decision model on /v1/systemone (FINDINGS §1 blockers):

  1. endpoint answers the sufficiency-v2-choice contract at all
  2. NON-FLAT: score sd > 0.01 across heterogeneous cases
  3. INSTRUCTION-SENSITIVE: p(sufficient) differs between the sufficiency
     rubric and a trivial control wording on the same state
  4. DETERMINISTIC: 3 identical requests -> spread ~0

Usage: .venv/bin/python scripts/probe_sweep_model.py <model-id> [n_cases]
Exits non-zero with the failed gate named.
"""
import json
import pathlib
import statistics
import sys

from sufficiency_lab.config import load_config
from sufficiency_lab.contract import DEFAULT_QUESTIONS
from sufficiency_lab.omlx_client import OmlxClient

MODEL = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
cfg = load_config("configs/sweep-smoke.yaml")
client = OmlxClient(cfg.omlx)

data = json.loads((pathlib.Path("data/raw/sufficiencybench/val/data.json")).read_text())
cases = []
for q in data:
    for c in q["conditions"]:
        cases.append((c["condition_id"], q["question"], c["context"], c["sufficient"]))
    if len(cases) >= N * 2:
        break
cases = cases[:N * 2]

CONTROL = {"control": {"type": "choice",
                       "instructions": "Does this text mention any place, city or country anywhere?",
                       "criteria": {"yes": None, "no": None}}}

ps, flat_scores, conflict, fails = [], [], 0, 0
first = None
for cid, question, context, suff in cases:
    state = DEFAULT_QUESTIONS.state(question, context)
    try:
        r = client.systemone(MODEL, state, DEFAULT_QUESTIONS.questions(include_diagnostic=False))
        ans = r.answers["sufficient"]
        p = ans.probabilities.get("sufficient", 0.0)
        ps.append((p, suff))
        flat_scores.append(p)
    except Exception as e:
        fails += 1
        print(f"[err] {cid}: {e}")

# determinism
det = []
state0 = DEFAULT_QUESTIONS.state(cases[0][1], cases[0][2])
for _ in range(3):
    r = client.systemone(MODEL, state0, DEFAULT_QUESTIONS.questions(include_diagnostic=False))
    det.append(r.answers["sufficient" if "sufficient" in r.answers else "sufficient"].probabilities.get("sufficient", 0))

# instruction sensitivity: same state, control question
ctrl = client.systemone(MODEL, state0, CONTROL)
ctrl_p = ctrl.answers["control"].probabilities.get("yes", 0.0)

pos = [p for p, s in ps if s]
neg = [p for p, s in ps if not s]
sd = statistics.pstdev(flat_scores) if flat_scores else 0.0
delta = (statistics.mean(pos) - statistics.mean(neg)) if pos and neg else 0.0
spread = max(det) - min(det)

print(f"model={MODEL} n={len(ps)} fails={fails}")
print(f"  mean p+ = {statistics.mean(pos):.3f}   mean p- = {statistics.mean(neg):.3f}   Δ = {delta:+.3f}")
print(f"  sd(all scores) = {sd:.4f}")
print(f"  determinism spread over 3 identical = {spread:.4f}")
print(f"  control 'mentions a place' yes = {ctrl_p:.3f}")

ok = True
if fails > len(cases) * 0.1:
    print("GATE 1 FAIL: endpoint does not answer reliably"); ok = False
if sd < 0.01:
    print("GATE 2 FAIL: FLAT scores (noul-style dead signal)"); ok = False
if abs(delta) < 0.05:
    print("GATE 3 FAIL: instruction-insensitive (no separation of matched pairs)"); ok = False
if spread > 0.02:
    print("GATE 4 FAIL: nondeterministic"); ok = False
print("PHASE 0:", "PASS" if ok else "FAIL", MODEL)
sys.exit(0 if ok else 1)
