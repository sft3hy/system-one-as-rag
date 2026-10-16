# PLAN: Purpose-trained sufficiency probe ("Clef-at-home")

Date: 2026-10-15. Status: approved-as-next-major-lever. Owner: whoever has the GPU afternoon.

## Why this, why now

Everything buyable has been bought and scored (§14 sweep, §15 ensembles, §16 tail
calibration). The map now says:

1. **Coverage is the locked value** — certified champion auto-answers 23.8% of traffic;
   roughly half of what it blocks was answerable (timidity analysis, REPORT.md).
2. **Off-the-shelf is exhausted:** the whole OpenJev family (incl. the certified
   Flash-9B fast-tier candidate) is CC-BY-NC — benchmark-only, confirmed on HF
   2026-10-15. Apache-tagged re-uploads (ZefanCai, APUS, etc.) are license-washing
   risk; unusable until the original authors themselves relicense.
3. **Self-calibration cannot fix the tail** (§16, negative result): purity-band vetoes
   tuned on val fail on test for *all ten arms*, champion included.
4. **Cross-model agreement can fix it** (§15, positive result): judge+OpenJev `prob_min`
   certifies at 0.885 AUROC. The fix lives in *second opinions*, which means the probe
   should not be one score — it should carry its own veto axis.

Conclusion: the only remaining path to big coverage is a probe **we train and own**,
built to our sufficiency definition, with a veto head, distilled from the certified
teacher pair.

## Design

- **Base:** DeBERTa-v3-base cross-encoder (~0.4 GB params) or ModernBERT-base; sequence
  pair `(question | retrieved context)` → two heads:
  - `p_sufficient` (calibrated via temperature on val),
  - `veto` (binary: overconfidence-trap detector — trained on the union of judge and
    OpenJev leak cases plus mined boundary cases; this head *only ever blocks*,
    never approves, so it can only tighten the certified teacher's operating point).
- **Teacher/students:** soft labels from clef-27B (certified) on unlabeled traffic;
  hard labels where clef-27B and OpenJev **agree** (§15 showed agreement ≈ safety).
- **Data:**
  1. SufficiencyBench train split if it ships one; otherwise stratified 80/20 carve of
     val *before* any threshold touches it (the current val is consumed — a fresh
     threshold set must be carved from test-time-fresh traffic later).
  2. ≥50k unlabeled (q, ctx) pairs from production shape → teacher-scored (≈2.5 GPU-hours
     at clef-27B's 2 QPS overnight, or flash at 6 QPS in an evening).
  3. Boundary mining: all blocked-but-answerable cases identified in prior runs
     (§15 leak/coverage tables) — the probe's hardest, most valuable examples.
  4. Contradiction pairs (stage-4 backlog) feed the veto head later.
- **Training:** 3 epochs, AdamW, class-balanced BCE + distillation KL; MPS on the Studio
     is plausible for base-size; otherwise CPU cluster overnight. Checkpoint every 200 steps.

## Certification protocol (non-negotiable, inherited)

- Tune thresholds + temperature on val only (max_risk 0.05, grid as per `thresholds.py`).
- Test reported **once**, seed 13, bootstrap 2,000 reps grouped by question.
- Beat bars to matter: (a) certify FPR ≤ 0.05; (b) coverage > 0.238 (champion); (c) p50
  < 150 ms (beat clef-flash — the prize is champion-quality at intern-speed).
- The veto head's standalone budget contribution must be evaluable: report coverage lost
  vs leaks caught, at its tuned point.

## Success gates and kill criteria

- **Ship bar:** certifies + coverage ≥ 0.30 at ≤ 150 ms → fast tier replaced, 27B demoted
  to escalated-only referee.
- **Continue bar:** certifies, coverage 0.24–0.30 → keep iterating with more distilled data.
- **Kill bar:** after 3 rounds of data expansion, no certified config → coverage gains
  require better retrieval, not better gating; redirect effort upstream.

## Risks

- Inheriting teacher blind spots (teacher-leak Jaccard 0.50 caps the student's ceiling):
  mitigated by the veto head trained on *union* leaks (both teachers' mistakes).
- Label drift once production traffic differs from SufficiencyBench: monthly retrain cadence.
- `results/` stays out of git; checkpoints live in `~/models/probe/` and DVC/LFS later.

## Sequencing

1. Carve fresh train/val from unlabeled pool + score with teachers (one overnight run).
2. Train base probe, certify (protocol above) — week 1.
3. Add veto head from mined boundary/leak set — week 2.
4. If ship bar met: shadow-mode deploy beside clef-27B for two weeks, compare live FPR.
