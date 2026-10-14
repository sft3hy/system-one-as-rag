# System-One as a RAG Sufficiency Gate — Test Report

**Date:** 2026-10-14 · **Machine:** Mac Studio M5 Ultra / 256 GB · **Server:** omlx 0.7.1.dev1
**Dataset:** `inab-lab/SufficiencyBench` — **FULL report set: 1,390 questions = 2,780
conditions/arm** (val 556, test 2,224), matched sufficient/insufficient pairs, length ratio 0.991
**Contract:** `sufficiency-v2-choice` · **Risk budget:** 5% false-sufficient, thresholds tuned on val only

Status: **testing is finished, and the full dataset has now been run.** `full-benchmark-v1`
completed in 78.4 min wall with **zero errors across all five arms** (13,900 predictions).
It confirms the capped-run verdict on 4–7× the sample. Unit tests pass 31/31.

---

## What we tested

| Arm | Kind | Role |
|---|---|---|
| `clef-flash-oQ4e` | system1 decision model | cheap gate |
| `clef-27b-oQ4e` | system1 decision model | strong gate |
| `Qwen3.8-Flash-Next-oQ4e-mtp` | autoregressive chat judge | escalation tier |
| `cascade` (flash → judge at ~22% escalation, calibrated) | cascade | the design bet |
| `lexical` (+ earlier `context_length`, `question_only` controls) | heuristics | confound controls |

## Headline — FULL dataset (n=2,224 test conditions, thresholds tuned on val @ 5% budget)

| shape | AUROC | coverage @5% | FPR | ECE | p50 | cost/cond |
|---|---|---|---|---|---|---|
| **clef-27b** | **0.871** | 0.238 | 0.049 | 0.093 | 534 ms | 0.55 s |
| clef-flash | 0.830 | 0.162 | 0.039 | 0.090 | **165 ms** | **0.17 s** |
| qwen38-judge | 0.826 | **0.256** | **0.055 ⚠** | 0.208 | 511 ms | 0.66 s |
| **calibrated cascade** | 0.823 | 0.187 | 0.047 | **0.081** | 174 ms | 0.31 s |
| lexical | 0.665 | 0.114 | 0.049 | 0.142 | ~0 | ~0 |

Paired grouped bootstrap (95% CI, n=2,224): 27B beats flash **+0.041 [+0.034, +0.048]** and
the judge **+0.044 [+0.038, +0.051]** — both significant and now much tighter than at n=300.
Flash-vs-judge is a **confirmed statistical tie** (+0.003 [−0.004, +0.011]). Cascade-vs-flash
lands at −0.0066 [−0.0134, +0.0002] — not significant, but no longer possibly a win.

**Verdict — unchanged and now settled at full n: deploy Clef-27B as the single-arm gate.**
Best ranking *and* calibration, FPR inside budget, no mixed-scale fragility. The calibrated
cascade remains the correct fallback if staying on Flash (best ECE of any arm, half the
wall-clock of 27B), but at full scale it buys no AUROC over flash. **New at full n: the
judge alone now exceeds the risk budget on test (FPR 0.055 > 0.05) — its top coverage is
officially unbuyable at 5% risk.**

## How each model did (full dataset)

- **Clef-27B:** the winner on every axis that matters together — AUROC 0.871, ECE 0.093,
  coverage 0.238 at 0.049 FPR. Threshold 0.93 (val) transferred cleanly to test.
- **Clef-Flash:** honest budget gate — AUROC 0.830 at 165 ms p50 (0.169 s/cond, 7.9 min
  for the whole report set). Blocks 71.5% of genuinely answerable questions at 5% risk:
  coverage, not safety, is the binding constraint.
- **Qwen3.8-Flash judge (AR):** AUROC 0.826 with the best raw coverage (0.256), but its
  calibration is twice as bad as any Clef arm (ECE 0.208 — spike-y {0, 0.1, 0.95, 1.0}
  output confirmed at scale) and its val-tuned 0.96 threshold **leaks over budget on test
  (0.055)**. Its p95 is also the worst (1,777 ms). Keep as offline auditor, not online gate.
- **Cascade (flash → calibrated judge):** escalation ~22%, plumbing clean. Best ECE of every
  arm (0.081) — calibration demonstrably works — coverage 0.187 vs flash's 0.162 (+0.025),
  at 2× flash's latency and no AUROC gain (0.823, statistically tied with flash). Correct
  safety-net design; still doesn't catch 27B (−0.048 AUROC, significant).
- **Baselines:** lexical 0.665 = the honest floor; earlier controls (`question_only` 0.500,
  `context_length` 0.508) confirmed the benchmark leaks nothing.

## What running full settled (vs capped runs)

| question | capped (n=300/split) | full (n=2,224 test) |
|---|---|---|
| judge beats flash on AUROC? | tie, CI ±0.023 | **tie, CI ±0.008 — dead tie confirmed** |
| cascade beats flash? | −0.010, CI ±0.017 | −0.0066 [+0.000, −0.013] — **cannot win** |
| 27B lead real? | +0.051, CI ±0.018 | **+0.041 [+0.034, +0.048] — real, precise** |
| judge meets 5% budget? | 0.047 val (looked fine) | **0.055 test — breaks budget** |
| timing projections | ~85 min projected | **78.4 min actual — model holds (1.09× conservative)** |

## Serving facts worth repeating

- **Throughput is pinned:** decision *and* chat endpoints hold ~6 QPS (flash) / ~2 QPS
  (27B) regardless of concurrency; omlx's batched prefill doesn't cover `/v1/systemone`.
- **Contention, not residency, is the latency lever:** concurrent 27B traffic inflates flash
  p50 4.4×. Run arms serially (harness does).
- **Billing is input-only** on the decision endpoint: cost = state length (~750 tok/case).
  Full run consumed 1.67M tokens/arm (flash & 27B), 1.93M (cascade).
- **Memory:** flash 36.5 GB, 27B 54.0 GB server peak — both fit beside resident chat models.

## Charts

All under `results/report-figures/`, regenerated by `scripts/make_report_figures.py
full-benchmark-v1` (plus `make_exec_figures.py`, `make_roc_explainer.py`) purely from
`results/` artefacts:

| file | shows |
|---|---|
| `leaderboard.png` | AUROC / coverage / ECE / p50, five arms, full data |
| `risk_coverage_final.png` | risk–coverage curves + tuned operating points vs 5% budget |
| `score_distributions.png` | judge's 4-spike output (now n=556) vs Clef continuous scores |
| `capacity_vs_latency.png` | 27B vs flash, 3q vs 1q (capacity run) |
| `concurrency_ceiling.png` | QPS pinned c1→c8; tails inflate linearly |
| `cost_per_condition.png` | cascade ≈ flash + escalation — half the wall-clock of 27B |

## Where the headroom is (next pass)

1. Coverage is still the constraint — 27B blocks 57% of answerable traffic at 5% risk.
   A purpose-trained sufficiency probe (ModernBERT / hidden-state) is the next lever.
2. Thinking-on escalated cases (option b from §11) — the only lever left for the cascade.
3. Contradiction tier needs labelled conflicts (stage 4); `t_conflict` still uncalibrated.
4. Break the ~6 QPS ceiling: client batching, second replica, or upstream decision-engine
   batching.
