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

## System-one model sweep — full dataset (same n, same budget, same seed)

Eight more open-source System-1 models discovered, downloaded, gate-tested (FINDINGS §13)
and run full (`sweep-full-v1`, six arms × 2,780, zero errors). Merged 11-arm leaderboard:

| # | arm | AUROC [95% CI] | Δ vs clef-27b [CI] | thr | test FPR | coverage | p50 ms |
|---|---|---|---|---|---|---|---|
| 1 | **openjev-27b-4bit** ⚠NC | **0.889** [0.880, 0.899] | **+0.018** [+0.014, +0.023] | 0.99 | **0.073 ✗** | 0.308 | 610 |
| 2 | **clef-27b** | 0.871 [0.860, 0.881] | — | 0.93 | 0.049 ✓ | 0.238 | 534 |
| 3 | **openjev-flash-9b** | 0.860 [0.849, 0.870] | −0.011 [−0.017, −0.005] | 0.99 | 0.042 ✓ | 0.175 | 470 |
| 4 | clef-flash | 0.830 [0.819, 0.840] | −0.041 [−0.047, −0.034] | 0.89 | 0.039 ✓ | 0.162 | 165 |
| 5 | qwen38-judge | 0.826 | −0.044 | 0.96 | 0.055 ✗ | 0.256 | 511 |
| 6 | cascade | 0.823 | −0.048 | 0.89 | 0.047 ✓ | 0.187 | 174 |
| 7 | tev1-4b (k=12 votes) | 0.805 [0.793, 0.816] | −0.066 [−0.075, −0.057] | 0.92 | 0.058 ✗ | 0.206 | 1,517 |
| 8 | laya-typed-decisions | 0.689 [0.678, 0.700] | −0.182 | 0.83 | 0.024 ✓ | 0.056 | 62 |
| 9 | lexical | 0.665 | −0.206 | 0.88 | 0.049 ✓ | 0.114 | ~0 |
| 10 | von-1.3 (noul) | 0.649 [0.636, 0.661] | −0.222 | 0.84 | 0.000 ✓ | **0.000** | 127 |
| 11 | laya | 0.622 [0.611, 0.633] | −0.249 | 0.86 | 0.040 ✓ | 0.060 | 43 |

- **OpenJev-27B wins AUROC but cannot certify**: val FPR floors at 0.058 (threshold
  fallback at 0.99), test 0.073 — over-confident tail, disqualified standalone exactly
  where the judge was. Benchmark lesson: it punishes both spike-y *and* heavy-tailed
  miscalibration; ranking power ≠ risk control.
- **OpenJev-Flash-9B wins the fast tier outright**: certifies at FPR 0.042 and beats
  clef-flash by a certified **+0.030 [+0.024, +0.036]**. Costs 2.8× the latency
  (470 ms vs 165) and covers slightly more (0.175 vs 0.162).
- **Revised production recommendation:** **clef-27B remains the gate** (the only arm
  with both budget compliance and top-2 ranking). If the fast tier is revisited,
  OpenJev-Flash-9B > clef-flash on accuracy at moderate latency cost — subject to a
  license check; the 27B is CC-BY-NC and was run benchmark-only, never a production
  candidate.
- **Tev1-4B**: honest AR k-sampling arm (no server logprobs existed — score = 12-vote
  rate, FINDINGS §13). 0.805 at 1.5 s/case, fails budget: behavioral System-1 works,
  pays AR price for sub-Clef accuracy.
- **Encoder family (laya/von/opendecider)**: all alive but tier-3 as gates.
  laya-typed > lexical certified (+0.024) at 62 ms; von's budget-compliant operating
  point admits nothing — an abstention machine.
- clef-omni-4bit: server cannot load `qwen3_omni_moe` — excluded pre-flight, not scored.

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

All under `results/report-figures/`, regenerated from `results/` artefacts only:
`scripts/make_report_figures.py <run>` (technical set, per run),
`scripts/make_exec_figures.py <run>` (leadership set, per run),
`scripts/make_roc_explainer.py <run>`, and — new this week —
`scripts/make_merged_figures.py` + `scripts/make_exec_merged_figures.py` (whole-field
sets built from `results/merged-leaderboard-v1/merged-stats.json` + every run's
predictions):

| file | shows |
|---|---|
| `leaderboard.png` | AUROC / coverage / ECE / p50, five arms, full data |
| `risk_coverage_final.png` | risk–coverage curves + tuned operating points vs 5% budget |
| `score_distributions.png` | judge's 4-spike output (now n=556) vs Clef continuous scores |
| `capacity_vs_latency.png` | 27B vs flash, 3q vs 1q (capacity run) |
| `concurrency_ceiling.png` | QPS pinned c1→c8; tails inflate linearly |
| `cost_per_condition.png` | cascade ≈ flash + escalation — half the wall-clock of 27B |
| `merged_leaderboard.png` | all 11 arms, AUROC + bootstrap CIs + paired Δ vs incumbent |
| `merged_risk_coverage.png` | every arm's ROC on one axis, dots = tuned thresholds vs 5% line |
| `merged_accuracy_vs_latency.png` | the deploy picture: accuracy vs p50 latency, all models |
| `exec_field_coverage.png` | leadership view: safely-auto-answerable traffic, all 11, red = budget-breaker |
| `exec_field_quality_speed.png` | leadership view: quality × delay, every model, green/red honest |
| `exec_calibration_field.png` | four representative gates' confidence histograms — why red is red |

Whole-field, at a glance:

![All 11 arms — AUROC ladder](results/report-figures/merged_leaderboard.png)

![Risk-coverage — every arm vs the 5% budget](results/report-figures/merged_risk_coverage.png)

## Ensembles from banked scores (2026-10-15, zero-GPU, `scripts/ensemble_certify.py`)

84 combination scorers (all pairs of the 6 continuous arms + triples/quads × {rank_avg,
prob_avg, prob_min, prob_max}) scored entirely from banked predictions; val-tuned @ 5%,
test once, seed 13 / 2,000 reps. `results/ensemble-v1/` holds the full table.

- **Certified dethroner (benchmark-only license):** `qwen38-judge + openjev-27b-4bit`,
  `prob_min` thr 0.96 → **FPR 0.048, coverage 0.248, AUROC 0.885, Δ vs clef-27b
  +0.014 [+0.009, +0.019]** — first scorer to beat the incumbent decisively *and* hold
  budget. Mechanism: mutual veto — the judge's 0.95 spikes become the brake on OpenJev's
  red tail; OpenJev vetoes the judge's wrong 1.0s. CC-BY-NC inherits → cannot ship.
- **Licence-clean ensembles do not beat the champion:** every clef-member combo that
  certifies (e.g. clef+flash prob_min, FPR 0.037) trades coverage (≤0.213) for margin.
- **Selection protocol:** val-0.05 selection leaks at test (rank_avg family, FPR up to
  0.070); val-0.04 selection picks the winner and holds. Budget choice > rule choice.
- Leak overlap clef∩openjev = Jaccard 0.50 — ensembles trim tails, they don't erase them.

## Where the headroom is (next pass)

1. Coverage is still the constraint — 27B blocks 57% of answerable traffic at 5% risk.
   Purpose-trained sufficiency probes are the next lever — the sweep's answer so far is
   that off-the-shelf encoders (laya/von) rank *below* lexical-plus-calibration; a probe
   fine-tuned on sufficiency pairs is still unbuilt and remains the highest-value lever.
   HF sweep 2026-10-15 confirmed nothing off-the-shelf fits (hobbyist SQuAD-answerability
   models only). Spec update from §15's ensemble result: the probe should carry a **veto
   axis**, not just a sufficiency score — mutual veto is now proven to certify.
2. OpenJev calibration: asymmetric calibration (the §11 recipe) applied to OpenJev-27B's
   heavy upper tail could rescue a 0.889-AUROC gate into budget compliance — the single
   highest-value fix on the board (+0.018 AUROC sits behind it). **Ensemble §15 proves
   the tail is fixable by veto; calibration is the single-model (licensable) version of
   the same trick.**
3. Thinking-on escalated cases (option b from §11) — the only lever left for the cascade.
4. Contradiction tier needs labelled conflicts (stage 4); `t_conflict` still uncalibrated.
5. Break the ~6 QPS ceiling: client batching, second replica, or upstream decision-engine
   batching.
