# Sufficiency Lab

Benchmarks whether **retrieved context is sufficient to answer a question**, comparing
System 1 decision models against autoregressive judges and classifier baselines on
identical inputs. Primary question: *which arm is the best production gate*, where
"best" means lowest false-sufficient rate at a fixed risk budget, per unit of latency
and memory.

Risk budget: **5% false-sufficient** (`risk.max_false_sufficient`). Thresholds are
tuned on `val` only; `test` is never seen before reporting.

## Why the framing matters

A gate can look excellent at sufficiency simply because it remembers the answer.
SufficiencyBench controls for this: each question ships a sufficient context and an
insufficient twin built by deleting the answering sentence and substituting a
topically similar one from the **same document** (mean length ratio 0.991), and labels
each with the model's closed-book confidence. That is why:

- `group_id` is the question id everywhere — bootstraps resample questions, never rows
- `question_only` and `context_length` arms exist as confound controls; if a decision
  model cannot beat them, it is not reading evidence
- every arm receives byte-identical `question` + `context` text

## Setup

```bash
uv sync

# dataset (19 MB, public)
git clone --depth 1 https://github.com/inab-lab/SufficiencyBench /tmp/sb
mkdir -p data/raw
cp -r /tmp/sb/datasets_for_github/sufficiencybench data/raw/sufficiencybench

# decision models (needs omlx >= 0.7.1.dev1, jundot/omlx#4315)
uv run hf download Jundot/clef-flash-oQ4e --local-dir ~/models/Jundot/clef-flash-oQ4e
uv run hf download Jundot/clef-oQ4e       --local-dir ~/models/Jundot/clef-oQ4e
```

omlx auto-detects Clef from `joint_head.safetensors` + `joint_head_config.json`.
Confirm before running anything:

```bash
uv run python scripts/probe_server.py
```

That script is the contract check: it lists `model_type == "decision"`, exercises the
two-noul contract, measures 1-vs-3-questions-in-one-request cost, and dumps the raw
response so the adapter is written against bytes rather than prose.

## Run

```bash
uv run sufficiency-lab -c configs/sufficiencybench.yaml --dry-run   # offline, no server
uv run sufficiency-lab -c configs/smoke.yaml                         # one model, real data
uv run sufficiency-lab -c configs/compare.yaml --cold                # 9B vs 27B + cold load
uv run sufficiency-lab -c configs/timing.yaml                        # cost calibration
uv run sufficiency-lab -c configs/cascade.yaml                       # production two-tier shape
uv run pytest                                                        # harness correctness
```

`--dry-run` scores a deterministic stub on synthetic fixtures with the real metrics and
report path, so schema regressions surface without a GPU. It namespaces its own run id
and refuses to write into a directory holding measured predictions.

Probes (run these before trusting any number):

```bash
uv run python scripts/probe_server.py --model clef-flash-oQ4e     # endpoint + contract
uv run python scripts/probe_instructions.py                        # does noul bind?
uv run python scripts/probe_stability.py --reps 12                 # reproducibility + interference
uv run python scripts/ab_judge.py --n 24                           # is chain-of-thought worth it?
uv run python scripts/estimate_full_run.py results/timing-100 --judge-share 0.25
uv run python scripts/make_figures.py results/<run_id>             # risk/coverage plots
uv run python scripts/report_from.py results/<run_id>              # rebuild report, no re-run
```

## The three serving shapes

| shape | config | what it costs | test cov @ 5% | verdict |
|---|---|---|---|---|
| Flash alone | `smoke.yaml` | 165 ms p50, ~6 QPS | 0.157 | cheapest, lowest coverage |
| **27B alone** | `compare.yaml` | 535 ms p50, ~1.9 QPS | **0.230** | **recommended** — best AUROC + calibration |
| Cascade | `cascade.yaml` | 165 ms + judge on ~22% band | 0.180 | affordable, but trails 27B |

A cascade answers outside `[escalate_low, escalate_high]` at the fast tier and sends
only the mid-confidence band to the judge. That mixes two score distributions —
Clef spreads over (0, 0.94), the no-think judge answers on ~{0.0, 0.1, 0.95,
1.0} — so **the judge must be calibrated onto flash's scale before the threshold
sees it**, or coverage collapses (FINDINGS §11). Fit it, then point the cascade
at it:

```bash
uv run python scripts/fit_cascade_calibration.py results/<run>   # -> calibration/<run>.json
uv run sufficiency-lab -c configs/cascade.yaml                   # loads it via extra.calibration
```

Calibration raises cascade test coverage 0.023 → 0.180 and its ECE to the best of
any arm (0.107), but it still trails 27B — the judge only earns its escalations at
its single `1.0` spike. `tier` is recorded per case regardless, so any downstream
analysis can stratify by the tier that produced the number.

## The two-tier contract

One rubric (`contract.py`, versioned) is shared by every arm. Two independent `noul`
questions ride in a **single** `/v1/systemone` request:

| question | meaning |
|---|---|
| `sufficient` | every fact needed to answer appears in the context |
| `conflicting` | the context states incompatible things about the answer |

```
p_conflicting >= t_conflict                       -> ABSTAINED_CONTRADICTION
p_sufficient  >= escalate_high                  -> ANSWERED
p_sufficient  <  escalate_low                   -> ABSTAINED_INSUFFICIENT_CONTEXT
inside the band, under fast_budget_ms           -> escalate to judge
inside the band, over  fast_budget_ms           -> resolve at fast tier (policy)
```

Two noul questions instead of one 3-way choice, so "missing fact" and "contradiction"
do not compete for probability mass, and so each threshold can be tuned independently.
`batch_questions: false` runs the same two questions as separate requests; the runner
reports the Δp and the latency ratio, which is the load-bearing assumption of the whole
design and gets measured rather than cited.

## Outputs

```
results/<run_id>/
├── raw/            frozen input snapshot (reproducibility)
├── predictions/  per-case scores, latency, tokens, errors
├── resources/    peak server RSS, CPU, min free memory, cold load
├── latency/      serial + concurrent (1/4/8) percentiles, QPS
├── reports/      report.json + report.md (leaderboard, risk curve, paired ΔAUROC)
└── figures/
```

## What the numbers mean

| metric | question it answers |
|---|---|
| `risk_fpr` | of truly insufficient contexts, how many did the gate authorise? **the cap** |
| `coverage` | how much traffic gets answered at that risk |
| `false_insufficient_rate` | how often it blocks an answerable question (cost of cowardice) |
| AUROC / AUPRC | threshold-free discrimination; AUPRC matters because classes are imbalanced |
| ECE / Brier | is the probability trustworthy enough to threshold, at this quantization |
| p50 / p95 | decision latency, client-measured, full path |
| peak RSS | does the arm fit next to the chat model already resident |

Paired ΔAUROC with grouped bootstrap CIs is the number that settles head-to-head
claims; unpaired CIs on the same 2,224 conditions overlap enough to prove nothing.

## Measured on this machine (see FINDINGS.md for the full account)

| arm | AUROC | thr | FPR | coverage | p50 |
|---|---|---|---|---|---|
| clef-oQ4e 27B | **0.879** | 0.94 | 0.047 | **0.223** | 535 ms |
| clef-flash-oQ4e | 0.828 | 0.89 | 0.033 | 0.158 | **165 ms** |
| lexical floor | 0.663 | 0.84 | 0.117 | 0.192 | ~0 |
| context_length | 0.508 | — | — | — | ~0 |
| question_only | 0.500 | — | — | — | ~0 |

Three results that changed the design:

1. **`noul` does not bind to its instruction** — identical p (0.932 vs 0.932) for
   matched sufficient/insufficient pairs, sd 0.0000 over 12 repeats. The gate's primary
   score is therefore a `choice` probability, contract `sufficiency-v2-choice`.
2. **Coverage, not safety, is the binding constraint.** Both models need thresholds of
   0.89–0.95 to respect 5% risk, and still block 57–72% of answerable questions.
3. **The decision engine does not batch across requests.** QPS is pinned at the serial
   reciprocal (6.2 for Flash) at both 4 and 8 concurrency; extra connections only buy
   linear tail latency. omlx's batched prefill (#4361) is chat-only and does not
   transfer to `/v1/systemone`. Size the gate at ~6 QPS, not at its 165 ms p50.

## Known gaps, stated rather than papered over

1. **`t_conflict` is uncalibrated.** SufficiencyBench ships no gold contradiction
   labels, so the contradiction tier reports distributions, not accuracy. Calibrating it
   needs a labelled set (stage 4, controlled retrieval degradation).
2. **No GPU-memory counter.** macOS exposes no per-process Metal counter, so server RSS
   plus `vm_stat` pressure is the reported footprint proxy. `resources.py` records the
   attempt so a report can never imply GPU memory was measured.
3. **Quantization is a variable, not a constant.** oQ keeps Clef's joint head out of
   the quantized backbone (head stays bf16), but the backbone is 4-bit and the PR's own
   validation shows max |Δp| 0.0055 at 16k tokens. Since thresholds are read off p, every
   row carries a `quantization` column.
4. **Release-note speedups do not transfer.** Batched prefill (#4361) is a chat
   feature; decision requests do one prefill and no decode. Throughput here is measured
   at 1/4/8 concurrent, not inferred from those tables.
5. **`context_length` arm near chance is a requirement, not a nicety.** If it clears
   chance, the matched-length construction leaked and every other number is suspect.
6. **Peak RSS in the leaderboard is server-wide, not per-arm.** Arms run sequentially
   and omlx does not aggressively evict, so an arm run after a big model was loaded
   inherits its footprint. Unload the other models (`POST /v1/models/{id}/unload`)
   between arms if you need clean per-model memory. See FINDINGS §7.
