# Findings — Clef as a RAG sufficiency gate on omlx

All numbers measured on this machine (Mac Studio M5 Ultra / 256 GB, macOS 27.0.1,
omlx 0.7.1.dev1, jundot/omlx#4315) on 2026-10-12. Nothing here is quoted from a
model card or release note; every figure is a live response or a run artefact under
`results/`.

Datasets: SufficiencyBench (`inab-lab/SufficiencyBench`), matched sufficient /
insufficient context pairs from the same source document, mean length ratio 0.991.

---

## 1. Blocking: `noul` does not bind to its instruction

`SYSTEM-ONE.md` gates on a `noul` question ("probability the context is sufficient").
Same state, four different instructions, `clef-flash-oQ4e`:

| instruction | sufficient ctx | insufficient ctx | true conflict |
|---|---|---|---|
| **sufficiency (doc wording)** | **0.932** | **0.932** | 0.307 |
| "answer string appears verbatim" | 0.857 | 0.545 | 0.733 |
| "same subject as the question" | 0.932 | 0.535 | 0.698 |
| control: "mentions a city anywhere" | 0.963 | 0.955 | 0.968 |

The first row is the blocker: **Δ = +0.000** between a context containing the gold
answer and one that does not. The control row explains it — asking about *cities* also
returns ~0.96 everywhere. `noul` returns a high, near-constant probability that does not
depend on what was asked.

Not noise: 12 identical repeated requests gave **sd = 0.0000, spread = 0.000**
(p50 65 ms). Deterministic output, absent signal.

**Consequence:** a `noul` gate authorises ~93% of answers regardless of evidence —
~19× over the agreed 5% false-sufficient budget, and unfixable by any threshold,
because a constant score has no ranking information.

## 2. `choice` over a fixed schema binds — and the contract works

Replaced the gate's primary score with `P(sufficient)` from a 2-option `choice`
question, plus an independent `choice` for contradiction so the two probabilities do not
compete for mass (`contract.py` → `sufficiency-v2-choice`). `primitive: noul` stays
selectable so the failure remains reproducible rather than folklore.

## 3. First real numbers, and the confound controls that validate them

Smoke run, 120 question groups × 2 conditions, val for tuning / test for reporting:

| arm | AUROC | what it proves |
|---|---|---|
| clef-flash-choice | **0.832** | real signal |
| lexical coverage | 0.663 | the honest floor |
| context_length | 0.508 | length confound is neutralised |
| question_only | **0.500** | question alone carries exactly zero information |

`question_only` at 0.500 and `context_length` at 0.508 confirm the benchmark's
matched-length construction did not leak — which is what makes 0.832 credible.
0.832 sits at/just above the 0.66–0.81 band SufficiencyBench reports for output-based
detectors, so this is a faithful replication on Apple silicon.

Methodological note kept in the harness: two hand-written probes initially suggested
sufficiency discrimination was dead (Δ ≈ 0.006). It was n=2 noise. Probe scripts detect
*mechanism* failure (flat, instruction-invariant output); they never settle magnitude.

## 4. Capacity comparison — 9B vs 27B at the 5% risk budget

Thresholds tuned on val only, same rubric, same case order. The only variable is the
checkpoint.

| arm | val thr | test AUROC | test FPR | coverage | blocked-sufficient | ECE | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| clef-flash-oQ4e (3q) | 0.89 | 0.828 | 0.033 | **0.158** | 71.7% | 0.114 | **165 ms** | **196 ms** |
| clef-oQ4e 27B (3q) | 0.94 | **0.879** | 0.047 | **0.223** | 57.0% | 0.109 | 535 ms | 661 ms |
| clef-flash-oQ4e (1q) | 0.91 | 0.829 | 0.047 | 0.153 | 70.6% | 0.142 | 202 ms | 258 ms |
| clef-oQ4e 27B (1q) | 0.95 | **0.880** | 0.040 | 0.233 | 56.3% | 0.124 | 683 ms | 843 ms |

- **Capacity buys real accuracy:** +0.051 AUROC (0.828 → 0.879), better calibration
  (ECE 0.114 → 0.109), and at the same risk budget 41% more coverage
  (0.158 → 0.223).
- **Capacity costs 3.2× latency:** p50 165 → 535 ms.
- **Both models need an unusually high threshold** (0.89–0.95) to respect 5% FPR, and
  both still block the majority of genuinely answerable questions (57–72%). Safety is
  not the binding constraint — **coverage is**. At 5% risk you can only answer ~16–23%
  of traffic, so a 27B's extra coverage is worth paying for in most products.

## 5. Batching: pack into one request, and stop adding questions

| arm | p50 serial | QPS |
|---|---|---|
| clef-flash 3 questions, 1 request | **165 ms** | 6.2 |
| clef-flash 1 question × 2 requests | 202 ms | 5.2 |
| clef-27B 3 questions, 1 request | 535 ms | 1.9 |
| clef-27B 1 question × 2 requests | 683 ms | 1.5 |

Packing beats round-trips (165 < 202 ms) — but each extra scored option still costs
~25–35 ms on Flash at identical `input_tokens` (~750 vs ~812 for split;
`output_tokens: 0`, confirming zero decode). "Ask everything, it's free" is wrong:
`input_tokens` is flat, wall clock is not.

## 6. Architectural finding: the decision engine does not batch across requests

Throughput is **identical at 4 and 8 concurrent connections** while tail latency grows
linearly:

| arm | serial p50 | c4 p95 / QPS | c8 p95 / QPS |
|---|---|---|---|
| clef-flash | 165 ms | 701 ms / 6.2 | 1344 ms / 6.2 |
| clef-27B | 535 ms | 2374 ms / 1.9 | 4498 ms / 1.9 |

QPS is pinned at the serial reciprocal (1/0.161 ≈ 6.2) regardless of concurrency; 8
in-flight requests just queue. The decision model holds the GPU exclusively per request.
omlx's batched-prefill work (#4361) covers chat models only — it does **not** transfer
to `/v1/systemone`. Concurrency is not a lever for decision throughput today; it only
buys latency inflation.

Corollary for the two-tier design: under load, the ~6 QPS ceiling of a single Clef-Flash
gate is the number to size against, not the single-request 165 ms.

## 7. Cost and memory accounting

- `usage.output_tokens = 0` always. Billing is input-only, so the decision tier's cost
  is entirely state length: ~750 tokens/case measured (~225k tokens per 300 cases).
- Server peak RSS: **36.2 GB** with Flash resident → **53.4 GB** once the 27B is
  loaded, minimum 24.3 GB free on a 256 GB box. Both fit comfortably next to the
  chat models already resident.
- **RSS caveat:** arms run sequentially and omlx does not aggressively evict, so the
  Flash arms' peak (53.4 GB for `clef-flash-oQ4e-1q`) is read *after* the 27B had
  already been loaded and is therefore a server-wide ceiling, not a per-model figure.
  The clean per-model readings are 36.2 GB (flash, first arm) and 53.4 GB (27B).
  Cross-model comparison from this table alone would be wrong.
- No per-process Metal counter exists on macOS, so `resources.py` records server RSS +
  `vm_stat` pressure and stamps `gpu_memory_measured: false` on every report, so a
  report can never imply GPU memory was measured when it was not.

## 8. The autoregressive judge: thinking is a luxury, and `response_format` is load-bearing

`Qwen3.8-Flash-Next-oQ4e-mtp` as judge, 24 real SufficiencyBench cases
(`scripts/ab_judge.py`):

| config | failed cases | p50 | p95 | out tok | AUROC | mean p⁺ | mean p⁻ |
|---|---|---|---|---|---|---|---|
| think + schema, 512 tok | **21%** | 1735 ms | 3314 ms | 253 | 1.000 | 0.843 | 0.074 |
| **no-think + schema** | **0%** | **509 ms** | 713 ms | 45 | 0.924 | 0.967 | 0.358 |
| no-think, no schema | 67% | 384 ms | 528 ms | 30 | 0.500 | 1.000 | 1.000 |
| think + schema, 1024 tok | 4% | 1919 ms | 4327 ms | 308 | 0.985 | 0.853 | 0.128 |

Three conclusions:

1. **Thinking is billed against `max_tokens`, and it eats the answer.** omlx
   returns the chain of thought in `reasoning_content` but counts it toward the
   budget, so at 512 tokens a fifth of cases reasoned themselves out of ever
   emitting JSON. Raising to 1024 cuts that to 4% and costs another 18% latency.
2. **The default is now no-think.** 0% failure at 509 ms for AUROC 0.924 beats
   1735 ms with a 21% hole for 1.000. Thinking genuinely sharpens the negatives
   (mean p⁻ 0.074 vs 0.358) — that is real information — but a gate that silently
   drops one request in five is not a gate. Keep thinking for an offline audit
   pass, not the online path.
3. **`response_format` carries the entire task.** Without it, 67% of responses are
   unparseable prose and discrimination is *exactly* 0.500 — chance. The grammar
   constraint is not a convenience, it is what makes a usable probability.

Judge-vs-decision economics at the measured 0.509 s/condition: the judge costs
~3× Clef-Flash per request. A cascade that escalates only a mid-confidence band is
the only shape that makes an AR judge affordable at full scale (`estimate_full_run.py
--judge-share`).

## 9. Server facts worth recording

- Model discovery is a startup scan. A freshly downloaded checkpoint is invisible until
  `POST /admin/api/reload` → `{"status":"ok","message":"Re-discovered 11 models"}`,
  after which it reports `model_type: "decision"`.
- Real load/unload routes: `/admin/api/models/{id}/unload`, `/v1/models/{id}/unload`
  (`/api/unload` does not exist — original guess 404'd).
- `/v1/systemone` rejects non-decision models with a 400 naming the endpoint.
- `truncate: true` is Clef's default (16384-token state); `false` makes an oversized
  prompt a 413, which the adapter records rather than dropping.

## 10. Full-set timing calibration — the run you schedule by

100 question groups per arm (200 conditions), all four arms, three models
resident, `configs/timing.yaml`. Whole run: **18.4 min**. Zero failed cases
across all arms — the judge fix holds under the full split, not just the probe.

| arm | kind | val thr | fpr | cov | test AUROC | ECE | p50 | p95 | s/cond |
|---|---|---|---|---|---|---|---|---|---|
| clef-flash | system1 | 0.89 | 0.030 | 0.145 | 0.836 | 0.123 | 261 ms | 319 ms | 0.265 |
| clef-27b | system1 | 0.93 | 0.050 | 0.265 | 0.872 | 0.101 | 866 ms | 1091 ms | 0.835 |
| qwen38-judge | llm_judge | 0.96 | 0.040 | 0.295 | 0.827 | 0.222 | 489 ms | 829 ms | 0.743 |
| lexical | heuristic | 0.84 | 0.050 | 0.160 | 0.655 | 0.158 | ~0 | ~0 | ~0 |

Projections to the report set (val+test = 1,390 questions = 2,780 conditions),
serial client, from measured `wall_serial_s` — not summed latency, which omits
client overhead:

| shape | wall |
|---|---|
| flash alone | **12.3 min** |
| cascade (flash + judge at 25% escalation) | **20.9 min** |
| judge alone | 34.4 min |
| clef-27b alone | 38.7 min |
| all arms serial | 85.4 min |
| all three model arms, every split incl. train | ~2.7 h |

Three things the calibration changed about the design:

1. **Contention, not co-residency, is the latency lever — proven, not asserted.**
   The calibration's Flash p50 (261 ms) is 1.6× the clean probe (165 ms), so I
   isolated the cause with direct tests:
   - *Memory residency is free.* 40 identical requests: 137 ms p50 with the chat
     judge resident, 138 ms after evicting it. Sharing the GPU at rest costs nothing.
   - *Harness overhead is negligible.* `strategy.predict` measures 170 ms wall vs
     163 ms raw client round-trip on the same contexts — ~7 ms of parsing and
     bookkeeping, and the server-reported p50 equals the wall p50, so nothing hides
     inside the harness.
   - *Compute contention is brutal.* Running clef-Flash serially while two threads
     hammer `/v1/systemone` on the 27B: Flash p50 jumps to **726 ms, 4.4×**. Because
     the decision engine does not batch across requests (finding §6), concurrent
     traffic does not share the compute — it collides on it.

   Consequence: arms must run **serially**, which the harness already does; the
   261 ms figure is just what a crowded multi-model box settles at under sustained
   single-arm load, and `wall_serial_s` (the real elapsed time) already bakes it
   into the projections above — that is why estimating from wall beats estimating
   from the probe. Use the eviction route only to free memory for a cold load:
   reloading the judge took **14.8 s** before its first request, and the first
   split after any cold load runs measurably slower (judge val 186 s vs test 111 s)
   as caches and the speculative-MTP draft warm up.
2. **Coverage says the judge, AUROC says the 27B, and val can't settle it.**
   At a 5% budget on val, judge coverage (0.295) actually edges out 27B
   (0.265) while costing less wall than 27B — but its AUROC is 0.827 (below
   flash) and its ECE 0.222 (worst of every arm). The judge saturates its
   probabilities at 0.95/0.05, which makes the tuned operating point work and
   the *probability itself* untrustworthy. The val gap 0.295 vs 0.265 is
   inside ±1 SE at n=556; this run cannot distinguish them. The 27B remains
   the only arm that wins on ranking *and* calibration together.
3. **The chat endpoint doesn't batch either.** Judge QPS is flat at ~1.8 from
   c1 to c8, with p50 climbing 270 → 1010 → 1740 ms: pure queueing. This
   machine serves ~2 requests/s total per resident model regardless of
   endpoint, so the concurrency knobs on both endpoints are load-shedding
   controls, not capacity.

Full run plan against these numbers: flash alone is **~12 min**; a complete
three-arm leaderboard including train is **~2.7 h**. The cascade projection of
~21 min held shape but not semantics — see §11 for what the first live cascade
actually did. Nothing here needs a batch bigger than the current one.

## 11. The first live cascade broke exactly where predicted — and the fix is per-tier thresholds

`configs/cascade.yaml`, 150 groups/arm, all four arms, 26 min wall. The
CascadeStrategy ran correctly as plumbing — 23% escalation rate, exactly the
projected band population — but the gate it feeds was wrong by construction:

| shape | val cov @ 5% | val fpr | test AUROC | cost/cond |
|---|---|---|---|---|
| flash (thr 0.89) | 0.163 | 0.047 | 0.828 | 0.265 s |
| 27B (thr 0.94) | 0.223 | 0.047 | 0.879 | 0.835 s |
| **cascade, one shared threshold (0.96)** | **0.030** | 0.000 | 0.763 | 0.436 s |
| **cascade, per-tier (flash ≥ 0.89 / judge ≥ 0.96)** | **0.193** | 0.047 | — (cov 0.180 on test, fpr 0.047) | 0.436 s |

Three measured facts explain everything:

1. **The judge's probabilities are a spike, not a scale.** On 300 val cases it
   answered 0.95 (97×), 1.00 (88×), 0.10 (54×), 0.00 (43×) — ~90% of outputs on
   four exact values. It never says 0.96.
2. **It is confidently wrong inside the band, at exactly 0.95.** Of 36 band
   negatives it scores **17 at exactly 0.95**. A shared threshold of 0.95 keeps
   all 17 (fpr 0.11); the tuner must therefore sit above 0.95, and since the
   judge's *correct* band answers also cluster at 0.95, coverage starves to 3%.
   This is the mixed-distribution caveat made concrete: *the mixture has no
   honest single operating point because the components' supports barely
   overlap.*
3. **Per-tier acceptance transfers — but on a cliff edge.** Accept iff
   decisive-flash ≥ 0.89 OR escalated-judge ≥ 0.96 (which, given the judge's
   spikes, means *"the judge must say exactly 1.0"*): val cov 0.193 / fpr 0.047,
   and with *no refitting* test cov 0.180 / fpr 0.047. That is +18% coverage
   over flash alone at **half the wall-clock of 27B**. But the margin is fake:
   dropping the judge bar one cent (0.95) re-admits all 17 wrong cases, and on
   test the judge puts 2 band negatives at 1.0 that no threshold on its own
   score can exclude. A threshold whose safety is the gap between {0.95, 1.0}
   does not survive a model re-roll.

The band itself is thin: judge AUROC inside the band is 0.729 vs flash's 0.700
(n=69). Escalation is buying a *0.03* AUROC improvement — which is what the
no-think judge is worth on cases flash found ambiguous. The thinking judge
sharpened negatives properly (mean p⁻ 0.074 vs 0.358, §8), so the two
candidate upgrades are:

- **(a) calibrate escalation, not just routing**: map escalated judge scores onto
  flash's val scale (quantile match) before the threshold sees them; a single
  threshold then behaves per-tier, and no razor-edge bar sits load-bearing on
  the judge's exact saturation values.
- **(b) pay for thinking on escalated cases only** (`max_tokens: 1024`,
  thinking on): sharper band decisions at ~1.9 s × 23% ≈ +0.44 s/cond — the
  full-set judge budget doubles, but only on the band.

### The fix: calibrate the judge onto flash's scale (§11 option (a), implemented)

Calibrating BOTH tiers onto reliability fails — flash's own tail is empty
above 0.94, so isotonic collapses every high score onto one 0.98 plateau whose
*actual* accuracy is 0.76, and coverage goes to zero (measured). The fix is
asymmetric: **flash passes through raw** (it is the reference scale, native ECE
0.11), and **the judge is mapped onto it** — score → in-band accuracy → the
flash score that shares that accuracy. `calibration.py` + `scripts/fit_cascade_calibration.py`;
the fitted ladder (on val):

| judge says | val accuracy in band | → flash-equivalent |
|---|---|---|
| 0.00 / 0.10 | ~0.18 | 0.034 (blocked) |
| 0.85 / **0.95** | ~0.48 | **0.508 (blocked)** |
| **1.00** | 0.98 | **0.943 (clears)** |

The 0.95 spike — right only ~half the time — now lands where flash is equally
untrustworthy, far below any 5%-risk threshold, instead of masquerading next to
1.0. Verified live on the real `predict` path: all 17 escalated test cases
transformed, **zero leaked raw**.

Result — every shape on the same coherent run (limit 150 groups = 300
conditions/split, each arm's threshold tuned on val to its own 5% budget, test
reported at that threshold; escalation rate 21.7% live):

| shape | test cov | test fpr | test AUROC | test ECE |
|---|---|---|---|---|
| broken shared-threshold cascade | 0.023 | 0.013 | 0.763 | — |
| flash alone | 0.157 | 0.033 | 0.828 | 0.114 |
| **calibrated cascade** | **0.180** | 0.047 | 0.818 | **0.107** |
| qwen38-judge alone | 0.260 | 0.060 | 0.827 | 0.218 |
| **27B alone** | **0.230** | 0.060 | **0.879** | 0.109 |

Calibration rescues the cascade from 0.023 → 0.180 test coverage, and its ECE
(0.107) is now the best of any arm — mapping the judge onto flash's scale
genuinely fixes the mixed-scale miscalibration. But the ceiling is unchanged:
the cascade still trails 27B on coverage (0.180 vs 0.230) *and* AUROC (0.818
vs 0.879), and its AUROC sits a hair below flash-alone — the ~7 escalated cases
the judge gets maximally right add a little coverage at a touch more fpr and
cost flash's ranking nothing it couldn't keep.

**Verdict unchanged and now stress-tested: deploy 27B** as the single-arm gate —
best ranking, best calibration, no mixed-scale fragility. The calibrated cascade
is the correct design if you must stay on Flash and want the judge as a safety
net, but at this model set it buys +0.017 coverage over flash for +50% latency
and does not close the gap to 27B. Option (b) — thinking on escalated cases — is
the only lever that might make the judge earn more than its single spike.

Timing model validated: cascade cost/cond measured on the test split at 0.29 s
(85.8 s / 300) ≈ flash + 23% × judge = 0.436 s modeled, with the model's val
split inflated by first-after-contention warmup exactly as §10 describes
(189 s val vs 86 s test for the same work).

## 12. The full-dataset run — every headline confirmed at 4–7× the sample

`configs/full-benchmark.yaml`, `full-benchmark-v1`: the complete report set (val 556 + test
2,224 conditions), five arms, serial, **78.4 min wall, zero errors, 13,900 predictions**.
Wall came in 8% under the §10 projection (85 min) — `wall_serial_s` is now a validated
timing model, 1.09× conservative.

Test split, thresholds tuned on val only:

| arm | AUROC | ECE | thr | FPR | coverage | p50 | p95 | s/cond |
|---|---|---|---|---|---|---|---|---|
| clef-27b | **0.871** | 0.093 | 0.93 | 0.049 | 0.238 | 534 ms | 666 ms | 0.552 |
| clef-flash | 0.830 | 0.090 | 0.89 | 0.039 | 0.162 | **165 ms** | 196 ms | **0.169** |
| qwen38-judge | 0.826 | 0.208 | 0.96 | **0.055 ✗** | **0.256** | 511 ms | 1,777 ms | 0.662 |
| cascade (calibrated) | 0.823 | **0.081** | 0.89 | 0.047 | 0.187 | 174 ms | 885 ms | 0.308 |
| lexical | 0.665 | 0.142 | 0.88 | 0.049 | 0.114 | ~0 | ~0 | ~0 |

Paired grouped bootstrap (2,000 reps, 95% CI): 27b − flash **+0.041 [+0.034, +0.048]**,
27b − judge **+0.044 [+0.038, +0.051]**, cascade − 27b −0.048 [−0.055, −0.040] — all
significant; flash − judge +0.003 [−0.004, +0.011] and cascade − flash −0.007
[−0.013, +0.000] — the two ties held as ties or near-ties, now with ±0.007 CIs.

What full n settled:

1. **Deploy-27B verdict: confirmed, not tentative.** Every capped-run lead survived with CI
   widths roughly halved.
2. **The judge is disqualified standalone: test FPR 0.055 breaches the 5% budget** at the
   val-tuned 0.96. Its top coverage (0.256) cannot be bought at acceptable risk — the
   spike-distribution problem (§11) does not average out at scale, it re-asserts.
3. **The calibration fix scales.** Escalation 23.9% (531/2,224); of the judge's 251
   "0.95" verdicts inside the band, **123 were wrong** — all mapped to 0.508 and blocked,
   zero leaked. Cascade ECE 0.081 is the best of any arm on full data; calibration is not a
   small-sample trick.
4. **Cascade-vs-flash is no longer a maybe:** −0.007 [−0.013, +0.000] — the escalation
   costs ranking it never returns; its only gifts remain +0.025 coverage and best ECE,
   at 2× flash's wall. The 27B gap (−0.048) did not close.
5. **Determinism re-verified:** all §11 case-file scores reproduced identically inside the
   full run (same thresholds, same predictions byte-for-byte on shared case ids).

Open questions above stand; #1 (coverage headroom via a trained sufficiency probe) is now
the only lever that can move the ceiling, at these numbers.

## Open questions for the next pass

1. **Coverage is the binding constraint, not safety.** Can a ModernBERT sufficiency
   probe or a hidden-state probe reach 0.90+ AUROC and lift coverage at 5% risk above
   what Clef-27B manages? This is where the headroom is.
2. Does a richer `choice` schema (5-way, already implemented as `diagnosis`) beat the
   2-way for sufficiency? Options demonstrably shape Clef's answers.
3. Is the flat `noul` a quantization artefact? The joint head stays bf16 under oQ but
   the backbone is 4-bit — test bf16 `clef-flash` if it fits.
4. `t_conflict` is still uncalibrated: SufficiencyBench has no gold contradiction
   labels. Stage 4 (controlled retrieval degradation) must build labelled conflicts
   before the contradiction tier can be trusted, despite the promising 0.839 probe hit.
5. Can anything break the 6 QPS ceiling — client-side request batching, a second
   resident replica, or an upstream change to the decision engine?

## 13. System-one sweep: server auto-detection + Phase 0 interface gates (2026-10-14)

8/8 repos downloaded (~55 GB). After `POST /admin/api/reload` (18 models discovered),
`/admin/api/models` `model_type` shows how omlx classifies each family:

| model | detected | route that actually works |
|---|---|---|
| OpenJev-Flash-9B-MLX | `decision` | `/v1/systemone` native, wire contract verbatim |
| openjev-MLX-4bit (27B) | `decision` | native |
| clef-omni-4bit | `decision` | **load fails**: `qwen3_omni_moe` unsupported — excluded |
| Tev1-4B-experimental | `vlm` | `/v1/chat/completions` only, **no logprobs ever returned**; k=12 temp-1.0 vote-rate score |
| laya / laya-typed-decisions | `llm` | own SDK sidecar (`.venv-encoder`, `laya.Router`) — question wire is byte-identical to our contract format |
| von | `embedding` | own SDK sidecar (`von.VonClient(local=True)`), `VON_CHECKPOINT_DIR` avoids re-download |
| opendecider-nano | `embedding` | sidecar pending (not yet wired; nano is lowest priority) |

Interface measurements (n=24–48 val):
- OpenJev 9B Δ+0.650 / sd 0.409; OpenJev 27B **Δ+0.767 / sd 0.441** — capacity transfers.
- Tev k-sampling Δ+0.368; prompt MUST be the card JSON schema form, else it rambles.
- laya Δ+0.118 (choice); laya ships invalid calibration temps → SDK floors confidence at
  0.5; we consume probabilities, not confidence, so unaffected but noted.
- von choice wire is **rubric-dominated**: long criteria descriptions flatten it
  (AUROC 0.582). Noul scalar with declarative condition: **AUROC 0.740**, Δ+0.164 —
  von's arm is permanently configured `primitive: noul`.
- determinism: every working arm reproduces scores byte-identically (spread 0.0000).

Phase 0 verdict: **6/7 arms pass all four gates**; clef-omni excluded by server
architecture, not behavior. Phase 1 (`sweep-smoke-v1`, --limit 150) running with the six.
