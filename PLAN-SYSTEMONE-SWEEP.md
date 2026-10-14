# Plan: System-One Model Sweep — widen the leaderboard beyond Clef

**Goal:** benchmark every promising open-source System 1 decision model we can actually
serve on this box against the identical sufficiency experiment, then **append** the new arms
to the existing `full-benchmark-v1` leaderboard — the two Clef arms are **not rerun**; their
thresholds and 13,900 predictions are already banked on the identical val/test splits.

Discovery was done live on 2026-10-14 against the HuggingFace API + this server. Every model
below is a real, verified repo with the quoted properties read from its model card.

---

## 1. The roster (what exists, verified today)

### Tier A — native `/v1/systemone` decision models (drop-in arms, same contract as Clef)
omlx 0.7.1.dev1 release notes: *"Clef and Clef-Flash (27B, 9B) and OpenJev (27B) are
detected automatically, or you can set the model type to decision."*

| model | size / download | license | notes |
|---|---|---|---|
| `openjev/openjev-MLX-4bit` | ~15.2 GB | **CC-BY-NC-4.0** ⚠ | the flagship open Jev; omlx explicitly supports. Benchmarks internally (research OK; NC blocks production) |
| `openjev/OpenJev-Flash-9B-MLX` | ~9.5 GB | CC-BY-NC-4.0 ⚠ | 9B/27B capacity split *inside the Jev family* — mirrors our Clef 9B-vs-27B finding |
| `Cloudflare/clef-omni` (or `mlx-community/clef-omni-4bit` ~17 GB) | 30B MoE | Apache-2.0 | official; "API fully compatible with Jev and SystemOne"; **verify omlx detects it as decision** (MoE + omni backbone may need `model type: decision` set manually) |
| `autotrust/JEV-9B` / `JEV-27B` | ~18/55 GB | check card | TypeSafe-Jev distill trained on the noul/choice/score trio — likely same arch; verify detection, else demote to Tier B test |

### Tier B — encoder decision models (run via a thin local sidecar, ~1–2 GB each)
Not omlx-native; these are ModernBERT-style non-autoregressive encoders with their own
serving SDKs. 10–20 ms/decision — potentially the latency floor of the whole board.

| model | size | license | notes |
|---|---|---|---|
| `wfzyx/von` (Von 1.3) | 395M / ~3.2 GB | Apache-2.0 | **8,192-token context** (Laya has 512/1024) — the only encoder that ingests our full ~750-token states with room; Option-Marker head; own `/v1/systemone` wire protocol → our client may work nearly unchanged |
| `convaiinnovations/laya-typed-decisions` | 421M / ~1.7 GB | Apache-2.0 | ModernBERT-large; card benchmarks itself on the same typed-decisions trio; **ctx 1024 → fits, but long ELI5 states truncate** |
| `manjunathshiva/opendecider-nano` (+`-small`) | ~0.4B/…, ~.8 GB | Apache-2.0 | claims 18 ms on M4 Max; card claims beats Laya head-to-head (0.796 vs 0.766) on typed-decisions |

### Tier C — autoregressive structured-decision baseline (new strategy, chat endpoint)
| model | size | license | notes |
|---|---|---|---|
| `togethercomputer/Tev1-4B-experimental` | ~8 GB | Qwen3.5 base Apache; **ft weights license "being finalized"** ⚠ | Jev-*inspired* but keeps the LM head — returns one option letter. Our harness reads the letter's logprobs → probability. Already slotted in `SYSTEM-ONE.md` as the AR-vs-NAR contrast arm. `Tev1-0.8B-experimental` if we want a size split |

### Blocked / deferred (found, but not runnable today)
- `Jundot/d3-oQ4e` etc. (Decision 3.0, 27B multimodal): needs omlx PR #4470 — **newer
  than our 0.7.1.dev1**. Candidate for the next server upgrade, not this sweep.
- TypeSafe Jev 1.13 (the commercial teacher): API-only; would need procurement sign-off.
- GGUF/ONNX/CoreML ports of all of the above: wrong runtime for omlx; ignore.

**Licensing flag:** OpenJev arms are NC — fine for this benchmark, must never become the
production gate. The sweep's Apache-only column (Clef, Clef-Omni, Von, Laya, OpenDecider)
is the shippable set.

## 2. Integration work

1. **Download (~70 GB total, 450 GB free):** `huggingface_hub.snapshot_download` into
   `~/models/<org>/<name>` (same layout omlx scans). `HF_TOKEN` is **currently empty in
   this shell** — everything selected above is public, so downloads should work anonymous;
   if any repo 401s (Tev's "license being finalized" is the likely gate), flag it and
   skip rather than block the sweep.
2. **Server registration (facts from §9):** freshly downloaded checkpoints are invisible
   until `POST /admin/api/reload` → verify each new model reports
   `model_type: "decision"` at `/v1/models`. Anything that doesn't: try manual
   `model type: decision` in omlx settings; if it still won't bind → demote to Tier B/C.
3. **New harness arms (code):**
   - Tier A: zero new code — `kind: system1` arms pointing at the new model names.
   - Tier B: new `strategies/encoder.py` — an HTTP client for the von-sdk /
     transformers sidecar exposing the same `predict(case) -> p_sufficient` interface;
     run one sidecar per encoder (they're ~1 GB; CPU/MPS; they do NOT occupy the GPU
     decision slot, so measure them separately, not interleaved, to keep §6/§10
     contention hygiene intact).
   - Tier C: new `strategies/ar_decision.py` — `/v1/chat/completions` with
     `logprobs: true, max_tokens: 1, temperature: 0`; probability = softmax over option
     letter logprobs (Tev's documented interface). Same rubric wording as
     `sufficiency-v2-choice` for a fair fight.
4. **Contract consistency:** all arms answer the identical 2-option sufficiency question;
   encoders that only do generic classification get the same instruction text the Clef
   `choice` questions use (no per-model prompt tuning — we're comparing models, not prompts).

## 3. Execution order — small first, then append (no Clef reruns)

**Phase 0 — smoke, limit 20 (~10 min, per §1's hard-won lesson):** per-model probe on 20
mixed cases. Gates a model must pass before it earns a real run:
- endpoint speaks the contract (probability comes back at all);
- **non-flat output** (the `noul` blocker: sd > 0.01 across heterogeneous cases, else the
  model is dead-on-arrival like `noul` was — §1);
- instruction sensitivity (Δ between sufficiency vs control wording > ε — §1's four-instruction test);
- parse/format determinism (10 identical requests → same score).

**Phase 1 — calibration-size run, limit 150 (~45–60 min):** all *new* arms (+ cascade
variants if any new fast tier beats Flash's cost curve), thresholds tuned on the val
subset, same seed/bootstrap. Compares directly against the banked `cascade-flash-judge-v1`
table at identical n.

**Phase 2 — full append, new arms only (~90–110 min):** `sweep-full-v1` on the complete
val+test (556/2,224). Clef-Flash and Clef-27B **not rerun**: thresholds tune on val and
test is reported once — the banked full-run predictions are on the byte-identical split
files, so the merged leaderboard is statistically coherent. A 20-case spot-check re-run of
Clef-Flash at the start of the window verifies the server hasn't shifted (scores must
match banked values exactly, §12.5 determinism check).

**Phase 3 — merge + report:** `report_from.py --merge full-benchmark-v1 sweep-full-v1`
(or a 30-line merge over the two prediction dirs — same schema, same case IDs); one
leaderboard, nine arms; new colors in `make_report_figures.py`; a capacity curve across
four families (Clef, OpenJev, Jev-distill, encoders) — AUROC vs latency, coverage at 5%.

## 4. Expected runtime & memory budget

| phase | wall | notes |
|---|---|---|
| downloads | ~30–60 min | ~70 GB, parallelizable; 450 GB free |
| reload + detection verify | 5 min | admin/api/reload, model_type checks |
| tier B/C wiring + unit tests | ~half a day of work | 2 strategies + sidecar runner + config |
| Phase 0 smoke | ~10 min | |
| Phase 1 calibration-size | ~45–60 min | OpenJev-27B ≈0.55 s/cond assumed; encoders ≈0 |
| Phase 2 full append | ~90–110 min | OpenJev-27B ~25 min, Jev9B/Omni ~12–20 each, Tev ~25, encoders ~1 |
| merge, charts, docs | ~1 h | |

Memory: server peak today 54 GB; OpenJev-4bit +15 GB, Omni-4bit +17 GB — serial arms,
~80 GB peak worst case on a 256 GB box. Encoders live off-GPU. All fits; enforce
§10's rule anyway: one GPU-heavy arm at a time, nothing co-tenant.

## 5. What this sweep can prove (the scientific stakes)

1. **Is Clef's win family-specific?** If OpenJev-27B lands at ~0.87 too, the capacity
   finding generalizes across NAR architectures; if it collapses, Clef's joint head matters.
2. **Can the 400M encoders touch the Clef-Flash lane?** Von/Laya/OpenDecider at ~15 ms
   vs Flash at 165 ms — if any encoder gets AUROC ≥0.80 at 5% FPR, it rewrites the
   serving economics (10× the QPS ceiling, no decision-engine batching needed). This is
   the §12 open-question-#1 probe, off the shelf.
3. **AR-with-letters (Tev) vs AR-with-schema (our judge) vs NAR:** completes the
   architecture triangle on identical data — Tev should dominate the current judge on
   latency (1 token vs 45) at similar AUROC if its distillation is good.
4. **The flat-score test as a gate:** expect at least one casualty — the sweep's smokes
   will re-run the §1 blocker checklist on six new models; survival is itself a result.

## 6. Risks

- **OpenJev NC license** → benchmark-only; never productionize those arms (already flagged).
- **clef-omni / autotrust detection** may fail → demotion paths defined above.
- **Tev gated weights** (HF_TOKEN empty right now) → skip gracefully.
- **Encoder context truncation** (Laya 1024) → record `truncated: true` per case; report
  AUROC on untruncated subset beside the headline, or the number lies.
- **Sidecars change the contention regime** → encoders measured in separate windows.
- **6 QPS ceiling still governs Tier A** → total sweep time scales linearly with arms;
  the encoder arms are the escape hatch, which is itself finding-worthy.

## 7. Decision points for you

1. Include the big downloads (OpenJev-4bit + Omni-4bit, ~32 GB) or start 9B-only?
2. NC-licensed OpenJev acceptable for benchmark use? (Recommended yes — it's the only
   other omlx-native decision family.)
3. Von sidecar first (Apache, 8K ctx, fastest to wire) as the Tier B pilot?
4. Want the HF_TOKEN re-exported before starting, in case Tev1 is gated?

---

## 8. STATUS 2026-10-14 — decisions answered, execution started

1. ✅ **Token**: stored at `~/.config/hf/token` (mode 600, auto-exported via `.zshrc`);
   verified — authenticated as **slammyT**.
2. ✅ **NC OpenJev approved** for benchmark-only use (flag persists in all docs).
3. ✅ Order chosen: small→big — downloads running now (`scripts/download_sweep_models.py`,
   log `results/sweep-download.log`), laya/von/opendecider first (~3 GB), then
   OpenJev-Flash-9B → OpenJev-27B-4bit → Tev1-4B → clef-omni-4bit last.
4. ✅ **Gating check with token: nothing on the roster is gated** (laya, laya-typed,
   von, opendecider-nano, OpenJev MLX ×2, Tev1-4B, clef-omni — all `gated=False`).
5. 🆕 **`convaiinnovations/laya` (base) — added by explicit request and upgraded in
   status:** it ships a `pip install laya` package whose `Router.predict(state, questions)`
   speaks the *exact* noul/choice/score protocol, plus multilingual 8192-token mode
   (`max_len=8192`) — the context-truncation risk (§6) is avoidable by config.
6. ✅ **Wiring complete, code banked:**
   - `strategies/encoder.py` — one `EncoderSidecarStrategy` serves all of Tier B
     (identical contract payload; stamps `truncated` per case).
   - `strategies/ar_decision.py` — Tev-style letter selection; soft probability from
     first-token logprobs over {A,B}; stamps `hard_choice` if it degrades to binary.
   - `scripts/encoder_sidecar.py` — stdlib HTTP sidecar hosting laya/von in a dedicated
     `.venv-encoder` (off-GPU on purpose).
   - `runner.py` — new `kind: encoder|ar_decision` arms build and appear in leaderboards
     like any other arm.
   - `configs/sweep-smoke.yaml` — Phases 0–1 config (limit 20 → 150).

**Next actions in order:** downloads finish → `POST /admin/api/reload` → confirm which new
models report `model_type: decision` (auto-detection matrix itself goes in FINDINGS §13) →
Phase 0 smoke gates per model → Phase 1 calibration-size run → Phase 2 full append →
merged 9-arm leaderboard + charts + docs refresh.

## 9. RESULTS — detection matrix + Phase 0 (in progress)

Downloads: **8/8 ok**, ~55 GB into `~/models`. Server reload: *"Re-discovered 18 models
from 2 directories"*. `/admin/api/models` auto-detection (what omlx thinks each new repo is):

| model | detected `model_type` | `engine_type` | ctx | usable via |
|---|---|---|---|---|
| OpenJev-Flash-9B-MLX | **decision** ✅ | decision | 262144 | `/v1/systemone` (native) |
| openjev-MLX-4bit (27B) | **decision** ✅ | decision | 262144 | `/v1/systemone` (native) |
| clef-omni-4bit (30B) | **decision** ✅ | decision | 131072 | `/v1/systemone` (native) |
| Tev1-4B-experimental | `vlm` | vlm | 262144 | `/v1/chat/completions` → Tier C `ar_decision` |
| laya | `llm` | batched | — | NOT auto-decision → Tier B sidecar (its own `laya` Router) |
| laya-typed-decisions | `llm` | batched | — | Tier B sidecar |
| von | `embedding` | embedding | 8192 | NOT auto-decision → Tier B sidecar (von-sdk) |
| opendecider-nano | `embedding` | embedding | 7999 | Tier B sidecar |

**Finding:** omlx auto-detects the **Jev/Clef decision architectures natively** (all three
tier-A arms came back `decision`, as the release notes promised) but sees the two
ModernBERT-encoder families as plain `llm`/`embedding` — its decision path keys on their
own weight layouts, not the modernbert+marker pattern. So the Tier B sidecar design
(`encoder_sidecar.py` + one `EncoderSidecarStrategy`) is **required, not just convenient** —
exactly the plan's contingency, now confirmed as the mainline. Tev lands as `vlm` (its
Qwen3.5-4B base) and is reached through the chat endpoint by the `ar_decision` strategy.

Phase 0 gates (`scripts/probe_sweep_model.py`, n=24 val, four §1 blockers):
- **OpenJev-Flash-9B-MLX: PASS.** p⁺ 0.879 / p⁻ 0.229 (Δ +0.650), sd 0.409 — a wide,
  live, binding signal, not `noul`-flat; determinism spread 0.0000; control ("mentions a
  place") 0.924 confirms it isn't just echoing the instruction. This is a working gate.
- **openjev-MLX-4bit (27B): PASS.** Δ +0.767, sd 0.441, spread 0.0000 — beats 9B's
  separation; capacity transfers. clef-omni-4bit: **BLOCKED** — omlx cannot load
  `qwen3_omni_moe` (HTTP 409, hard architectural gap on this server); dropped from sweep.
- **Tev1-4B-experimental: PASS (k-sampling).** omlx serves **no logprobs** on chat for
  this checkpoint (`choices[0]` has no `logprobs` key even when requested) and rejects it
  on `/v1/systemone` ("not a decision model"). The letter-logprob plan was dead; rewrote
  `ar_decision` to the only honest soft score left: k=12 votes at temperature 1.0,
  p̂ = vote rate (greedy label recorded separately; determinism gate runs on greedy).
  Prompt must be the card's JSON `{"context":…, "schema": {"answer": {"enum": […]}}}`
  form — a plain "OPTIONS: A=…" prompt made it ramble ("The"). n=24: p⁺ 0.826 /
  p⁻ 0.458 (Δ +0.368), sd 0.277, greedy deterministic, off-format votes dropped.
- **laya: PASS** via sidecar :8101 (choice wire, native dict-question format = our
  contract format verbatim): Δ +0.118, sd 0.250, deterministic. (Router warns the
  checkpoint ships out-of-range temperatures → confidence rescaled to 0.5 floor;
  confidence from laya arms is uncertified, probabilities are ours.)
- **laya-typed-decisions: PASS** via dedicated sidecar :8103: Δ +0.118, sd 0.144,
  deterministic. (Caught a config bug worth keeping: both laya arms initially pointed at
  :8101, which silently would have run the typed arm on the multilingual head.)
- **von-1.3: PASS on its native noul axis only.** Choice-option scoring is near-dead on
  our matched pairs (wire AUROC 0.582, n=48) — its option-marker head is dominated by
  the criteria *descriptions*, and our long rubric text crushes the signal. The **noul
  scalar** (declarative condition, no criteria) measured AUROC **0.740** / Δ +0.164 /
  sd 0.213 / deterministic. von's arm config carries `primitive: noul` + declarative
  instructions permanently. (Irony banked: the contract's `noul` primitive, kept alive
  because the first noul model's endpoint died, turned out to be von's best face.)

**Phase 0 verdict: 6/7 new arms cleared every gate** (3 native decision + 1 AR-sampled +
2 encoder families; von on noul). Only clef-omni is excluded, by server architecture
support, not by behavior. Phase 1 (calibration-size, --limit 150) is go for all six.

### Phase 1 calibration-size results (`sweep-smoke-v1`, 150 groups = 300 conditions/split, zero errors)

| arm | val AUROC | test AUROC (uncertified — small n) |
|---|---|---|
| openjev-27b-4bit | **0.911** | **0.883** |
| openjev-flash-9b | 0.877 | 0.845 |
| tev1-4b (k=12 votes) | 0.819 | 0.816 |
| laya-typed-decisions | 0.721 | 0.676 |
| von-1.3 (noul) | 0.651 | 0.643 |
| laya | 0.632 | 0.626 |

The headline if it survives full n: **OpenJev-27B beats the incumbent clef-27b
(0.871 test) and OpenJev-Flash-9B beats clef-flash (0.830)** — a genuinely new
leaderboard, with the CC-BY-NC 27B benchmark-only and Flash-9B as the production
candidate. Tev's k-sampling landing at judge-level AUROC (~0.82) is the second
surprise. laya at 0.63 ≈ lexical level: the encoder-family finding is
typed-decisions > base-laya, but neither beats a decision checkpoint.

Clef drift guard (20 val cases re-scored vs banked full-run predictions):
**byte-identical on both Clef arms — banked predictions remain valid for the merge.**

Phase 2 (`sweep-full-v1`, six arms × 2,780 conditions) launched; merge via
`scripts/merge_sweep_leaderboard.py`.

**Incident log:** at 12:51 the runner and two sidecars were SIGKILLed mid-run
(likely memory-pressure eviction during openjev-27b cold load — `phase2-run.log`
empty, buffers lost with the process). Nothing measured was lost: predictions persist
per arm-split. Completed and banked:
laya, laya-typed-decisions, von, openjev-flash-9b (all 2,780 each, zero errors;
**openjev-flash-9b test 0.860 vs clef-flash 0.830** — lead holds at full n;
openjev-27b val 0.914). Second failure's **root cause found**: at 15:10 a co-tenant
`vllm serve LiquidAI/LFM2.5-24B --host 127.0.0.1 --port 8000` (from
cooker-deep-dives) bound the *more specific* loopback address and stole every local
connection from omlx's wildcard `*:8000` — the resume died of ConnectTimeout against
an imposter server answering `lfm25-24b-a2b`. User-approved: vllm killed (PID 62237),
loopback reclaimed (18 models back, decision endpoint verified, omlx PID 88476 never
restarted so banked predictions stay valid). Resume relaunched detached under nohup
(`configs/sweep-full-resume.yaml`, same run dir, remaining arms only); completion
marker `results/phase2-resume.done`.
**Lesson banked: before every run, assert `GET /v1/models` reports `owned_by: omlx`
with the expected 18 ids — a wildcard bind is not ownership.**

Nothing in Phases 0–2 touches the banked Clef predictions; worst case a new arm is
discarded and the merged leaderboard still stands on the full run.
