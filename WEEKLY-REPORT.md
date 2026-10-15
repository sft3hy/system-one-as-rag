# Weekly Activity Report — AI Answer Safety Gate

**Week ending:** Friday 16 Oct 2026 (report prepared 14 Oct)
**Team:** Search & Answering Infrastructure
**In one sentence:** we tested four ways of checking whether an AI's retrieved documents actually
contain the answer before it responds, and we now have a clear recommendation for which one to deploy.

---

## 1. What we're building, in plain English

You already know how RAG works: when a user asks a question, the system *retrieves* a handful
of relevant documents and hands them to a large language model, which writes an answer from
them. That works well — **until the retrieved documents don't actually contain the answer.**
The model then either makes something up or says "I don't know," and we have no way of knowing
in advance which will happen.

Our project adds a **safety gate in front of the answer**. Think of it as an airport security
checkpoint for every question:

- **Green light** → the documents clearly contain the answer; answer immediately and confidently.
- **Red light** → the documents are insufficient; refuse or escalate to a human.

The gate works by asking a small, very fast AI model one question per request:
*"What is the probability these documents are sufficient to answer this question?"* — and only
letting answers through when that probability is high enough. "Small and fast" matters: this
checkpoint runs on *every* query, so if it's slow or expensive, the whole product slows down
with it.

**The single number that governs everything is our risk budget:** we agreed the gate may let
through a confidently-wrong answer on **no more than 5% of the questions it approves**. That
5% is the "false-sufficient rate" (FPR) you'll see throughout.

## 2. What we did this week

1. **Finished all benchmark runs — including the complete exam.** Every candidate model was
   tested on the full public SufficiencyBench exam (1,390 questions, each paired with a
   deliberately complete and an incomplete document set, so we always know the right verdict in
   advance). The final full run took 78 minutes and produced **13,900 judgements with zero
   failures**. All automated checks pass.
2. **Fixed a serious flaw in our two-step design** (see §4, "the combo"), re-ran it, and
   confirmed the fix works end-to-end — and still holds at full scale.
3. **Made the deployment decision** and stress-tested it on the complete dataset: this is the
   week we can say, with statistics behind it, which gate to ship.
4. **Widened the talent search: eight new open-source "instant judgement" models** from
   around the internet were discovered, vetted, and sat the same full exam under the same
   rules (§10). Six completed; one couldn't even run on our hardware; one failed an
   entrance exam. **Result: the crown stands, but the runner-up changes.**

## 3. The five contestants

| Contestant | What it is | Analogy |
|---|---|---|
| **Clef-Flash** | Small (3B) purpose-built "instant judgement" model | Sharp junior analyst — fast, usually right |
| **Clef-27B** | Same idea, 9× bigger | Senior analyst — slower, noticeably more reliable |
| **Chat judge** | General chat model asked to referee each case | Consultant — talks a good game, overconfident |
| **Two-step combo** | Junior triages everything; only the hard 22% goes to the consultant | Junior + escalation |
| **Baselines** | Simple word-overlap and size checks | Checklists, no AI at all |

### …and six new challengers (§10)

| Challenger | What it is | Analogy |
|---|---|---|
| **OpenJev-27B** | Biggest of a rival "instant judgement" family | Brilliant but cocky senior analyst — sharpest instincts, signs off on things he shouldn't |
| **OpenJev-Flash-9B** | Mid-size from the same family | Solid senior analyst — beats our junior on the exam, works a bit slower |
| **Tev** | A chat model trained to commit to an answer without waffle | Analyst forced to answer yes/no, 12 times, no hedging |
| **Laya & Laya-typed** | Lightweight "one-glance" checkers (one pass, no thinking) | Speed-reading interns |
| **Von** | Single-glance gut-feel checker, Apache-licensed | The intern who stamps "don't know" rather than risk an error |

## 4. Results — the chart that matters

![Safe auto-answer rate, whole field](results/report-figures/exec_field_coverage.png)

**How to read this chart:** each bar is one of the eleven models we've now tested; the length
is the share of all user questions the gate would let the AI answer directly while keeping its
error rate within our 5% budget. **Green = the model honestly keeps that promise; red = it
breaks it.** Shorter bars aren't failures — the cheap interns (Laya, 6%) simply aren't
smart enough to tell which questions are safe, so they approve almost nothing.

**What it says (full dataset, 2,224 test questions — final numbers):**
- The three widest bars — OpenJev-27B (31%), chat judge (26%), Tev (21%) — are **all red**:
  they cover the most traffic by promising more than they can deliver. That coverage cannot be
  bought at the risk we accept.
- **Clef-27B (green, 24%) remains the widest bar that is honest.** This week's eight-model
  search did not dislodge it — it confirmed that combining top judgement with top restraint is
  rare.
- **OpenJev-Flash-9B (green, 18%) is this week's genuine catch**: a budget-compliant fast
  option that judges better than Clef-Flash (see §10).
- **The bottleneck is still timidity, not danger**: even the honest champions auto-answer only
  16–24% of traffic, and roughly half of what they block was actually safe to answer.
  Coverage — not safety — is where the business value is locked up (§9).
- **The chat judge's bar is a trap** (§6): it reaches 26% only by sounding artificially
  confident; on the full exam it breaks the 5% budget outright (5.5%).

## 5. How each candidate performed — the scoreboard

![Quality vs speed, whole field](results/report-figures/exec_field_quality_speed.png)

**How to read this chart:** every dot is one model — all eleven we've now tested. Right =
higher judgement quality (AUROC, explained below); further up = lower delay per question
(the axis multiplies by ten each step — Laya at the bottom-left answers 35× faster than Tev).
There is no free lunch; the interesting question is who gets the best trade-off — and whether
they keep their 5% safety promise (green) or break it (red).

**What it says (all 11 models, full dataset):**
- **Clef-27B (👑) stays the pick:** the only model that is simultaneously among the best
  judges *and* green. Its quality lead over the whole green field is statistically decisive.
- **OpenJev-27B (red, top-right) is the week's big story:** highest judgement quality we
  have ever measured (0.889) — but red, disqualified (§6, §10).
- **OpenJev-Flash-9B (green star)** beats Clef-Flash on quality decisively at moderate speed —
  the fast-tier upgrade to price next week.
- **The intern tier (bottom-left) is cheap but blunt:** Laya/Von answer almost nothing
  safely; Tev pays chat-model speed (far right) for mid-table green-less accuracy.

### The scoreboard numbers, demystified

| Term | Plain-English meaning |
|---|---|
| **AUROC** (0–1) | *Discrimination.* Give the judge a known-good case and a known-bad case at random — AUROC is the chance it rates the good one higher. 0.50 = coin flip; 0.88 = strong; 1.00 = perfect. |
| **FPR** (our 5% budget) | *The risk.* Of the questions the gate **approved**, how many actually had insufficient evidence — i.e. confidently wrong answers shipped to users. We cap this at 5%. |
| **Coverage** | *Throughput.* % of all traffic the gate lets through at that risk level. This is the business upside: coverage × traffic = auto-answered queries. |
| **ECE** (0–1, lower better) | *Calibration — "does its confidence match its accuracy?"* If the gate says "90% sure" on a batch of cases, it should be right ~90% of the time. High ECE means its stated confidence can't be trusted even if its gut decisions are partly right. |
| **p50 / p95 latency** | *Speed.* Typical response time (p50) and worst-case-for-95%-of-traffic (p95). We design around p95 because unhappy users remember the slow ones. |
| **Significance (95% CI)** | *Did it really win?* We re-scored the test thousands of times with reshuffled samples; if the measured gap holds in 95% of those re-runs, it's real, not luck. Clef-27B's lead is significant; the middle three are statistically tied. |

### What is AUROC? The 30-second version

![AUROC explained](results/report-figures/exec_roc_explainer.png)

That word — **AUROC** — is the one most likely to trip people up, so here it is with no jargon.
*(The blue curve is the real, measured performance of our best gate on this week's full
2,224-question test set; the purple, orange, green and red curves are reference lines to
calibrate your eye.)*

Every safety gate faces the same tension: be **strict** (few wrong answers slip through, but
lots of good questions get blocked) or **relaxed** (lots of questions answered, but more wrong
ones slip through). You can dial this strictness knob anywhere along a scale — a "threshold".
Pick the strictest setting and you answer almost nothing; pick the loosest and you answer
everything, correctly or not.

This chart shows *every possible setting of the knob at once*:

- **Right axis (the risk):** the share of unanswerable questions wrongly waved through.
- **Up axis (the payoff):** the share of answerable questions correctly approved.
- **Each point on a curve** is one strictness setting. The curve traces what happens as you
  slide the knob from ultra-strict (far left) to ultra-relaxed (far right).

The three reference lines:

- The **purple "perfect" line** hugs the top-left corner: a flawless judge answers everything
  worth answering and nothing that isn't.
- The **red dashed diagonal** is the coin flip: to approve twice as many good questions, you're
  forced to wave through exactly twice as many bad ones. Zero skill — the payoff never outruns
  the risk.
- **Every real judge sits in between**, and the question is simply *how far above the coin-flip
  line its curve bows*.

**AUROC is just the area of the shaded region under a gate's curve, as a score from 0.50
(coin flip) to 1.00 (perfect).** It's the number we quote because it grades a gate's *raw
judgement ability* independent of where we happen to set the threshold — the curve lifts up and
left as a judge gets better, and AUROC is the single-number summary of that lift.

Two useful gut checks when you see one quoted:

- **0.50 = useless** — pure coin flip. (Our dumb control tests score exactly this, which is
  how we know the exam itself isn't fooling us.)
- **The gap is what matters, not the decimal.** The difference between 0.83 and 0.87 is *not*
  "4% better" in plain terms — in live traffic it translates directly into how many more
  questions the machine can be trusted to handle without a human. Clef-27B's AUROC of 0.87
  vs Clef-Flash's 0.83 is a genuinely large, statistically significant capability jump — and on
  the full exam the statistics are now decisive, not suggestive (§5).


## 6. The week's key discovery: three models, three ways to fake confidence

![Why the red models are red — confidence across the whole field](results/report-figures/exec_calibration_field.png)

**How to read this chart:** each panel shows how one model spread its confidence across the
same 2,224 test questions — blue = cases it should approve, red = cases it should block. The
dashed bar is the approval threshold. A *good* judge pushes blue right and red left with a
clean gap; **any red that leaks past the dashed bar is risk we pay for.**

**What it says — the new rival, the judge, and the champion:**
- **OpenJev-27B (second panel) fails differently from the judge — and it's the more
  interesting failure.** It is *not* shouty: its scores look beautifully separated (that's the
  0.889). But look at the left: a big red mountain of "0–10% sure" correct rejections… and a
  thin but stubborn tail of red cases sitting at 95–100% that should have been blocked. Even
  the strictest threshold can't cut them — that tail is its 7.3% budget breach. Brilliant
  instincts, unlucky certainty. Our existing "translator" calibration recipe (§10 next steps)
  is designed to squash exactly this tail — the highest-value follow-up now on the board.
- **The chat judge (third panel) shout-storm confirmed at full scale:** almost everything lands
  on four values (0%, 10%, 95%, 100%), and of 531 escalated cases **251 came back "95% sure"
  — 123 of them wrong**. Our shipped translator maps 95% → ~51% (block) and rescued the
  two-step combo from 2% → 19% coverage, best-calibrated on the board — still not enough to
  beat Clef-27B.
- **Clef-27B (first panel) does the boring, correct thing:** opinions spread smoothly, no
  spikes, no heavy tail past the bar. Boring is exactly what a safety gate should look like.
- **Laya-typed (right)** is honest but timid — its energies never reach the bar, so it
  approves almost nothing: safe, and nearly useless.

*Historical note: the first version of this chart (below) is how we caught the judge's fake
confidence last week — same phenomenon, now with the whole field to compare against.*

![Score distributions](results/report-figures/score_distributions.png)

## 7. What the machine costs (and it fits)

| | Clef-Flash | Clef-27B | OpenJev-Flash-9B ★ | OpenJev-27B (research) | Tev | Laya-typed |
|---|---|---|---|---|---|---|
| Typical delay per checkpoint | 0.17 s | 0.54 s | 0.47 s | 0.61 s | 1.52 s (×12 votes) | 0.06 s |
| Questions per second (serial) | ~6 | ~2 | ~2 | ~1.6 | ~0.7 | ~16 |
| Memory | ~36 GB peak (measured) | ~53 GB peak (measured) | 9.3 GB weights | 14.8 GB weights | 9.1 GB weights | off-box sidecar (~3 GB) |
| Cost basis | input only, ~750 tokens/question | same | same | same | 13× requests | local, no tokens |

- Both Clefs fit comfortably on the Mac Studio (256 GB) **alongside** the chat models we
  already run (peaks measured in the full-benchmark run). The OpenJev checkpoints are
  physically smaller than their Clef rivals (weights listed; resident peaks during serving
  weren't separately captured this week — flagged for the next run), and the encoder interns
  run off the GPU in their own little side-cars.
- Important operational finding: **running several models at once doesn't add capacity** —
  throughput stays pinned and requests just queue up. We now schedule the heavy checks serially
  and design around ~6 questions/second per model, which changes how we size a future rollout.

## 8. Case files — what a real test case looks like

Numbers aside, here is what one exam question actually looks like: the same question asked
about two nearly identical documents — one that contains the answer, one that doesn't. (Every
example below is verbatim from this week's runs.)

### Case 1 — the easy win (history)

> **Question:** *"Who was in charge of the papal army in the War of Barbastro?"*
>
> **Document A (contains the answer):**
> *"The legendary religious zeal of the Normans was exercised in religious wars long before the
> First Crusade… In 1064, during the War of Barbastro, **William of Montreuil led the papal
> army** and took a huge booty."*
>
> **Document B — same document, answer sentence surgically removed:**
> *"…In 1018, Roger de Tosny travelled to the Iberian Peninsula to carve out a state for
> himself from Moorish lands, but failed. **By then however, there were already Norman
> mercenaries serving as far away as Trebizond and Georgia.**"*

| Gate | Doc A (answer present) | Verdict | Doc B (answer removed) | Verdict |
|---|---|---|---|---|
| Clef-Flash | **93%** sure | ✅ answer | **11%** sure | ⛔ block |
| Clef-27B | **97%** sure | ✅ answer | **2%** sure | ⛔ block |
| Chat judge | **100%** sure | ✅ answer | **0%** sure | ⛔ block |

All three read the pair like humans would: *the answer is in there* vs *it isn't*. This is
the gate working as designed — and the two documents are **the same length and topic** on
purpose, so no model can cheat by counting words.

### Case 2 — the trap, and the save (health question)

> **Question:** *"Is it possible to become physically addicted to weed?"*
> **Correct answer:** *"No. Marijuana is not physically addictive, but it is possible to
> become psychologically addicted to it."*
>
> **Document A (sufficient):** *"**No.** Marijuana is not physically addictive, but it is
> possible to become psychologically addicted to it."*
>
> **Document B (insufficient — tricky!):** *"**You absolutely could.** Marijuana is not
> physically addictive, but it is possible to become psychologically addicted to it."*

| Gate | Doc A | Doc B | What happened |
|---|---|---|---|
| Clef-Flash | 90% ✅ | 73% | correct, twice — and flagged Doc B as *contradictory* |
| Chat judge | 95% ✅ | 95% | **gave both docs the identical score** — fooled |
| Two-step combo | ✅ | ⛔ | judge's 95% translated to *"really only ~51% trustworthy"* → correctly blocked |

This single pair is the project in miniature: one word different ("No" vs "You absolutely
could"), the chat judge couldn't tell them apart, and the translation layer we built this week
(§6) is what caught the trap. Clef-Flash even *diagnosed* why — it labelled Doc B
"contradictory," not just "insufficient."

### Case 3 — what "coverage" costs look like in practice

> **Question:** *"…determine if the value of an input is less than what variable?"*
> (a computer-science question about factorization)

| Gate | Confidence | Threshold | Outcome |
|---|---|---|---|
| Clef-Flash | 74% | 89% | ⛔ blocked — "not confident enough" |
| Clef-27B | 91% | 93% | ⛔ blocked — *so close* |
| Chat judge | 100% | — | saw it clearly, and the combo gate let it answer |

This is where auto-answer traffic gets lost today: all three models sort of knew, no single
gate crossed its bar. Whether stronger combos can safely harvest these near-misses is exactly
what next week's work targets.

## 9. Recommendation & next week

**Ship Clef-27B as the production safety gate.** On the full exam it wins on accuracy
(AUROC 0.87) and trustworthiness together; its lead over every alternative is now measured on
2,224 test questions and is statistically decisive, and the infrastructure supports it today.
Keep Clef-Flash as the budget/lower-latency fallback, and park the two-step combo — correctly
engineered now (it is literally the best-calibrated option on the board), but it can't yet
justify its complexity. The full exam also delivered a new, clean disqualifier: the chat
judge run alone **exceeds our 5% safety budget (5.5%)** — its headline coverage cannot be
bought at the risk we accept.
**The eight-model search (§10) did not overturn this — it strengthened it.** The one rival
that judged *better* than Clef-27B broke the same safety budget the chat judge did; every
other rival was further back. Clef-27B remains the only contestant that wins quality and
obedience together. One concrete upgrade did fall out of the search: if/when we run the
budget-compliant fast tier, **OpenJev-Flash-9B beats Clef-Flash by a statistically decisive
margin** (+0.03 AUROC) at a moderate latency cost — a swap worth pricing (§10).

1. **Biggest opportunity:** every option today answers only ~1 in 5 questions it could safely
   answer. Next we test a purpose-trained "sufficiency probe" that could lift that substantially
   — coverage, not safety, is where the business value is unlocked.
2. Make the chat judge *earn* its escalations (a smarter "thinking mode" for hard cases only).
3. Build a labelled contradiction test set — the gate's second job (catching documents that
   disagree with each other) has promising early signs but no proof yet.
4. Escalate the throughput ceiling to the platform team: today the checkpoint caps at ~6
   questions/second per model; a second replica or an upstream fix (batching) unblocks rollout
   scale.

## 10. The talent search: eight new models, one verdict

This week we asked: *is there a better gate we haven't tried?* We searched Hugging Face (the
public library of AI models) for every open-source model that promises the same
"one-glance sufficiency judgement" our Clef models do, vetted licenses, and put every one
that could run on our machine through the identical exam, identical 5% safety budget,
identical statistics.

![The whole field](results/report-figures/merged_leaderboard.png)

**How to read this chart:** every model we have ever tested, ranked by raw judgement quality
(AUROC — see §5), with its honest statistical range shown as a black whisker. The red dashed
line is our incumbent Clef-27B's score. A model only truly beats it if its *whole whisker*
sits to the right of that line.

**What the field said:**

- **One rival judged better — and failed the same test that sank the chat judge.**
  OpenJev-27B scored a genuinely higher AUROC (0.889 — a decisive, significant win over
  Clef's 0.87). But its confidence runs too hot at the top: at *every* possible strictness
  setting it waves through 7.3% unsafe answers — over our 5% budget, no exceptions. A gate
  that judges brilliantly but can't be trusted to say "no" is not a gate. **The crown
  stands.**
- **A genuine upgrade for the budget tier: OpenJev-Flash-9B.** It *does* stay inside the
  5% budget, and beats Clef-Flash decisively (0.860 vs 0.830). If we keep a fast tier,
  this is the one to price — with a license check first (its bigger sibling carries a
  non-commercial license; that one was tested strictly as research and never considered
  for production).
- **The cheap interns are real but junior.** The two Laya models and Von run in
  0.04–0.13 seconds — the cheapest AI options on the board — and Laya-typed measurably
  beats the no-AI checklist (a significant +0.02). But none comes close to a Clef. Von
  passed the safety test by refusing to approve *anything* — safe, useless.
- **A curiosity with a lesson:** Tev is a chat model trained to commit to snap answers.
  Quizzed 12 times per case and averaged, it scored 0.805 — respectable, but it pays full
  chat-model speed (1.5 seconds — 9× our junior) for mid-table accuracy.

The meta-lesson for the roadmap: **raw judgement quality is no longer scarce — reliable
self-restraint is.** Two of three high-accuracy rivals self-destructed on the calibration
test that Clef passes. Our next big lever (§9's sufficiency probe) should be judged on both.

---

*Appendix — full technical detail for anyone who wants it:* `FINDINGS.md` (14 sections of
measurements and methodology; §13–§14 are this week's model search), `REPORT.md`
(model-by-model summary incl. the merged 11-arm leaderboard), `PLAN-FULL-RUN.md` and
`PLAN-SYSTEMONE-SWEEP.md` (how the runs were planned vs how they went), raw run data in
`results/full-benchmark-v1/` (13,900 predictions) and `results/sweep-full-v1/` (16,680),
the merged tables in `results/merged-leaderboard-v1/`, all charts reproducible via
`scripts/make_report_figures.py`, `scripts/make_exec_figures.py`,
`scripts/make_roc_explainer.py` and `scripts/make_merged_figures.py`.*
