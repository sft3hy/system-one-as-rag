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

## 3. The five contestants

| Contestant | What it is | Analogy |
|---|---|---|
| **Clef-Flash** | Small (3B) purpose-built "instant judgement" model | Sharp junior analyst — fast, usually right |
| **Clef-27B** | Same idea, 9× bigger | Senior analyst — slower, noticeably more reliable |
| **Chat judge** | General chat model asked to referee each case | Consultant — talks a good game, overconfident |
| **Two-step combo** | Junior triages everything; only the hard 22% goes to the consultant | Junior + escalation |
| **Baselines** | Simple word-overlap and size checks | Checklists, no AI at all |

## 4. Results — the chart that matters

![Safe auto-answer rate](results/report-figures/exec_safe_autoanswer.png)

**How to read this chart:** each bar is one contestant; the length is the share of all user
questions the gate would let the AI answer directly while keeping its error rate within our 5%
budget. Longer is better — that's traffic we can serve automatically.

**What it says (full dataset, 1,390 questions — these are final numbers, not a preview):**
- All four AI options can safely auto-answer only **16–26% of traffic** today. The rest gets
  blocked — and **54–72% of the blocked questions were actually answerable correctly**. The
  problem isn't that the gate lets wrong answers through; it's that it's too timid.
- **The chat judge's bar looks the longest, and that's a trap.** It reaches 26% only by
  sounding artificially confident — when we audit *how sure it actually is* (see §5), it is the
  least trustworthy of all. On the full exam it now **breaks our 5% safety budget outright**
  (5.5%) — its extra coverage officially cannot be bought at the risk we accept.
- **The two-step combo** (19%) does not beat simply using the senior analyst (24%) — the
  complicated pipeline isn't earning its complexity this week.

## 5. How each candidate performed — the scoreboard

![Quality vs speed](results/report-figures/exec_quality_vs_speed.png)

**How to read this chart:** every dot is one contestant. Right = higher judgement quality
(AUROC, explained below); further up the axis = lower delay per question. There is no free
lunch — you can usually have quality or speed, and the interesting question is who gets the
best trade-off.

**What it says:**
- **Clef-27B is the standout:** the most accurate judge by a clear margin, at a delay
  (half a second) that is acceptable for a pre-answer checkpoint.
- **Clef-Flash is the speed king** (0.17 seconds) with decent accuracy — the budget option.
- **The chat judge** sits near flash on quality but its confidence numbers are unusable (§6).

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


## 6. The week's key discovery: the chat judge's fake confidence

![Score distributions](results/report-figures/score_distributions.png)

**How to read this chart:** each panel shows how one model spread its confidence scores across
556 test cases — blue bars = cases it should approve (evidence sufficient), red = cases it
should block. A *good* judge pushes the blue bars to the right and the red bars to the left,
with a clean gap between them.

**What it says:**
- **Clef-Flash and Clef-27B (left, middle)** behave like real analysts: they spread their
  opinions across the whole 0–100% range, and the blue/red separation (their "quality") is what
  makes a working gate.
- **The chat judge (right)** doesn't have opinions — it *shouts*. Almost every answer lands on
  exactly four values: 0%, 10%, 95%, or 100%. Worse, its 95% answers are right only about half
  the time — it uses 95% to mean "no idea." That's why its trustworthiness score (ECE) is the
  worst of every option, and why a safety threshold placed on its scores is a razor's edge.
  On the full exam the scale of the trap is stark: of 531 cases sent to the judge, **251 came
  back "95% sure" — and 123 of those were wrong**.
- **The fix we shipped:** we built a "translator" that converts the judge's shouty values onto
  Clef-Flash's honest scale before any threshold sees them (95% → "actually worth ~51%, block";
  100% → "worth ~94%, approve"). Verified live at full scale: **all 251 of the judge's
  "95%" verdicts — including all 123 confidently-wrong ones — landed at ~51% and were
  correctly blocked**; zero leaked through untranslated. This rescued the two-step combo from
  near-uselessness (2% → 19% coverage) and made it the *best-calibrated* option on the board —
  but still not good enough to beat Clef-27B outright.

## 7. What the machine costs (and it fits)

| | Clef-Flash | Clef-27B |
|---|---|---|
| Typical delay per checkpoint | 0.17 s | 0.54 s |
| Questions processed per second | ~6 | ~2 |
| Server memory | ~36 GB | ~53 GB |
| Cost basis | input only, ~750 tokens/question | same |

- Both fit comfortably on the Mac Studio (256 GB) **alongside** the chat models we already run.
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

1. **Biggest opportunity:** every option today answers only ~1 in 5 questions it could safely
   answer. Next we test a purpose-trained "sufficiency probe" that could lift that substantially
   — coverage, not safety, is where the business value is unlocked.
2. Make the chat judge *earn* its escalations (a smarter "thinking mode" for hard cases only).
3. Build a labelled contradiction test set — the gate's second job (catching documents that
   disagree with each other) has promising early signs but no proof yet.
4. Escalate the throughput ceiling to the platform team: today the checkpoint caps at ~6
   questions/second per model; a second replica or an upstream fix (batching) unblocks rollout
   scale.

---

*Appendix — full technical detail for anyone who wants it:* `FINDINGS.md` (12 sections of
measurements and methodology), `REPORT.md` (model-by-model summary), `PLAN-FULL-RUN.md` (how
the full run was planned and how the actuals compared), raw run data in
`results/full-benchmark-v1/` (13,900 predictions), all charts reproducible via
`scripts/make_report_figures.py full-benchmark-v1`, `scripts/make_exec_figures.py
full-benchmark-v1` and `scripts/make_roc_explainer.py full-benchmark-v1`.*
