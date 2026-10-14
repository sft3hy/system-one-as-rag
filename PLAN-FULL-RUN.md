# Plan: full-dataset benchmark run (`full-benchmark-v1`)

**Status:** ready to execute — config written at `configs/full-benchmark.yaml`, dry-run validated.
**Dataset:** complete SufficiencyBench report set: val 278 questions + test 1,112 questions
= **1,390 groups = 2,780 conditions per arm** (vs 300/split on capped runs so far).

## Why every run to date is capped, and what "full" adds

All five runs so far used `limit: 100–150` — deliberate: calibrate cost before paying it
(§10), and iterate fast on the cascade fix (§11). The capped runs establish every number in
the leaderboards, but at n=300/split the confidence intervals are ±0.02–0.03 AUROC wide, and
some claims are explicitly flagged "inside ±1 SE, cannot distinguish" (§10.2). The full run:

- **Halves CI widths** on the headline comparisons (n→556 val / 2,224 test): settles
  judge-vs-flash coverage (0.295 vs 0.265, currently tied) and cascade-vs-flash AUROC
  (−0.010, currently tied).
- **Densifies the calibration ladder**: the isotonic fit on band cases goes from ~69 points
  to ~500+, removing the plateau that currently maps every judge answer onto two cliffs.
- **Turns projections into measurements**: coverage/latency/QPS at 4–8× the sample, on the
  full question-type mix (multi_hop 44%, subjective 18%) instead of a capped prefix that
  over-represents whichever sources come first in the file.

## Timing budget (from measured `wall_serial_s`, FINDINGS §10 — not summed latency)

| arm | wall (report set, serial) |
|---|---|
| lexical + question_only + context_length | < 1 min total |
| clef-flash | ~12 min |
| qwen38-judge | ~34 min |
| cascade (flash + ~22% judge) | ~21 min |
| clef-27b | ~39 min |
| **whole run, arms serial** | **~85–95 min** |

+ buffer for three cold model loads (~15 s reload + first-split warmup each, §10.1): budget
**~2 h wall, unattended**. Arms run serially by design — concurrency on this box is queueing,
not capacity (§6), and co-tenant compute inflates Flash 4.4× (§10).

## Procedure

```
# 0. hygiene (5 min)
git status                        # working tree clean; configs/full-benchmark.yaml committed
curl -H "Authorization: Bearer $OMLX_API_KEY" localhost:8000/v1/models   # server up, key works
.venv/bin/python scripts/probe_server.py                                  # decision endpoint responds
vm_stat | head -3                 # >= ~60 GB free before 27B cold-load (§7: server peak 53 GB)

# 1. offline validation (seconds, no server, stub arm)
.venv/bin/sufficiency-lab -c configs/full-benchmark.yaml --dry-run && rm -rf results/full-benchmark-v1-dry

# 2. THE RUN — ~2 h wall, unattended, arms serial in config order (cheapest first,
#    so flash — not the headline 27B arm — eats the first-after-cold-load warmup)
nohup .venv/bin/sufficiency-lab -c configs/full-benchmark.yaml \
      > results/full-run.log 2>&1 &

# 3. checkpoint reviews (30 s each, at ~25/50/75 min)
wc -l results/full-benchmark-v1/predictions/*.jsonl   # row counts climbing, errors: 0
tail -3 results/full-run.log

# 4. on completion
cat results/full-benchmark-v1/reports/report.md
.venv/bin/python scripts/estimate_full_run.py results/full-benchmark-v1   # measured vs projected
```

What stays fixed and why: `limit: null`, thresholds tune on **val only**, test reported once
at the val-tuned threshold, seed 13 and bootstrap 2000 unchanged so CI widths are directly
comparable across runs, `concurrency: [1, 4, 8]` re-measured at full sample, cascade loads
`calibration/cascade-flash-judge-v1.json` (runner refuses to run it uncalibrated).

## What this run must decide (its success criteria)

| open question (FINDINGS) | capped-run status | settles at n=2,780 |
|---|---|---|
| judge vs flash AUROC | tied (−0.001, CI ±0.023) | CI ≈ ±0.011 → tie or real order |
| judge vs 27B coverage on val | 0.295 vs 0.265, inside 1 SE | CI ≈ ±0.037 → decidable |
| cascade vs flash AUROC cost | −0.010, CI ±0.017 | CI ≈ ±0.008 → pay-for-escalation cost known |
| deploy verdict: 27B | from n=300 | confirmed or overturned, with tighter bands |
| calibration ladder resolution | ~69 band points, 2-cliff mapping | ~500 band points → finer judge mapping |

Secondary outputs to fold into docs afterwards: tokens/case on the full question mix
(ELI5's subjective questions skew it), per-question-type AUROC slices (multi_hop is 44% of
the full set vs a heavier factual tilt in capped prefixes), and the cold-load + warmup
deltas at real arm lengths.

## Risks and mitigations

- **Judge parse failures.** Held 0% under the fix at n=2,780 (timing run); retries: 2 in
  config; monitor `errors` in the log at checkpoints — any nonzero stops the run at review.
- **Disk.** Predictions ≈ 4 arms × 5,560 rows × ~0.5 KB ≈ 90 MB — negligible.
- **Cold loads.** Three (flash, judge, 27B), ~15 s reload + first-split warmup each (§10);
  buffer is inside the 2 h budget; arms serial so nothing collides.
- **Interruption.** No resume — if the box must be used mid-run, kill it and restart;
  predictions are append-safe per arm but metrics expect complete splits.
- **Do not** run other GPU work concurrently — this is the single biggest measurement
  corruptor measured on this machine (4.4× latency inflation).

## Effort summary

~5 min hygiene + **~2 h unattended run** + ~1 h numbers-and-charts afterwards
(re-run `make_report_figures.py` / `make_exec_figures.py`, refresh REPORT.md + WEEKLY-REPORT
tables). Nothing new to build — the config exists, dry-run validates, and every number that
made the plan is already measured.
