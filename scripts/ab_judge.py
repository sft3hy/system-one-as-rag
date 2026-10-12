"""Judge configuration A/B: is chain-of-thought worth its latency?

Measured, not assumed. The judge arm cost 2.2 s/condition versus 0.167 s for
Clef-Flash, and omlx bills thinking tokens against max_tokens — so some cases
spent their whole budget thinking and emitted no JSON at all (a ~10% failure
rate on the calibration run). Three questions this answers:

  1. Does `enable_thinking: false` remove the truncation failures?
  2. How much latency does thinking actually buy in AUROC?
  3. Does `response_format` alone keep the output schema-valid without thinking?

If no-thinking costs little discrimination, it is the right production default
and thinking belongs only in an offline audit pass. If it costs a lot, the two
tiers must be re-planned around the thinking cost.

Run: uv run python scripts/ab_judge.py [--n 24]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.config import OmlxConfig  # noqa: E402
from sufficiency_lab.contract import LLM_JUDGE_SCHEMA  # noqa: E402
from sufficiency_lab.metrics import discriminate  # noqa: E402
from sufficiency_lab.schema import Case, read_jsonl  # noqa: E402

CONFIGS = {
    "think+schema": {"response_format": True, "thinking": True, "max_tokens": 512},
    "nothink+schema": {"response_format": True, "thinking": False, "max_tokens": 256},
    "nothink+plain": {"response_format": False, "thinking": False, "max_tokens": 64},
    "think+schema-1024": {"response_format": True, "thinking": True, "max_tokens": 1024},
}


def build_body(model, prompt, cfg):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg["max_tokens"], "temperature": 0.0, "stream": False}
    if cfg["response_format"]:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "sufficiency", "schema": LLM_JUDGE_SCHEMA,
                           "strict": True},
        }
    if not cfg["thinking"]:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def parse(text):
    import re
    try:
        o = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            o = json.loads(m.group(0))
        except Exception:
            v = re.search(r'"sufficient"\s*:\s*([0-9.]+)', text)
            return {"sufficient": float(v.group(1))} if v else None
    if isinstance(o, list):
        o = next((x for x in o if isinstance(x, dict)), None)
    return o if isinstance(o, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-oQ4e-mtp")
    a = ap.parse_args()

    key = OmlxConfig().resolved_key()
    run_raw = Path("results/timing-100/raw/sufficiencybench_val.jsonl")
    if not run_raw.exists():
        print("no calibration snapshot; run configs/timing.yaml first")
        return 1
    cases = [Case.from_json_dict(r) for r in read_jsonl(run_raw)]
    # balanced, deterministic sample
    suf = [c for c in cases if c.sufficient][: a.n // 2]
    ins = [c for c in cases if not c.sufficient][: a.n // 2]
    sample = suf + ins

    from sufficiency_lab.contract import DEFAULT_QUESTIONS

    client = httpx.Client(base_url="http://127.0.0.1:8000/v1",
                          headers={"Authorization": f"Bearer {key}"}, timeout=600)
    print(f"{len(sample)} cases, model {a.model}\n")
    print(f"{'config':<20} {'fail':>5} {'p50ms':>8} {'p95ms':>8} {'out_tok':>8} "
          f"{'AUROC':>7} {'mean+':>7} {'mean-':>7}")
    print("-" * 76)
    for label, cfg in CONFIGS.items():
        scores, gold, lats, fails, otok = [], [], [], 0, []
        for c in sample:
            body = build_body(a.model, DEFAULT_QUESTIONS.judge_prompt(c.question, c.context), cfg)
            t0 = __import__("time").perf_counter()
            try:
                r = client.post("/chat/completions", json=body)
                r.raise_for_status()
                payload = r.json()
                ch = (payload.get("choices") or [{}])[0]
                text = ((ch.get("message") or {}).get("content")) or ""
                p = parse(text)
                if p is None or "sufficient" not in p:
                    fails += 1
                    continue
                v = p["sufficient"]
                v = 1.0 if v is True else 0.0 if v is False else float(v)
                scores.append(v)
                gold.append(1 if c.sufficient else 0)
                lats.append((__import__("time").perf_counter() - t0) * 1000.0)
                otok.append((payload.get("usage") or {}).get("completion_tokens") or 0)
            except Exception as exc:
                fails += 1
                if fails == 1:
                    print(f"   first error: {type(exc).__name__}: {str(exc)[:140]}")

        if not scores:
            print(f"{label:<20} {fails:>5} ALL FAILED")
            continue
        y = np.array(gold); p = np.array(scores)
        d = discriminate([Case(id=str(i), question="q", context="c",
                               sufficient=bool(g)) for i, g in enumerate(gold)], scores)
        print(f"{label:<20} {fails:>5} {statistics.median(lats):>8.0f} "
              f"{np.percentile(lats, 95):>8.0f} {statistics.fmean(otok):>8.0f} "
              f"{d['auroc']:>7.3f} {p[y==1].mean():>7.3f} {p[y==0].mean():>7.3f}")
    client.close()
    print("\nRead: 'mean+' should be high and 'mean-' low. A config that fails")
    print("cases cannot be a production gate no matter how fast it is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
