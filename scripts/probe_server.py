#!/usr/bin/env python
"""Live probe of omlx's decision endpoint before any experiment runs.

Answers, in order:
  1. Is the server up and is the API key valid?
  2. Which models does it serve, and which are model_type == "decision"?
  3. Does /v1/systemone answer the two-noul contract, and what is the raw
     response shape (so the adapter is written against bytes, not prose)?
  4. If a decision model is loaded: cost of 1 vs 3 questions in one request,
     and the p-shift when the same questions are split across requests.

Run:  uv run python scripts/probe_server.py [--model clef-flash-oQ4e]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sufficiency_lab.config import OmlxConfig  # noqa: E402
from sufficiency_lab.contract import DEFAULT_QUESTIONS  # noqa: E402
from sufficiency_lab.omlx_client import OmlxClient, OmlxError  # noqa: E402

STATE = {
    "question": "Which city hosted the conference?",
    "retrieved_context": "The conference was held in Denver in 2025, drawing "
    "four thousand attendees to the downtown convention centre.",
}

NEGATIVE = {
    "question": "Which city hosted the conference?",
    "retrieved_context": "The organisers met in Vienna in 2019 to agree the "
    "criteria for selecting future venues.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()

    cfg = OmlxConfig(base_url=args.base_url)
    print(f"api key: {'found' if cfg.resolved_key() else 'MISSING'}")
    client = OmlxClient(cfg)

    try:
        models = client.list_models()
        print(f"models ({len(models)}): {models}")
    except OmlxError as exc:
        print(f"FAIL list_models: {exc}")
        return 1

    status = client.model_status()
    decisions = [m for m in status if m.get("model_type") == "decision"]
    print(f"decision models: {[m.get('id') or m.get('name') for m in decisions] or 'NONE'}")
    for m in status:
        print(f"  - {m.get('id') or m.get('name')}: type={m.get('model_type')} "
              f"loaded={m.get('loaded')} ctx={m.get('max_context_length')}")

    model = args.model or next(
        iter([m.get("id") or m.get("name") for m in decisions] or []), None)
    if not model:
        print("\nNo decision model loaded. Load one, e.g.:")
        print("  hf download Jundot/clef-flash-oQ4e --local-dir ~/models/Jundot/clef-flash-oQ4e")
        print("\nProbe of a *chat* model on /v1/systemone should return 400 pointing "
              "to this endpoint. Testing that now with the first chat model:")
        chat = next(iter(models), None)
        if chat:
            try:
                client.systemone(chat, STATE,
                                {"sufficient": {"type": "noul",
                                             "instructions": DEFAULT_QUESTIONS.sufficient_instructions}})
                print(f"  unexpected: {chat} answered on /v1/systemone")
            except OmlxError as exc:
                print(f"  as documented: {exc}")
        return 0

    qs = DEFAULT_QUESTIONS.questions(include_diagnostic=True)
    print(f"\n--- contract: {DEFAULT_QUESTIONS.version} on {model} ---")
    for label, state in (("sufficient-context", STATE), ("insufficient-context", NEGATIVE)):
        lat = []
        last = None
        for _ in range(args.reps):
            r = client.systemone(model, state, qs, truncate=True)
            lat.append(r.latency_ms)
            last = r
        a = last.answers
        print(f"{label}:")
        print(f"  sufficient noul = {a['sufficient'].noul}")
        print(f"  conflicting noul = {a['conflicting'].noul}")
        print(f"  diagnosis = {a.get('diagnosis').choice if a.get('diagnosis') else None} "
              f"{(a.get('diagnosis').probabilities or {}) if a.get('diagnosis') else {}}")
        print(f"  usage = {last.usage}")
        print(f"  latency ms: p50={statistics.median(lat):.1f} "
              f"min={min(lat):.1f} max={max(lat):.1f} over {args.reps} reps")

    # 1 vs 3 questions in one request: the load-bearing batching claim
    print("\n--- batching cost (same state, fewer questions) ---")
    for label, questions in (
        ("1 noul", {"sufficient": qs["sufficient"]}),
        ("2 noul", {k: qs[k] for k in ("sufficient", "conflicting")}),
        ("2 noul + choice", qs),
    ):
        lat = []
        for _ in range(args.reps):
            r = client.systemone(model, STATE, questions, truncate=True)
            lat.append(r.latency_ms)
        print(f"  {label:>18}: p50={statistics.median(lat):.1f} ms "
              f"tokens={r.usage.get('prompt_tokens')}")

    print("\nraw response sample:")
    r = client.systemone(model, NEGATIVE, qs, truncate=True)
    print(json.dumps(r.raw, indent=2)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
