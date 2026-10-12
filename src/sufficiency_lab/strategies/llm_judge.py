"""Autoregressive judge over /v1/chat/completions.

Latency honesty: this arm pays for prompt processing *and* decode, so its
latency is recorded as ttft + decode time, and the number of output tokens is
kept. Comparing its wall clock against a decision model's single prefill is
the whole point of the two-tier design, so nothing here is trimmed.
"""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

import httpx

from ..config import OmlxConfig
from ..contract import LLM_JUDGE_SCHEMA, Questions
from ..schema import Case
from .base import BaseStrategy

_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _salvage_partial(text: str) -> dict[str, Any] | None:
    """Recover fields from truncated JSON like '{"sufficient": 0.99, "conf'.

    Truncation is a max_tokens artefact, not a model failure; discarding the
    case would silently bias the split toward easy questions the model finished
    talking about, so we keep whatever bounded numbers were actually emitted and
    flag it.
    """
    out: dict[str, Any] = {}
    for key in ("sufficient", "conflicting"):
        m = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if m:
            out[key] = float(m.group(1))
    return out or None


class LLMJudgeStrategy(BaseStrategy):
    kind = "llm_judge"

    def __init__(
        self,
        cfg: OmlxConfig,
        model: str,
        questions: Questions,
        *,
        name: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        use_response_format: bool = True,
        enable_thinking: bool = False,
    ):
        """Thinking is OFF by default, and that is a measured choice.

        A/B on 24 real SufficiencyBench cases, this model, this server:

        | config | failure | p50 | AUROC |
        |---|---|---|---|
        | think + schema, 512 tok | 21% | 1735 ms | 1.000 |
        | **no-think + schema**  |  **0%** | **509 ms** | 0.924 |
        | no-think, no schema    | 67% | 384 ms | 0.500 (chance) |

        Thinking sharpens the negatives (mean p 0.074 vs 0.358) but omlx bills
        thinking against `max_tokens`, so at 512 a fifth of cases spent their
        whole budget reasoning and emitted no JSON — a gate that silently drops
        21% of traffic is not a gate. `response_format` is load-bearing: without
        it the output is prose and discrimination is exactly chance.
        """
        super().__init__(questions)
        self.model = model
        self.name = name or f"judge_{model}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_response_format = use_response_format
        self.enable_thinking = enable_thinking
        headers = {}
        key = cfg.resolved_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=cfg.base_url, headers=headers, timeout=cfg.timeout_s
        )

    def close(self) -> None:
        self._http.close()

    def _score(self, case: Case):
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.questions.judge_prompt(
                case.question, case.context
            )}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.use_response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "sufficiency", "schema": LLM_JUDGE_SCHEMA,
                               "strict": True},
            }
        if not self.enable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}

        start = perf_counter()
        r = self._http.post("/chat/completions", json=body)
        if r.status_code >= 400 and self.use_response_format:
            # server may not accept strict schemas for this model: degrade once
            self.use_response_format = False
            return self._score(case)
        if r.status_code >= 400:
            raise RuntimeError(f"chat/completions HTTP {r.status_code}: {r.text[:200]}")
        ttft_ms = (perf_counter() - start) * 1000.0

        payload = r.json()
        choices = payload.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], dict) else {}
        msg = choice.get("message") or {}
        if not isinstance(msg, dict):
            msg = {}
        text = msg.get("content") or ""
        # reasoning is billed but never scored: the gate reads only the JSON
        reasoning_len = len(msg.get("reasoning_content") or "")
        usage = payload.get("usage") or {}
        parsed = _parse_json(text)
        p_suf = float(parsed.get("sufficient", float("nan")))
        p_con = parsed.get("conflicting")
        meta = {
            "label": "contradictory" if float(p_con or 0) > 0.5 else None,
            # prefer the server's own split timing when it reports it
            "ttft_ms": (usage.get("time_to_first_token") or 0) * 1000.0 or ttft_ms,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "decode_ms": (usage.get("generation_duration") or 0) * 1000.0,
            "gen_tps": usage.get("generation_tokens_per_second"),
            "reasoning_chars": reasoning_len,
            "truncated": choice.get("finish_reason") == "length",
            "salvaged": bool(parsed.get("_salvaged")),
            "missing": parsed.get("missing") or [],
        }
        probs = {"sufficient": p_suf}
        if p_con is not None:
            probs["conflicting"] = float(p_con)
        return p_suf, (float(p_con) if p_con is not None else None), probs, meta

    def warmup(self, cases) -> None:  # includes prompt-cache warm-up
        for case in cases:
            self.predict(case)


def _parse_json(text: str) -> dict[str, Any]:
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            # models occasionally wrap the answer as [{...}]; the first object
            # is the verdict, and discarding the case would bias the split
            obj = next((o for o in obj if isinstance(o, dict)), None)
        if isinstance(obj, dict):
            return obj
    except Exception:
        m = _JSON.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    salvaged = _salvage_partial(text)
    if salvaged:
        return {**salvaged, "_salvaged": True}
    raise ValueError(f"judge returned non-JSON: {text[:120]!r}")
