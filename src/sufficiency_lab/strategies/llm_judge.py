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


class LLMJudgeStrategy(BaseStrategy):
    kind = "llm_judge"

    def __init__(
        self,
        cfg: OmlxConfig,
        model: str,
        questions: Questions,
        *,
        name: str | None = None,
        max_tokens: int = 96,
        temperature: float = 0.0,
        use_response_format: bool = True,
    ):
        super().__init__(questions)
        self.model = model
        self.name = name or f"judge_{model}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_response_format = use_response_format
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
        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        parsed = _parse_json(text)
        p_suf = float(parsed.get("sufficient", float("nan")))
        p_con = parsed.get("conflicting")
        meta = {
            "label": "contradictory" if float(p_con or 0) > 0.5 else None,
            "ttft_ms": ttft_ms,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
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
        return json.loads(text)
    except Exception:
        m = _JSON.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    raise ValueError(f"judge returned non-JSON: {text[:120]!r}")
