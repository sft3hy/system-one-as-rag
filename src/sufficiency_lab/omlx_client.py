"""Thin client for omlx's decision endpoint.

Verbatim shape of ``POST /v1/systemone`` per jundot/omlx#4315 (v0.7.1.dev1):

    {"model": ..., "state": str|object|array, "questions": {name: {...}},
     "images": [...], "truncate": bool}

``noul`` -> {"type": "noul", "noul": p_yes}
``choice`` -> {"type": "choice", "choice": k, "probabilities": {...}, "confidence": c}
``score``  -> {"type": "score", "score": i, "probabilities": {...},
               "confidence": c, "legend": {...}}

Decision models are hidden from the chat picker; /v1/chat/completions on one
returns 400. omlx flags them in /v1/models/status as model_type == "decision".
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import OmlxConfig


class OmlxError(RuntimeError):
    pass


@dataclass
class SystemOneAnswer:
    name: str
    type: str
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    legend: dict[str, str] | None = None


@dataclass
class SystemOneResponse:
    model: str
    answers: dict[str, SystemOneAnswer]
    usage: dict[str, Any]
    latency_ms: float
    raw: dict[str, Any]


class OmlxClient:
    def __init__(self, cfg: OmlxConfig):
        self.cfg = cfg
        key = cfg.resolved_key()
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._http = httpx.Client(
            base_url=cfg.base_url, headers=headers, timeout=cfg.timeout_s
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ models

    def list_models(self) -> list[str]:
        r = self._http.get("/models")
        self._raise(r, "list models")
        return [m["id"] for m in r.json().get("data", [])]

    def model_status(self) -> list[dict[str, Any]]:
        """model_type == 'decision' identifies Clef/OpenJev checkpoints."""
        r = self._http.get("/models/status")
        if r.status_code >= 400:
            return []
        try:
            payload = r.json()
        except Exception:
            return []
        if isinstance(payload, dict):
            return payload.get("models", []) or []
        return payload if isinstance(payload, list) else []

    def decision_models(self) -> list[str]:
        return [
            m.get("id") or m.get("name")
            for m in self.model_status()
            if m.get("model_type") == "decision"
        ]

    # ------------------------------------------------------------- systemone

    def systemone(
        self,
        model: str,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any]],
        *,
        truncate: bool | None = None,
        images: list[str] | None = None,
    ) -> SystemOneResponse:
        if state is None:
            raise ValueError("state must not be null")
        if not questions:
            raise ValueError("at least one named question is required")
        body: dict[str, Any] = {
            "model": model,
            "state": state,
            "questions": questions,
        }
        if truncate is not None:
            body["truncate"] = truncate
        if images:
            body["images"] = images

        last_err: Exception | None = None
        for attempt in range(1, self.cfg.retries + 2):
            start = time.perf_counter()
            try:
                r = self._http.post("/systemone", json=body)
            except httpx.HTTPError as exc:  # network flap, retry
                last_err = exc
                time.sleep(min(2.0 * attempt, 5.0))
                continue
            latency_ms = (time.perf_counter() - start) * 1000.0
            if r.status_code in (429, 502, 503) and attempt <= self.cfg.retries:
                last_err = OmlxError(f"{r.status_code}: {r.text[:200]}")
                time.sleep(min(2.0 * attempt, 5.0))
                continue
            self._raise(r, f"systemone({model})")
            payload = r.json()
            return SystemOneResponse(
                model=payload.get("model", model),
                answers=_parse_answers(payload.get("answers") or {}),
                usage=payload.get("usage") or {},
                latency_ms=latency_ms,
                raw=payload,
            )
        raise OmlxError(f"systemone failed after retries: {last_err}")

    @staticmethod
    def _raise(r: httpx.Response, what: str) -> None:
        if r.status_code < 400:
            return
        detail = r.text[:400]
        if r.status_code == 401:
            detail += " (set OMLX_API_KEY or point config at ~/.omlx/settings.json)"
        raise OmlxError(f"{what} -> HTTP {r.status_code}: {detail}")


def _parse_answers(raw: dict[str, Any]) -> dict[str, SystemOneAnswer]:
    out: dict[str, SystemOneAnswer] = {}
    for name, ans in raw.items():
        if not isinstance(ans, dict):
            continue
        probs = ans.get("probabilities")
        out[name] = SystemOneAnswer(
            name=name,
            type=ans.get("type", ""),
            noul=_maybe_float(ans.get("noul")),
            choice=ans.get("choice"),
            score=_maybe_float(ans.get("score")),
            probabilities=(
                {str(k): float(v) for k, v in probs.items()} if probs else None
            ),
            confidence=_maybe_float(ans.get("confidence")),
            legend=ans.get("legend"),
        )
    return out


def _maybe_float(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
