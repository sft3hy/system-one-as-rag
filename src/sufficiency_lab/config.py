"""Config loading. Everything a run needs lives in configs/*.yaml so that
adding a model never means editing the runner."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]

# omlx stores its own server settings here; we read the API key from it so
# the key is never copied into this repo.
Omlx_DEFAULT_SETTINGS = Path.home() / ".omlx" / "settings.json"


def _expand(value: Any) -> Any:
    """Expand ~ and ${ENV} in config strings."""
    if isinstance(value, str):
        value = os.path.expanduser(value)
        return re.sub(
            r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value
        )
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


@dataclass
class OmlxConfig:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str | None = None
    settings_path: Path | None = None
    timeout_s: float = 300.0
    retries: int = 2
    max_concurrency: int = 4
    truncate: bool = True

    def resolved_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        env = os.environ.get("OMLX_API_KEY")
        if env:
            return env
        for cand in (self.settings_path, Omlx_DEFAULT_SETTINGS):
            if cand and Path(cand).exists():
                try:
                    import json

                    key = json.loads(Path(cand).read_text())
                    api = key.get("auth", {}).get("api_key")
                    if api:
                        return api
                except Exception:
                    continue
        return None


@dataclass
class ModelSpec:
    """One arm of the experiment."""

    name: str
    kind: str
    """'system1' | 'llm_judge' | 'nli' | 'heuristic' | 'modernbert'"""

    model: str | None = None
    enabled: bool = True
    questions: str = "sufficiency_v1"
    batch_questions: bool = True
    """True: sufficiency + contradiction in ONE /v1/systemone prefill.
    False: separate requests, for the round-trip cost measurement."""

    max_tokens: int = 64
    gate: str = "fast"
    """'fast' resolves at the decision tier; 'judge' is the escalation tier."""

    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    max_false_sufficient: float = 0.05
    grid: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(1, 100))
    bootstrap: int = 2000
    seed: int = 13

    @property
    def risk_points(self) -> tuple[float, ...]:
        return (0.01, 0.02, 0.05, 0.10)


@dataclass
class LatencyConfig:
    warmup: int = 3
    repeats: int = 1
    concurrency: tuple[int, ...] = (1, 4, 8)
    flush_cache_command: str | None = None
    fast_budget_ms: float = 80.0
    judge_budget_ms: float = 1200.0
    escalate_low: float = 0.35
    escalate_high: float = 0.75
    t_conflict: float = 0.50


@dataclass
class RunConfig:
    run_id: str
    dataset: str = "sufficiencybench"
    data_path: Path = REPO / "data" / "raw"
    out_dir: Path = REPO / "results"
    limit: int | None = None
    splits: tuple[str, ...] = ("val", "test")
    omlx: OmlxConfig = field(default_factory=OmlxConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    models: tuple[ModelSpec, ...] = ()

    def models_for(self, kinds: set[str] | None = None) -> list[ModelSpec]:
        return [
            m
            for m in self.models
            if m.enabled and (kinds is None or m.kind in kinds)
        ]

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.run_id


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    raw = _expand(yaml.safe_load(path.read_text()) or {})

    run = raw.get("run", {})
    omlx_raw = raw.get("omlx", {})
    omlx = OmlxConfig(
        base_url=omlx_raw.get("base_url", OmlxConfig.base_url),
        api_key=omlx_raw.get("api_key"),
        settings_path=(
            Path(omlx_raw["settings_path"]) if omlx_raw.get("settings_path") else None
        ),
        timeout_s=float(omlx_raw.get("timeout_s", 300.0)),
        retries=int(omlx_raw.get("retries", 2)),
        max_concurrency=int(omlx_raw.get("max_concurrency", 4)),
        truncate=bool(omlx_raw.get("truncate", True)),
    )

    risk_raw = raw.get("risk", {})
    risk = RiskConfig(
        max_false_sufficient=float(risk_raw.get("max_false_sufficient", 0.05)),
        bootstrap=int(risk_raw.get("bootstrap", 2000)),
        seed=int(risk_raw.get("seed", 13)),
    )

    lat_raw = raw.get("latency", {})
    latency = LatencyConfig(
        warmup=int(lat_raw.get("warmup", 3)),
        repeats=int(lat_raw.get("repeats", 1)),
        concurrency=tuple(int(c) for c in lat_raw.get("concurrency", (1, 4, 8))),
        flush_cache_command=lat_raw.get("flush_cache_command"),
        fast_budget_ms=float(lat_raw.get("fast_budget_ms", 80.0)),
        judge_budget_ms=float(lat_raw.get("judge_budget_ms", 1200.0)),
        escalate_low=float(lat_raw.get("escalate_low", 0.35)),
        escalate_high=float(lat_raw.get("escalate_high", 0.75)),
        t_conflict=float(lat_raw.get("t_conflict", 0.50)),
    )

    reserved = {"name", "kind", "model", "enabled", "questions",
                "batch_questions", "max_tokens", "gate"}
    models = []
    for m in raw.get("models", []):
        # Strategy knobs may be nested under `extra:` or written flat alongside
        # the reserved keys. Both are accepted and merged, so a knob is never
        # silently dropped by a whitelist that forgot a name.
        extra = {k: v for k, v in m.items() if k not in reserved and k != "extra"}
        nested = m.get("extra") or {}
        if not isinstance(nested, dict):
            raise ValueError(f"model {m.get('name')!r}: `extra` must be a mapping")
        extra.update(nested)
        models.append(ModelSpec(
            name=m["name"],
            kind=m["kind"],
            model=m.get("model"),
            enabled=bool(m.get("enabled", True)),
            questions=m.get("questions", "sufficiency_v1"),
            batch_questions=bool(m.get("batch_questions", True)),
            max_tokens=int(m.get("max_tokens", 64)),
            gate=m.get("gate", "fast"),
            extra=extra,
        ))
    models = tuple(models)

    cfg = RunConfig(
        run_id=run.get("id", "run"),
        dataset=run.get("dataset", "sufficiencybench"),
        data_path=Path(run.get("data_path", REPO / "data" / "raw")),
        out_dir=Path(run.get("out_dir", REPO / "results")),
        limit=run.get("limit"),
        splits=tuple(run.get("splits", ("val", "test"))),
        omlx=omlx,
        risk=risk,
        latency=latency,
        models=models,
    )
    if not cfg.models:
        raise ValueError(f"{path}: no models configured")
    return cfg
