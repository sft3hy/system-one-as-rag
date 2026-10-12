"""Shared case and result types.

Every dataset adapter normalises into ``Case``; every strategy emits
``Decision``. Keeping these narrow is what makes strategies interchangeable:
a strategy never sees a dataset, and a dataset never sees a model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


class FailureType(str, Enum):
    """Why a context is not sufficient. ``NONE`` means it is sufficient."""

    NONE = "none"
    MISSING_FACT = "missing_fact"
    BROKEN_MULTI_HOP = "broken_multi_hop"
    AMBIGUOUS = "ambiguous"
    CONTRADICTORY = "contradictory"
    IRRELEVANT = "irrelevant"


class TerminalState(str, Enum):
    """Explicit outcomes so a correct abstention is never read as a failure."""

    ANSWERED = "ANSWERED"
    ABSTAINED_INSUFFICIENT_CONTEXT = "ABSTAINED_INSUFFICIENT_CONTEXT"
    ABSTAINED_CONTRADICTION = "ABSTAINED_CONTRADICTION"
    RETRIEVAL_EXHAUSTED = "RETRIEVAL_EXHAUSTED"
    ERROR = "ERROR"


# Quadrant: sufficiency x parametric knowledge. Derived, never trusted from
# upstream files: SufficiencyBench's README and metadata.json disagree about
# which label means what, so we recompute it from `sufficient`/`model_confident`.
QUADRANTS: dict[tuple[bool, bool], str] = {
    (True, True): "Q1_sufficient_knows",
    (True, False): "Q2_sufficient_unknown",
    (False, True): "Q3_insufficient_knows",
    (False, False): "Q4_insufficient_unknown",
}


@dataclass(frozen=True)
class Case:
    """One (question, context) pair with a gold sufficiency label."""

    id: str
    question: str
    context: str
    sufficient: bool
    split: str = "test"
    source: str = "custom"
    group_id: str | None = None
    """Questions sharing a group_id are matched variants of the same question.
    Thresholds tune on val groups; bootstraps resample groups, not rows."""

    question_type: str | None = None
    quadrant: str | None = None
    required_facts: tuple[str, ...] = ()
    reference_answer: str | None = None
    failure_type: FailureType | None = None
    model_knows_answer: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def group(self) -> str:
        return self.group_id or self.id

    @property
    def chars(self) -> int:
        return len(self.question) + len(self.context)

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failure_type"] = self.failure_type.value if self.failure_type else None
        return d

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> Case:
        ft = d.get("failure_type")
        known = set(Case.__dataclass_fields__)
        passthrough = {
            k: v for k, v in d.items()
            if k in known and k not in ("failure_type", "required_facts")
        }
        return cls(
            **passthrough,
            required_facts=tuple(d.get("required_facts") or ()),
            failure_type=FailureType(ft) if ft else None,
        )


@dataclass(frozen=True)
class Decision:
    """A strategy's output for one case.

    ``p_sufficient`` is the primary gate score and must be a real probability,
    not a binarised label — calibration and threshold sweeps depend on it.
    ``p_conflicting`` is the independent second-tier score (see the two-noul
    contract in strategies/sufficiency.py).
    """

    case_id: str
    strategy: str
    p_sufficient: float
    p_conflicting: float | None = None
    label: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempts: int = 1
    error: str | None = None
    contract_version: str | None = None
    questions_in_request: int | None = None
    # strategy-specific timing/diagnostic detail (decode_ms, gen_tps,
    # reasoning_chars, truncated, salvaged, requests, ...)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateAction:
    case_id: str
    strategy: str
    action: TerminalState
    p_sufficient: float
    p_conflicting: float | None
    tier: str
    """'fast' = resolved at the decision-model tier, 'judge' = escalated."""

    latency_ms: float
    extra_retrievals: int = 0
    correct_action: bool | None = None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)
