"""Dataset adapters. Everything normalises into Case; nothing invents labels.

SufficiencyBench layout (verified against inab-lab/SufficiencyBench@master):
a list of questions, each with two matched ``conditions`` — one sufficient, one
built by surgically removing the answer sentence and substituting a topically
similar non-answering sentence from the *same* document (mean length ratio
0.991). That pairing is why ``group_id`` is the question id everywhere: a
threshold tuned on a sufficient variant must not be scored against its own
insufficient twin.

Quadrants are recomputed, not imported: the repo's README table and its
metadata.json disagree about which label means "sufficient + knows", so we
derive both from the per-condition ``sufficient`` / ``model_confident`` fields
and record the derivation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from .schema import Case, FailureType

# SufficiencyBench's own files disagree, so the derivation is explicit here:
# positive quadrant axes are (sufficient, model_confident).
QUADRANT_FROM_FIELDS = {
    (True, True): "Q1_sufficient_knows",
    (True, False): "Q2_sufficient_unknown",
    (False, True): "Q3_insufficient_knows",
    (False, False): "Q4_insufficient_unknown",
}


def _quadrant(sufficient: bool, model_confident: Any) -> str | None:
    if model_confident is None:
        return None
    return QUADRANT_FROM_FIELDS.get((bool(sufficient), bool(model_confident)))


def _failure_type(qtype: str | None, sufficient: bool) -> FailureType:
    """Conservative: the benchmark states *that* a fact was removed, not which.

    Marking multi-hop removals as BROKEN_MULTI_HOP is supported by the
    construction notes; everything else insufficient is MISSING_FACT. Never
    assign CONTRADICTORY here — nothing in this dataset is labelled as such,
    and inventing that gold would corrupt the second-tier calibration.
    """
    if sufficient:
        return FailureType.NONE
    if qtype == "multi_hop":
        return FailureType.BROKEN_MULTI_HOP
    return FailureType.MISSING_FACT


def load_sufficiencybench(root: Path, split: str = "test") -> list[Case]:
    """root is the directory containing train/ val/ test/ data.json."""
    path = Path(root) / split / "data.json"
    if not path.exists():
        raise FileNotFoundError(f"SufficiencyBench split not found: {path}")
    rows = json.loads(path.read_text())
    cases: list[Case] = []
    for q in rows:
        qid = str(q["id"])
        qtype = q.get("question_type")
        knows = q.get("model_knows_answer")
        for cond in q.get("conditions", []):
            sufficient = bool(cond["sufficient"])
            confident = cond.get("model_confident", knows)
            cases.append(
                Case(
                    id=str(cond.get("condition_id") or f"{qid}_{cond.get('sufficient')}"),
                    question=q["question"],
                    context=cond["context"],
                    sufficient=sufficient,
                    split=split,
                    source="sufficiencybench",
                    group_id=qid,
                    question_type=qtype,
                    quadrant=_quadrant(sufficient, confident),
                    reference_answer=q.get("gold_answer"),
                    failure_type=_failure_type(qtype, sufficient),
                    model_knows_answer=None if knows is None else bool(knows),
                    metadata={
                        "context_length": cond.get("context_length", len(cond["context"])),
                        "expected_behavior": cond.get("expected_behavior"),
                        "closed_book_confidence": q.get("closed_book_confidence"),
                        "failure_label_source": "derived_construction",
                    },
                )
            )
    return cases


def load_sufficientcontext(path: Path, split: str = "test") -> list[Case]:
    """hljoren/sufficientcontext: passages + question + sufficiency rating.

    Only rows with an explicit gold sufficiency flag are kept; unrated rows are
    skipped rather than weakly labelled by the model under test.
    """
    rows = _read_any(path)
    cases: list[Case] = []
    for i, r in enumerate(rows):
        gold = r.get("sufficient", r.get("is_sufficient", r.get("label")))
        if gold is None:
            continue
        if isinstance(gold, str):
            gold = gold.strip().lower() in {"1", "true", "yes", "sufficient"}
        context = r.get("context") or r.get("passage") or r.get("evidence") or ""
        if isinstance(context, list):
            context = "\n\n".join(str(c) for c in context)
        qid = str(r.get("id") or f"sufficientcontext_{i}")
        cases.append(
            Case(
                id=f"{split}:{qid}",
                question=str(r.get("question") or r.get("query") or ""),
                context=context,
                sufficient=bool(gold),
                split=split,
                source="sufficientcontext",
                group_id=qid,
                question_type=r.get("question_type"),
                reference_answer=str(r.get("answer") or r.get("gold_answer") or "") or None,
                failure_type=_failure_type(r.get("question_type"), bool(gold)),
                metadata={"failure_label_source": "none"},
            )
        )
    return cases


def load_rgb(path: Path, split: str = "test") -> list[Case]:
    """chen700564/RGB: the negative-rejection subset is the relevant axis.

    'query + docs list + answer' rows: a row counts as insufficient when its
    gold answer is the unanswerable sentinel. Anything else is skipped.
    """
    sentinel = re.compile(r"^(unknown|unanswerable|i cannot|cannot answer|n/?a)\.?$",
                          re.IGNORECASE)
    rows = _read_any(path)
    cases: list[Case] = []
    for i, r in enumerate(rows):
        answer = str(r.get("answer") or r.get("gold_answer") or "")
        if not answer:
            continue
        unanswerable = bool(sentinel.match(answer.strip()))
        docs = r.get("docs") or r.get("documents") or r.get("positive_passages") or []
        context = "\n\n".join(str(d) for d in docs) if isinstance(docs, list) else str(docs)
        qid = str(r.get("id") or f"rgb_{i}")
        cases.append(
            Case(
                id=f"rgb:{split}:{qid}",
                question=str(r.get("query") or r.get("question") or ""),
                context=context,
                sufficient=not unanswerable,
                split=split,
                source="rgb",
                group_id=qid,
                failure_type=FailureType.MISSING_FACT if unanswerable else FailureType.NONE,
                reference_answer=None if unanswerable else answer,
                metadata={"failure_label_source": "gold_sentinel"},
            )
        )
    return cases


def load_cases(path: Path, split: str = "test") -> list[Case]:
    """Universal JSONL loader for custom cases in the documented schema."""
    out: list[Case] = []
    for raw in _read_any(path):
        raw = dict(raw)
        raw.setdefault("split", split)
        raw.setdefault("source", "custom")
        out.append(Case.from_json_dict(raw))
    return out


ADAPTERS = {
    "sufficiencybench": load_sufficiencybench,
    "fixtures": load_sufficiencybench,  # same layout, offline smoke data
    "sufficientcontext": load_sufficientcontext,
    "rgb": load_rgb,
    "custom": load_cases,
}


def _read_any(path: Path) -> Sequence[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "instances", "examples", "questions"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    raise ValueError(f"unsupported payload in {path}")
