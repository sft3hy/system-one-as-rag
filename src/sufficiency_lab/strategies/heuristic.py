"""Cheap gold-blind baselines.

These exist to answer "how far do we get without a model at all?" and to
prove that a fancier arm is not just re-deriving lexical overlap.
SufficiencyBench matches sufficient/insufficient contexts for length from the
same document, so overlap features are the honest floor here: if a decision
model cannot beat these, it is not adding semantics.

Reference answers are deliberately NOT used: the gate must work before an
answer exists.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ..schema import Case
from .base import BaseStrategy

_WORD = re.compile(r"[a-z0-9]+")
_NUM = re.compile(r"\d[\d,.]*")
_CAPS = re.compile(r"\b[A-Z][a-z]{2,}\b")
_STOP = set(
    """a an the what when where who which whom whose why how is are was were be
    been being do does did of in on at to for with by from as that this these
    those it its and or not no yes what's i you he she they we us my your""".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class LexicalCoverageStrategy(BaseStrategy):
    """IDF-free token + numeric + entity coverage of question in context."""

    name = "lexical_coverage"
    kind = "heuristic"

    def __init__(self, weight_token: float = 1.0, weight_num: float = 1.5,
                 weight_entity: float = 1.2):
        super().__init__()
        self.wt, self.wn, self.we = weight_token, weight_num, weight_entity

    def _score(self, case: Case):
        q_tok, c_tok = _tokens(case.question), set(_tokens(case.context))
        coverage = (
            sum(1 for t in q_tok if t in c_tok) / len(q_tok) if q_tok else 0.0
        )

        q_num = set(_NUM.findall(case.question))
        c_num = set(_NUM.findall(case.context))
        if q_num:
            num_cov = sum(1 for n in q_num if n in c_num) / len(q_num)
        else:
            num_cov = coverage  # nothing numeric asked: fall back to text

        q_ent = {e.lower() for e in _CAPS.findall(case.question)}
        c_ent = {e.lower() for e in _CAPS.findall(case.context)}
        if q_ent:
            ent_cov = sum(1 for e in q_ent if e in c_ent) / len(q_ent)
        else:
            ent_cov = coverage

        raw = (self.wt * coverage + self.wn * num_cov + self.we * ent_cov) / (
            self.wt + self.wn + self.we
        )
        # squash toward (0.02, 0.98): a lexical score is never certainty
        p = 0.02 + 0.96 * raw
        return p, None, {"coverage": coverage, "num_cov": num_cov,
                         "entity_cov": ent_cov}, {"label": "heuristic"}


class QuestionOnlyStrategy(BaseStrategy):
    """Degenerate control: scores with the question alone, context hidden.

    Anything that cannot beat this is reading the question, not the evidence.
    """

    name = "question_only"
    kind = "heuristic"

    def _score(self, case: Case):
        tok = _tokens(case.question)
        rare = sum(1 for t in tok if len(t) > 6) / max(len(tok), 1)
        p = 0.5 + 0.1 * min(rare, 1.0)
        return p, None, {"rare_ratio": rare}, {"label": "prior_only"}


class LengthStrategy(BaseStrategy):
    """Confound probe: does context length alone predict sufficiency?

    SufficiencyBench matches lengths (mean ratio 0.991), so this arm should
    land near chance. If it does not, our paired construction leaked.
    """

    name = "context_length"
    kind = "heuristic"

    def __init__(self, midpoint: float = 600.0, scale: float = 600.0):
        super().__init__()
        self.mid, self.scale = midpoint, scale

    def _score(self, case: Case):
        z = (len(case.context) - self.mid) / self.scale
        p = 1.0 / (1.0 + math.exp(-z))
        return p, None, {"context_chars": len(case.context)}, {"label": "length"}
