"""Fixture generator: SufficiencyBench-shaped data without the download.

Matched pairs, same construction rule as the real benchmark: the answering
sentence is removed and replaced by a topically similar non-answering sentence
of comparable length, so length cannot carry the label.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = [
    {
        "answer_sent": "In 1064, during the War of Barbastro, William of Montreuil led the papal army.",
        "distractor": "By then however, there were already Norman mercenaries serving as far away as Trebizond.",
        "q": "Who was in charge of the papal army in the War of Barbastro?",
        "gold": "William of Montreuil",
        "qtype": "factual",
        "prefix": "The legendary religious zeal of the Normans was exercised in religious wars long before the First Crusade carved out a Norman principality in Antioch. They were major foreign participants in the Reconquista in Iberia.",
    },
    {
        "answer_sent": "The conference was held in Denver in 2025, drawing four thousand attendees.",
        "distractor": "The conference organisers met in Vienna in 2019 to agree on the next venue criteria.",
        "q": "Which city hosted the conference?",
        "gold": "Denver",
        "qtype": "factual",
        "prefix": "The series rotates between North American and European sites on a multi-year cycle.",
    },
    {
        "answer_sent": "Kiran joined the robotics team in 2021, which won the national final that spring.",
        "distractor": "The robotics club has been a student favourite since its founding decade.",
        "q": "Did the team Kirian joined win the national final, and in what year did he join?",
        "gold": "Yes, 2021",
        "qtype": "multi_hop",
        "prefix": "The school funds several STEM clubs from an alumni endowment renewed every five years.",
    },
    {
        "answer_sent": "Model A scores 91 on the gate while Model B scores 64 on the same split.",
        "distractor": "Both models were trained on data collected over several years by different teams.",
        "q": "Which model scores higher on the gate, and by how much?",
        "gold": "Model A, by 27",
        "qtype": "comparative",
        "prefix": "The evaluation harness runs every candidate on identical inputs and records latency.",
    },
    {
        "answer_sent": "Reviewers found the pacing slow but praised the sound design highly.",
        "distractor": "The film ran at several festivals before its wide release date was announced.",
        "q": "How did reviewers feel about the film overall?",
        "gold": "Mixed: slow pacing, strong sound design",
        "qtype": "subjective",
        "prefix": "Audience reception was tracked through festival polls and written reviews.",
    },
]


def build(root: Path) -> None:
    for split, n in (("train", 8), ("val", 6), ("test", 10)):
        rows = []
        for i in range(n):
            f = FIXTURES[i % len(FIXTURES)]
            qid = f"{split}_{i}"
            rows.append({
                "id": qid,
                "question": f["q"],
                "gold_answer": f["gold"],
                "question_type": f["qtype"],
                "model_knows_answer": (i % 4 == 0),
                "closed_book_confidence": 0.4 if i % 4 else 0.9,
                "conditions": [
                    {
                        "condition_id": f"{qid}_suf",
                        "context": f"{f['prefix']} {f['answer_sent']}",
                        "sufficient": True,
                        "model_confident": (i % 4 == 0),
                        "quadrant": "Q1" if i % 4 == 0 else "Q2",
                        "expected_behavior": "answer",
                        "context_length": len(f"{f['prefix']} {f['answer_sent']}"),
                    },
                    {
                        "condition_id": f"{qid}_insuf",
                        "context": f"{f['prefix']} {f['distractor']}",
                        "sufficient": False,
                        "model_confident": (i % 4 == 0),
                        "quadrant": "Q3" if i % 4 == 0 else "Q4",
                        "expected_behavior": "abstain",
                        "context_length": len(f"{f['prefix']} {f['distractor']}"),
                    },
                ],
            })
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        (d / "data.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "fixtures"))
    a = ap.parse_args()
    build(Path(a.out))
    print(f"fixtures written to {a.out}")
