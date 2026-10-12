Yes — and I think this is worth building as a standalone benchmark rather than just comparing a few prompts.

The key is to evaluate whether the retrieved evidence is sufficient to answer the question, independently of whether the main LLM happens to know the answer already. Otherwise, a model can appear excellent at context sufficiency simply because it remembers facts from its training data.

I found existing research that gives us a strong starting point. I would use public benchmarks first, then build a controlled evaluation harness to compare your own System 1 decision models against conventional classifiers, LLM judges, and adaptive retrieval strategies.

## 1. Existing benchmarks to start with

[Deeper insights into retrieval augmented generation: The role of sufficient context](https://images.openai.com/static-rsc-4/vcccumgXGhI6Sp4QRmLSDr7clYDSMY2OSG9lwGUeuog9ktnPiLMI24kKXIGL8Qq1B-BsZc9-PPj10s21I9LIUMbKTbNqIgWLnon7J4bs9MbsmUL4CF9M0P7auurSkO8HfYf3o04uuZ34JUO7bDfDr8qHNafHoc5ptZQZRH-L5V0?purpose=inline)

### 1. SufficiencyBench

Start here

A particularly relevant benchmark for your exact research question. It controls for confounds such as context length, entity density, and the model's parametric knowledge. Its 2026 manuscript reports that conventional output-based detectors plateau at 0.66–0.81 AUROC, while hidden-state probes reach 0.91–0.97 AUROC on its evaluated models. These are reported research results, not guarantees of performance on your data.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Code and benchmark: [inab-lab/SufficiencyBench](https://github.com/inab-lab/SufficiencyBench)

[ICLR Poster Sufficient Context: A New Lens on Retrieval Augmented Generation Systems](https://images.openai.com/static-rsc-4/LspWEp4YBfladnyRqKK3kjN11D-5lvrwq25MX2dqMK8oOoXmITWENNdDdQ-Ijy9nCqnEXF28ru1AG15LISHvPEdJKVQ32NhGAbewELes0Lsm7pW-pvDl1IXbWHrPrxkrG8zKISQLbRoHVVTesQuxEXYsb91D0IE6q--foiUlBSE?purpose=inline)

### 2. Sufficient Context

A foundational benchmark and autorating study focused on whether a diligent reader could construct a definitive answer from the supplied text, including multi-hop questions. Its authors report results for an LLM autorater and baselines such as NLI. It is useful for testing evidence completeness rather than generic answer quality.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Code and research: [hljoren/sufficientcontext](https://github.com/hljoren/sufficientcontext)

[Benchmarking Large Language Models in Retrieval-Augmented Generation | by Ananya Joshi | Medium](https://images.openai.com/static-rsc-4/3nt3vLOwa6ME8rBGqL0RDlnGGo2Bf7aRExpprPuFVJ0kdJr9Z9fL86-C9U3xofRq5OWzZvQi91ZwBXHp7VQtXLULlBoKZNk9bDNpedo9tfDm8BS9s7pmo9Lky6YwS4ZEtdjiw3H-kcVj9B_lWJw-_jzav79T3AHNARh6tpQAqws?purpose=inline)

### 3. RGB

A broader RAG benchmark with four important capabilities: noise robustness, negative rejection, information integration, and counterfactual robustness. Negative rejection is especially relevant when the supplied context cannot support an answer.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://doi.org\&sz=32)

DOI

+1



Code and data: [chen700564/RGB](https://github.com/chen700564/RGB)

[Deeper insights into retrieval augmented generation: The role of sufficient context](https://images.openai.com/static-rsc-4/-KJT-qdeIiBpZkkZL7G4bhXovK_WZof0uWg2q5d2i7LQ9iusB-372WNN0trIDiHdPMwpN4QvlGcQk7Lo6taLNJFuO05k4U1Ykl54RvC0fMmqVYxevnuSfTBJcqflUnhmfPK2AtJ8MNkqNqUeWCkf_TLlEtqemaL9YbQ60sxVFlI?purpose=inline)

### 4. RAGTruth

Useful as a complementary evaluation set for checking whether generated answers contain unsupported claims and whether a system refuses incorrectly. It evaluates downstream behavior, not just the sufficiency decision itself.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



Code and data: [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth)

My recommendation: start with SufficiencyBench, add Sufficient Context and RGB, and build a small, controlled benchmark of your own. The public datasets answer slightly different questions, so don't combine their scores into one leaderboard without reconciling their labels and evaluation protocols.

## 2. The architecture I would build

I'd call the project RAG Sufficiency Lab. It should separate three questions that are often conflated:

1. Evidence sufficiency: Does the retrieved context contain all the information needed to answer?
2. Decision quality: Can the gate recognize that reliably, including when the context is misleading?
3. System impact: Does using that gate improve final answer quality enough to justify its latency, cost, and abstentions?

The architecture should support swapping the decision method without changing the dataset, retrieval results, or evaluation logic.

Benchmark cases

Question + retrieved context + gold sufficiency label + answer requirements

### Identical input → interchangeable decision strategies

A. Heuristics

Entity coverage, embeddings, keyword checks

B. NLI

Entailment and contradiction classifiers

C. System 1

Laya or another structured decision model

D. LLM judges

Small local autoregressive model with a fixed rubric

E. Learned sufficiency probe

Fine-tuned ModernBERT classifier and, separately, hidden-state probes where supported

Calibrated sufficiency gate

Probability of sufficient evidence → threshold → answer, retrieve more, or abstain

Answer

Generate with cited evidence

Repair

Retrieve missing evidence

Abstain

No supported answer

Evaluation and experiment tracking

Discrimination, calibration, false answers, false abstentions, latency, memory, retrieval calls, and downstream answer quality.

The most important design choice is that every strategy receives the same exact question and context. Otherwise, you end up benchmarking retrieval quality and decision quality together and cannot tell which component won.

### Where Laya fits

I would absolutely test a self-hosted System 1 model, but I would treat it as a hypothesis, not the assumed winner. I couldn't verify a System 1 model named “CLEF”; if you mean Laya, that is the model family I would evaluate.

There is also directly relevant new research: Fast Models, Slow Evidence evaluates Laya and Jev across agent decisions and reports that the models tie on RAG relevance gating in its evaluation. That makes independent benchmarking especially important rather than assuming non-autoregressive inference automatically yields better decisions.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



The initial candidates should be:

- Laya/System 1: low-latency structured decision baseline.
- ModernBERT classifier: dedicated encoder-based sufficiency detector.
- NLI cross-encoder: score evidence against the answer requirements, including contradiction.
- Local LLM judge: a structured, rubric-based decision.
- Simple heuristic baseline: a useful sanity check.
- Hidden-state probe: a research track, using SufficiencyBench's implementation rather than starting from scratch.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub



A classifier that is less impressive on raw accuracy can still win in production if it is better calibrated and makes fewer dangerous false-positive decisions.

## 3. The experimental protocol

I'd run the benchmark in stages.

| Stage | Experiment                            | What it tells us                                                                           |
| ----- | ------------------------------------- | ------------------------------------------------------------------------------------------ |
| 1     | Public benchmark replication          | Whether our results align with existing work                                               |
| 2     | Same cases, different gate strategies | Which method detects sufficiency best                                                      |
| 3     | Calibration and threshold tuning      | How safely each method can control generation                                              |
| 4     | Controlled retrieval degradation      | Whether gates recognize missing facts, broken multi-hop chains, and contradictory evidence |
| 5     | End-to-end RAG evaluation             | Whether gating improves final answer quality and total cost                                |
| 6     | Robustness and throughput             | Whether results survive domain shifts, longer contexts, concurrency, and model changes     |

Use the official test split for final reporting. Tune thresholds on validation data only. For your own synthetic cases, split by underlying question or source, not by individual context variant; otherwise, near-duplicate examples can leak between training and test.

### The metrics that actually matter

AUROC and AUPRC

Measure discrimination across thresholds. AUPRC is especially useful when insufficient contexts are rare or class balance differs from production.

False-sufficient rate

Among truly insufficient contexts, how often does the gate incorrectly authorize an answer? This is the key risk metric.

False-insufficient rate

Among sufficient contexts, how often does the gate unnecessarily trigger retrieval or abstention?

Calibration and latency

Brier score, expected calibration error, median and p95 latency, throughput, and peak memory. Measure the entire decision path, not just model inference.

Downstream utility

Final answer correctness, unsupported-claim rate, abstention quality, extra retrieval calls, and total time and cost per correctly answered question.

One particularly useful evaluation is risk versus coverage: how many questions can the system answer while keeping unsupported answers below a predefined risk limit? That gives you a practical way to choose the gate threshold.

## 4. Build the benchmark harness

I would keep the benchmark itself independent of any particular inference server. That lets you compare local models, an OpenAI-compatible endpoint, and classifiers using one runner.

### Project structure

```
rag-sufficiency-lab/
├── configs/
│   └── benchmark.yaml
├── data/
│   ├── raw/                 # Public benchmark data
│   ├── processed/           # Normalized, versioned cases
│   └── custom/              # Your own RAG examples
├── src/sufficiency_lab/
│   ├── schema.py            # Shared case and result types
│   ├── datasets.py          # Dataset adapters
│   ├── runner.py            # Identical-input experiment runner
│   ├── metrics.py           # AUROC, FPR, calibration, CIs
│   ├── thresholds.py        # Validation-only threshold selection
│   ├── gating.py            # Answer / retrieve / abstain
│   ├── retrieval.py         # Optional retrieval repair
│   └── strategies/
│       ├── base.py
│       ├── heuristic.py
│       ├── nli.py
│       ├── system1.py
│       ├── llm_judge.py
│       ├── modernbert.py
│       └── hidden_state.py
├── tests/
│   ├── test_metrics.py
│   ├── test_leakage.py
│   └── test_gating.py
├── results/
│   ├── predictions/
│   ├── reports/
│   └── figures/
├── pyproject.toml
└── README.md
```

### Define one universal case format

Every dataset adapter should normalize its data to this format. Keep the original dataset ID and split so the results remain traceable.

```
{
  "id": "example-001",
  "source": "custom",
  "split": "test",
  "question": "Which city hosted the conference?",
  "context": "The conference was held in Denver in 2025.",
  "sufficient": true,
  "required_facts": ["The conference took place in Denver."],
  "reference_answer": "Denver",
  "failure_type": null
}
```

For insufficient examples, use `sufficient: false` and identify why: missing fact, incomplete multi-hop chain, ambiguity, contradiction, or irrelevant context.

Treat `required_facts` as optional for public datasets that do not provide them. Do not silently manufacture ground-truth labels with the same model you are evaluating.

## 5. Starter code

Here is a minimal working evaluation core. It lets you plug in a decision strategy, record per-case scores and latency, and compute the first metrics. It is a scaffold, not a claim that all model adapters have already been implemented.

Install the initial dependencies:

```
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn pyyaml
```

Create `src/sufficiency_lab/benchmark.py`:

```
from dataclasses import dataclassfrom time import perf_counterfrom typing import Protocolimport numpy as npimport pandas as pdfrom sklearn.metrics import (    roc_auc_score,    average_precision_score,    brier_score_loss,    confusion_matrix,)@dataclassclass Case:    id: str    question: str    context: str    sufficient: bool    split: str = "test"@dataclassclass Decision:    # Probability that the context is sufficient.    p_sufficient: float    strategy: str    latency_ms: floatclass Strategy(Protocol):    name: str    def predict(self, case: Case) -> Decision:
```

One important implementation detail: the baseline above measures lexical overlap, not true sufficiency. It exists to establish a cheap reference point. All strategies must return a meaningful probability, not just a binary label, for the calibration and threshold analysis to be useful.

### Add a System 1 adapter

Keep the adapter thin: convert the case into a typed decision request, invoke the model, and map its output to `p_sufficient`. Do not let each strategy have its own evaluation prompt or threshold-selection logic.

```
class System1Strategy:    name = "system1"    def __init__(self, client, model_id: str):        self.client = client        self.model_id = model_id    def predict(self, case: Case) -> Decision:        start = perf_counter()        # Replace this with the actual decision API for your        # chosen System 1 model. Do not assume every API has        # the same schema or returns calibrated probabilities.        result = self.client.decide(            model=self.model_id,            question=case.question,            context=case.context,            decision="Is the context sufficient to answer?",            options=["insufficient", "sufficient"],        )        probability = float(result.probabilities["sufficient"])        return Decision(            p_sufficient=probability,            strategy=self.name,            latency_ms=(perf_counter() - start) * 1000,        )
```

The `client.decide` method is intentionally an adapter contract, not a claim about Laya's exact SDK. Implement it against the actual model's supported interface after verifying its model card and serving API.

## 6. A fair System 1 versus LLM experiment

For the first experiment, I would compare these methods on the same fixed test cases:

| Method                       | Autoregressive generation?          | Main question                                                       |
| ---------------------------- | ----------------------------------- | ------------------------------------------------------------------- |
| Keyword/entity baseline      | No                                  | How far can cheap checks get?                                       |
| NLI classifier               | No                                  | Does entailment detect missing evidence?                            |
| ModernBERT sufficiency probe | No                                  | Can a dedicated classifier learn the task?                          |
| Laya / System 1              | No, if using its decision interface | Can fast structured decisions match or beat other gates?            |
| Local LLM judge              | Yes                                 | Does explicit reasoning improve reliability?                        |
| Hidden-state probe           | No extra generation pass            | Does the target LLM contain a useful sufficiency signal internally? |

For the LLM judge, use a fixed rubric and a constrained output schema. Record its full inference latency, including prompt processing and any required output tokens. For the hidden-state probe, measure extraction and probe costs separately, and report whether the hidden-state extraction is actually available in your serving stack.

Use paired comparisons: every method should see the same case IDs. Report confidence intervals using bootstrap resampling grouped by question, so paired sufficient/insufficient variants stay together.

## 7. Make the benchmark production-relevant

After the basic benchmark works, add a second experiment that runs the actual gate in a RAG pipeline:

1. Retrieve a fixed set of documents.
2. Run the sufficiency strategy.
3. If sufficient, generate an answer grounded in the context.
4. If insufficient, run one bounded retrieval-repair cycle.
5. If still insufficient, abstain.
6. Record the entire trace, including evidence, decisions, and cost.

For your layered RAG architecture, I would make the repair path explicit: hybrid dense + BM25 retrieval, reranking, then targeted escalation to knowledge graphs or other sources. Keep permission-aware retrieval, collection scope, and provenance intact. The sufficiency gate should never be able to retrieve evidence that the user is not authorized to access.

The terminal states should be explicit:

- `ANSWERED`
- `ABSTAINED_INSUFFICIENT_CONTEXT`
- `ABSTAINED_CONTRADICTION`
- `RETRIEVAL_EXHAUSTED`
- `ERROR`

This avoids conflating a correct abstention with a failed request or an incorrect refusal.

## 8. What I would do first

My recommended order is:

1. Clone and reproduce [SufficiencyBench](https://github.com/inab-lab/SufficiencyBench).
2. Use its existing evaluation code to establish reference results.
3. Add the universal case/result schema and benchmark runner above.
4. Implement the ModernBERT, NLI, System 1, and local LLM judge adapters.
5. Add validation-only threshold tuning and risk-coverage plots.
6. Run end-to-end gating experiments only after the standalone sufficiency evaluation is stable.

This gives you a defensible answer to both questions: which strategy identifies sufficient evidence most accurately? and which strategy makes the best production gate?

One detail I would confirm before implementing the first model adapter: when you say “CLEF,” do you mean Laya, the self-hostable non-autoregressive System 1 model, or a different model? I can keep the benchmark interface model-agnostic either way.
