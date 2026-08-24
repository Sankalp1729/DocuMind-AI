# RAG Evaluation

DocuMind AI treats retrieval quality as an engineering metric rather than a subjective demo result.

## Reproducible benchmark

Run the same retrieval stack used by the API against a versioned dataset:

```powershell
python scripts/run_rag_benchmark.py --dataset example_rag_benchmark --top-k 5
```

Write a JSON result to a custom location when needed:

```powershell
python scripts/run_rag_benchmark.py --dataset example_rag_benchmark --top-k 5 --output data/evaluations/manual_benchmark.json
```

The benchmark records the exact `top_k`, query-level retrieval results, ranking metrics, groundedness, hallucination risk, and latency.

## Metrics

The benchmark layer reports:

- **Precision@K** — fraction of the retrieved top-K results that are relevant.
- **Recall@K** — fraction of known relevant evidence retrieved in the top-K.
- **Average Precision (AP)** — rewards relevant evidence appearing earlier in the ranked list and penalizes missed relevant targets.
- **Mean Reciprocal Rank (MRR)** — measures how early the first relevant result appears.
- **nDCG@K** — measures ranking quality against an ideal ordering.
- **Groundedness rate** — proportion of benchmark answers classified as grounded by the existing groundedness scorer.
- **Hallucination rate** — proportion of benchmark answers classified as high-risk.
- **Retrieval/reranking/generation latency** — tracks the cost of the RAG pipeline.

## Metric contract

`top_k` is validated and applied consistently to ranking metrics. Recall is bounded to `[0, 1]`, and AP uses the total number of known relevant targets as its denominator so that missed evidence lowers the score.

The legacy `precision_at_10` and `recall_at_10` keys remain available for the admin/leaderboard contract, even when a benchmark is configured with another K.

## Experiment protocol

When comparing retrieval strategies, keep these fixed unless the variable is intentionally under test:

1. benchmark dataset and relevance judgments;
2. document corpus and ingestion/chunking configuration;
3. embedding model;
4. answer-generation model and prompt;
5. evaluation cutoff `top_k`;
6. environment/hardware when comparing latency.

Change one retrieval component at a time, persist every result, and compare quality against latency. Do not publish the example fixture's output as production performance.

## Benchmark dataset

A dataset is a JSON document matching `BenchmarkDataset` in `backend/schemas/evaluation.py`:

```json
{
  "dataset_name": "example_rag_benchmark",
  "description": "Small reproducible benchmark example.",
  "version": "1.0",
  "queries": [
    {
      "query_id": "q1",
      "question": "What is the retrieval strategy?",
      "relevant_sources": ["retrieval.md"],
      "relevant_pages": [1],
      "answer_references": ["hybrid retrieval"]
    }
  ]
}
```

Store benchmark datasets under the application's `evaluation_datasets` directory. The `EvaluationService` discovers JSON datasets automatically.

## Regression workflow

1. Run a benchmark against a fixed dataset.
2. Persist the benchmark result.
3. Run the same dataset after a retrieval change.
4. Compare Precision@K, Recall@K, MAP, MRR, nDCG, groundedness, hallucination rate and latency.
5. Investigate any quality regression before merging.

### Placement interview story

A strong engineering discussion is not "I used RAG." It is:

> I built a measurable retrieval pipeline, defined relevance judgments, tracked ranking metrics and latency, and added regression tests so retrieval changes could be evaluated rather than judged by a few manual prompts.

Do not publish benchmark percentages until they have been measured against a fixed dataset and configuration.
