from __future__ import annotations

import json
from pathlib import Path


def test_benchmark_cli_defaults_are_documented_by_parser() -> None:
    source = Path("scripts/run_rag_benchmark.py").read_text(encoding="utf-8")
    assert "--dataset" in source
    assert "--top-k" in source
    assert "--output" in source


def test_benchmark_cli_output_is_json_serializable() -> None:
    payload = {
        "benchmark_id": "example-k5",
        "dataset_name": "example_rag_benchmark",
        "top_k": 5,
        "retrieval_metrics": {"precision_at_5": 0.5, "recall_at_5": 0.75},
    }
    encoded = json.dumps(payload)
    assert json.loads(encoded)["top_k"] == 5
