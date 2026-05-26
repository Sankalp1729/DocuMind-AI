from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from backend.core.config import DATA_DIR
from backend.evaluation.benchmark_runner import load_benchmark_dataset, run_benchmark
from backend.schemas.evaluation import (
    BenchmarkDataset,
    BenchmarkResult,
    EvaluationHistoryEntry,
    LeaderboardEntry,
)


logger = logging.getLogger(__name__)


class EvaluationService:
    """Persists and retrieves evaluation benchmark results."""
    
    def __init__(self, storage_dir: Path = DATA_DIR):
        self.storage_dir = storage_dir / "evaluations"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = storage_dir / "evaluation_datasets"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.storage_dir / "benchmark_history.jsonl"

    def _coerce_payload(self, result: Dict | BenchmarkResult) -> Dict:
        if isinstance(result, BenchmarkResult):
            return result.model_dump(mode="json")
        return dict(result)

    def _write_history_entry(self, payload: Dict) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")

    def _load_history_entries(self) -> List[Dict]:
        if not self.history_path.exists():
            return []

        entries: List[Dict] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed evaluation history line")
        return entries

    @staticmethod
    def _metrics_payload(payload: Dict) -> Dict[str, float]:
        return payload.get("retrieval_metrics") or payload.get("metrics") or {}

    def list_datasets(self) -> List[BenchmarkDataset]:
        datasets: List[BenchmarkDataset] = []
        for path in sorted(self.dataset_dir.glob("*.json")):
            datasets.append(load_benchmark_dataset(path))
        return datasets

    def load_dataset(self, dataset_name: str) -> Optional[BenchmarkDataset]:
        path = self.dataset_dir / f"{dataset_name}.json"
        if not path.exists():
            return None
        return load_benchmark_dataset(path)

    def run_dataset_benchmark(self, rag_service, dataset_name: str, benchmark_id: Optional[str] = None, top_k: int = 10, answer_fn=None) -> BenchmarkResult:
        dataset = self.load_dataset(dataset_name)
        if dataset is None:
            raise FileNotFoundError(f"Benchmark dataset not found: {dataset_name}")

        previous = self.list_benchmark_results()
        previous_run_id = None
        if previous:
            previous_run = previous[-1]
            previous_run_id = previous_run.get("benchmark_id") or previous_run.get("run_id")
        baseline = previous[-1].get("retrieval_metrics", {}) if previous else None
        result = run_benchmark(
            rag_service=rag_service,
            dataset=dataset,
            top_k=top_k,
            benchmark_id=benchmark_id,
            baseline=baseline,
            regression_against=previous_run_id,
            answer_fn=answer_fn,
        )
        return result
    
    def save_benchmark_result(self, run_id: str, result: Dict) -> str:
        """Save a benchmark result to disk."""
        filepath = self.storage_dir / f"benchmark_{run_id}.json"
        payload = self._coerce_payload(result)
        result_with_ts = {**payload, "benchmark_id": payload.get("benchmark_id", run_id), "timestamp": payload.get("timestamp", datetime.now().isoformat())}
        filepath.write_text(json.dumps(result_with_ts, indent=2, default=str), encoding="utf-8")
        self._write_history_entry(result_with_ts)
        logger.info(f"Saved benchmark result to {filepath}")
        return str(filepath)
    
    def load_benchmark_result(self, run_id: str) -> Optional[Dict]:
        """Load a benchmark result from disk."""
        filepath = self.storage_dir / f"benchmark_{run_id}.json"
        if not filepath.exists():
            return None
        return json.loads(filepath.read_text(encoding="utf-8"))

    def load_benchmark_history(self) -> List[EvaluationHistoryEntry]:
        history = []
        for payload in self._load_history_entries():
            metrics = self._metrics_payload(payload)
            benchmark_id = payload.get("benchmark_id") or payload.get("run_id") or "unknown"
            history.append(
                EvaluationHistoryEntry(
                    benchmark_id=benchmark_id,
                    dataset_name=payload.get("dataset_name", "unknown"),
                    timestamp=payload.get("timestamp", datetime.now().isoformat()),
                    composite_score=self._composite_score(metrics, payload.get("groundedness_rate", 0.0), payload.get("hallucination_rate", 0.0), payload.get("total_latency_ms", 0.0)),
                    metrics=metrics,
                    regression_delta=payload.get("regression_delta", {}),
                    notes=payload.get("notes"),
                )
            )
        return history

    def _composite_score(self, metrics: Dict[str, float], groundedness_rate: float, hallucination_rate: float, total_latency_ms: float) -> float:
        precision = float(metrics.get("precision_at_10", metrics.get("precision_at_k", 0.0)))
        recall = float(metrics.get("recall_at_10", metrics.get("recall_at_k", 0.0)))
        mrr = float(metrics.get("mrr", 0.0))
        ndcg = float(metrics.get("ndcg", 0.0))
        latency_penalty = min(1.0, total_latency_ms / 10000.0)
        hallucination_penalty = max(0.0, hallucination_rate)
        return round(
            max(
                0.0,
                (precision * 0.25) + (recall * 0.2) + (mrr * 0.25) + (ndcg * 0.15) + (groundedness_rate * 0.15) - (latency_penalty * 0.1) - (hallucination_penalty * 0.15),
            ),
            4,
        )

    def leaderboard(self, limit: int = 20) -> List[LeaderboardEntry]:
        history = self.load_benchmark_history()
        benchmark_lookup = {run.get("benchmark_id") or run.get("run_id"): run for run in self.list_benchmark_results()}
        leaderboard = [
            LeaderboardEntry(
                benchmark_id=entry.benchmark_id,
                dataset_name=entry.dataset_name,
                timestamp=entry.timestamp,
                composite_score=entry.composite_score,
                precision_at_10=float(entry.metrics.get("precision_at_10", entry.metrics.get("precision_at_k", 0.0))),
                recall_at_10=float(entry.metrics.get("recall_at_10", entry.metrics.get("recall_at_k", 0.0))),
                mrr=float(entry.metrics.get("mrr", 0.0)),
                ndcg=float(entry.metrics.get("ndcg", 0.0)),
                groundedness_rate=float((benchmark_lookup.get(entry.benchmark_id) or {}).get("groundedness_rate", 0.0)),
                hallucination_rate=float((benchmark_lookup.get(entry.benchmark_id) or {}).get("hallucination_rate", 0.0)),
                total_latency_ms=float((benchmark_lookup.get(entry.benchmark_id) or {}).get("total_latency_ms", 0.0)),
            )
            for entry in history
        ]
        leaderboard.sort(key=lambda item: item.composite_score, reverse=True)
        return leaderboard[:limit]
    
    def list_benchmark_results(self) -> List[Dict]:
        """List all stored benchmark results."""
        results = []
        for f in sorted(self.storage_dir.glob("benchmark_*.json")):
            results.append(json.loads(f.read_text(encoding="utf-8")))
        return results

    def benchmark_dashboard(self) -> Dict:
        """Aggregate stored benchmark runs into dashboard-friendly metrics."""
        benchmarks = self.list_benchmark_results()
        if not benchmarks:
            return {
                "runs": 0,
                "latest_run": None,
                "average_metrics": {},
                "trend": [],
                "leaderboard": [],
                "history": [],
            }

        metric_keys = ["precision_at_10", "recall_at_10", "map", "mrr", "ndcg"]
        average_metrics = {
            key: round(mean(float(self._metrics_payload(run).get(key, 0.0)) for run in benchmarks), 4)
            for key in metric_keys
        }

        latest_run = benchmarks[-1]
        trend = [
            {
                "run_id": run.get("run_id") or run.get("benchmark_id"),
                "timestamp": run.get("timestamp"),
                "precision_at_10": self._metrics_payload(run).get("precision_at_10"),
                "recall_at_10": self._metrics_payload(run).get("recall_at_10"),
                "groundedness_rate": run.get("groundedness_rate"),
                "hallucination_rate": run.get("hallucination_rate"),
            }
            for run in benchmarks[-20:]
        ]

        return {
            "runs": len(benchmarks),
            "latest_run": latest_run,
            "average_metrics": average_metrics,
            "trend": trend,
            "leaderboard": [entry.model_dump(mode="json") for entry in self.leaderboard()],
            "history": [entry.model_dump(mode="json") for entry in self.load_benchmark_history()],
        }
    
    def save_evaluation_run(self, run_id: str, metrics: Dict, dataset_name: str) -> str:
        """Save an evaluation run."""
        filepath = self.storage_dir / f"eval_{run_id}.json"
        result = {
            "run_id": run_id,
            "dataset": dataset_name,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        filepath.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"Saved evaluation run to {filepath}")
        return str(filepath)

    def save_benchmark_dashboard_result(self, benchmark_result: BenchmarkResult) -> str:
        """Persist a structured benchmark result model."""
        payload = benchmark_result.model_dump(mode="json")
        return self.save_benchmark_result(benchmark_result.benchmark_id, payload)
