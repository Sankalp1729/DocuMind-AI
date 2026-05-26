from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.core.config import DATA_DIR


@dataclass(slots=True)
class MemoryEntry:
    query: str
    answer: str
    tags: list[str]
    metadata: dict[str, Any]


class MemoryEnhancedRetrieval:
    """Very small persistent memory layer for agentic retrieval.

    Stores prior successful question/answer pairs and exposes a lexical lookup.
    This keeps the dependency surface minimal while allowing upgrade to embeddings later.
    """

    def __init__(self, storage_dir: Path | None = None):
        self.storage_dir = Path(storage_dir or DATA_DIR / "agentic_memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.storage_dir / "memory.jsonl"

    @staticmethod
    def _keywords(text: str) -> Counter[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return Counter(token for token in tokens if len(token) > 2)

    def remember(self, query: str, answer: str, metadata: dict[str, Any] | None = None) -> None:
        entry = {
            "query": query,
            "answer": answer,
            "tags": list(self._keywords(query).keys())[:20],
            "metadata": metadata or {},
        }
        with self.memory_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recall(self, query: str, limit: int = 3) -> list[MemoryEntry]:
        if not self.memory_file.exists():
            return []

        query_terms = self._keywords(query)
        scored: list[tuple[int, MemoryEntry]] = []
        with self.memory_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                candidate = MemoryEntry(
                    query=payload.get("query", ""),
                    answer=payload.get("answer", ""),
                    tags=payload.get("tags", []),
                    metadata=payload.get("metadata", {}),
                )
                candidate_terms = self._keywords(candidate.query + " " + candidate.answer)
                score = sum(min(query_terms[token], candidate_terms[token]) for token in query_terms.keys() & candidate_terms.keys())
                if score > 0:
                    scored.append((score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]
