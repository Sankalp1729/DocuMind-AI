from __future__ import annotations

from typing import Iterable, List
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Simple BM25 retriever built on top of rank_bm25.

    Expects pre-tokenized documents (list of tokens per document) and returns top-k doc indices and scores.
    """

    def __init__(self, tokenized_corpus: List[List[str]]):
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_size = len(tokenized_corpus)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        # lightweight tokenizer: split on whitespace; production should use a robust tokenizer
        return [t.lower() for t in text.split() if t.strip()]

    @classmethod
    def from_texts(cls, texts: Iterable[str]):
        tokenized = [cls.tokenize_text(t) for t in texts]
        return cls(tokenized)

    def retrieve(self, query: str, top_k: int = 10):
        tokens = self.tokenize_text(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in ranked]
