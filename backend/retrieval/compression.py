from __future__ import annotations

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def compress_context(passages: List[str], max_tokens: int = 512) -> str:
    """Compress multiple passages into a shorter context by selecting top-ranked sentences.

    Simple approach: split passages into sentences, TF-IDF rank them, return top sentences until token budget.
    """
    sentences = []
    for p in passages:
        for s in p.split("."):
            s = s.strip()
            if s:
                sentences.append(s)

    if not sentences:
        return ""

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()
    ranked_idx = scores.argsort()[::-1]

    selected = []
    current_tokens = 0
    for idx in ranked_idx:
        s = sentences[idx]
        token_est = len(s.split())
        if current_tokens + token_est > max_tokens:
            continue
        selected.append(s)
        current_tokens += token_est
        if current_tokens >= max_tokens:
            break

    return ". ".join(selected)
