from __future__ import annotations

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer


def pseudo_relevance_feedback(docs: List[str], top_m: int = 3, top_terms: int = 5) -> List[str]:
    """Extract top terms from top-M documents using TF-IDF for query expansion.

    Returns a list of expansion terms.
    """
    if not docs:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    # aggregate TF-IDF across top_m docs
    top_m = min(top_m, X.shape[0])
    agg = X[:top_m].sum(axis=0)
    agg = agg.A1
    ranked_idx = agg.argsort()[::-1]
    terms = []
    for idx in ranked_idx[: top_terms * 3]:
        term = feature_names[idx]
        if term not in terms:
            terms.append(term)
        if len(terms) >= top_terms:
            break

    return terms
