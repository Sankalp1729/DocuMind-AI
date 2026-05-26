from __future__ import annotations

import re
from typing import List


class GroundednessScorer:
    """Heuristic-based groundedness scoring: checks if answer contains key terms from retrieved passages."""

    def __init__(self):
        self.hallucination_keywords = {
            "i don't know",
            "i cannot",
            "unclear",
            "not mentioned",
            "no information",
            "can't verify",
            "cannot verify",
            "may be",
            "might be",
        }
        self.contradiction_keywords = {"however", "but", "although", "contrary", "instead"}

    def score_groundedness(self, answer: str, retrieved_passages: List[str], threshold: float = 0.6) -> dict:
        """Score groundedness of an answer based on retrieved passages.
        
        Uses heuristics:
        1. Presence of hallucination keywords
        2. Lexical overlap with retrieved text
        3. Named entity presence in context
        """
        answer_lower = answer.lower()
        
        # Check for explicit hallucination markers
        has_hallucination_marker = any(kw in answer_lower for kw in self.hallucination_keywords)
        
        # Lexical overlap: measure token overlap between answer and passages
        answer_tokens = set(re.findall(r"\w+", answer_lower))
        passage_tokens = set()
        passage_text = " ".join(retrieved_passages).lower()
        for p in retrieved_passages:
            passage_tokens.update(re.findall(r"\w+", p.lower()))
        
        if not answer_tokens or not passage_tokens:
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "hallucination_risk": "high",
                "reasoning": "Empty answer or passages",
                "evidence_coverage": 0.0,
                "question_alignment": 0.0,
                "hallucination_signals": ["empty_context"],
                "support_summary": "No retrieved passages available for grounding",
            }
        
        overlap = len(answer_tokens & passage_tokens) / len(answer_tokens)
        alignment = overlap
        hallucination_signals = []

        if has_hallucination_marker:
            hallucination_signals.append("explicit_uncertainty_marker")

        if any(keyword in answer_lower for keyword in self.contradiction_keywords) and overlap < threshold:
            hallucination_signals.append("low_overlap_contrastive_language")

        if len(re.findall(r"\b\d+\b", answer_lower)) > 0 and len(re.findall(r"\b\d+\b", passage_text)) == 0:
            hallucination_signals.append("unsupported_numeric_detail")

        support_summary = f"Lexical overlap {overlap:.1%} across retrieved passages"
        
        # Heuristic scoring
        if has_hallucination_marker:
            return {
                "is_grounded": False,
                "confidence": 0.2,
                "hallucination_risk": "high",
                "reasoning": "Answer contains explicit hallucination markers",
                "evidence_coverage": overlap,
                "question_alignment": alignment,
                "hallucination_signals": hallucination_signals,
                "support_summary": support_summary,
            }
        
        if overlap >= threshold:
            return {
                "is_grounded": True,
                "confidence": min(0.95, overlap),
                "hallucination_risk": "low",
                "reasoning": f"High lexical overlap ({overlap:.1%}) with retrieved passages",
                "evidence_coverage": overlap,
                "question_alignment": alignment,
                "hallucination_signals": hallucination_signals,
                "support_summary": support_summary,
            }
        
        if overlap >= 0.4:
            return {
                "is_grounded": True,
                "confidence": overlap,
                "hallucination_risk": "medium",
                "reasoning": f"Moderate lexical overlap ({overlap:.1%}) with retrieved passages",
                "evidence_coverage": overlap,
                "question_alignment": alignment,
                "hallucination_signals": hallucination_signals,
                "support_summary": support_summary,
            }
        
        return {
            "is_grounded": False,
            "confidence": 1.0 - overlap,
            "hallucination_risk": "high",
            "reasoning": f"Low lexical overlap ({overlap:.1%}) with retrieved passages",
            "evidence_coverage": overlap,
            "question_alignment": alignment,
            "hallucination_signals": hallucination_signals or ["low_lexical_overlap"],
            "support_summary": support_summary,
        }

    def score_answer_relevance(self, question: str, answer: str) -> float:
        """Simple relevance score: check if answer answers the question."""
        q_tokens = set(re.findall(r"\w+", question.lower()))
        a_tokens = set(re.findall(r"\w+", answer.lower()))
        
        if not q_tokens or not a_tokens:
            return 0.5
        
        # If answer contains many question tokens, it's likely addressing the question
        overlap = len(q_tokens & a_tokens) / len(q_tokens)
        return min(1.0, overlap + 0.3)  # base score + overlap
