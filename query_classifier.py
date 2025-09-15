"""Lightweight + LLM-based query classification and optional rephrase."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Literal, Optional
import json

logger = logging.getLogger(__name__)

QueryType = Literal["short_fact", "how_to", "troubleshooting", "ambiguous", "other"]

@dataclass
class QueryClassification:
    query_type: QueryType
    needs_rephrase: bool = False
    suggested_k: int = 5


class QueryClassifier:
    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, question: str) -> QueryClassification:
        """Classify a question using LLM when available, else heuristics."""
        if self.llm:
            try:
                prompt = (
                    "Classify the user's question. Return JSON with keys: query_type(one of: short_fact, how_to, "
                    "troubleshooting, ambiguous, other), suggested_k(integer in [3,4,5,6,8]), and needs_rephrase(true/false).\n"
                    "Question: " + question
                )
                resp = self.llm.invoke(prompt)
                text = resp if isinstance(resp, str) else str(resp)
                data = json.loads(text.strip())
                qt = data.get("query_type", "other")
                nr = bool(data.get("needs_rephrase", False))
                k = int(data.get("suggested_k", 5))
                k = max(1, min(10, k))
                if qt not in ("short_fact", "how_to", "troubleshooting", "ambiguous", "other"):
                    qt = "other"
                return QueryClassification(qt, nr, k)
            except Exception as e:
                logger.warning(f"LLM classify failed, using heuristics: {e}")
        q = question.strip().lower()
        length = len(q.split())
        # Heuristic rules
        if length < 4:
            return QueryClassification("short_fact", suggested_k=3)
        if any(x in q for x in ["how do", "how to", "steps", "example"]):
            return QueryClassification("how_to", suggested_k=6)
        if any(x in q for x in ["error", "fail", "issue", "exception"]):
            return QueryClassification("troubleshooting", suggested_k=8)
        if q.endswith("?") and length <= 3:
            return QueryClassification("ambiguous", needs_rephrase=True, suggested_k=4)
        return QueryClassification("other")

    def rephrase(self, question: str, classification: QueryClassification) -> Optional[str]:
        """Return an LLM-rephrased, self-contained question when flagged as ambiguous.
        Returns None if not rephrased or on failure.
        """
        if not self.llm or not classification.needs_rephrase:
            return None
        try:
            prompt = (
                "Rewrite the user's question to be explicit and self-contained for retrieval. "
                "Preserve the original intent and important details. Return only the rewritten question.\n"
                f"Question: {question}"
            )
            new_q = self.llm.invoke(prompt)
            new_q = new_q if isinstance(new_q, str) else str(new_q)
            new_q = new_q.strip()
            logger.info("Rephrased ambiguous query with LLM")
            return new_q or None
        except Exception as e:
            logger.warning(f"LLM rephrase failed: {e}")
            return None
