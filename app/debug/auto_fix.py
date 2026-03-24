from __future__ import annotations

import re
from typing import Dict, List

MIN_QUERY_FIX_TOP_K = 3
QUERY_FALLBACK_SUFFIX = "information"


class AutoFixEngine:
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _clean_query(self, query: str) -> str:
        seen: set[str] = set()
        cleaned_tokens: List[str] = []

        for token in self._tokenize(query):
            if token not in seen:
                seen.add(token)
                cleaned_tokens.append(token)

        if not cleaned_tokens:
            return query


        if len(cleaned_tokens) == 1:
            cleaned_tokens.append(QUERY_FALLBACK_SUFFIX)

        return " ".join(cleaned_tokens)

    def apply_fix(self, query: str, failure_type: str, original_top_k: int = 1) -> Dict:
        if failure_type == "query_issue":
            improved_query = self._clean_query(query)
            return {
                "query": improved_query,
                "top_k": max(MIN_QUERY_FIX_TOP_K, original_top_k * 3),
                "fix_applied": [
                    "Removed repeated words",
                    f"Expanded query to: {improved_query}",
                ],
            }

        if failure_type == "missing_context":
            return {
                "query": query,
                "top_k": 5,
                "fix_applied": [
                    "Increased retrieval depth",
                    "Fetched more chunks to improve context coverage",
                ],
            }

        if failure_type == "retrieval_failure":
            return {
                "query": query,
                "top_k": 8,
                "fix_applied": [
                    "Increased retrieval depth",
                    "Expanded top_k to improve recall",
                ],
            }

        if failure_type == "grounding_failure":
            return {
                "query": query,
                "top_k": 3,
                "fix_applied": [
                    "Reduced context size",
                    "Focused the model on fewer retrieved chunks",
                ],
            }

        return {
            "query": query,
            "top_k": 5,
            "fix_applied": ["No fix applied"],
        }
