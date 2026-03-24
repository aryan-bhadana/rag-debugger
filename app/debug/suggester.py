from __future__ import annotations

from typing import Dict, List


class Suggester:
    def suggest(self, debug: Dict) -> List[str]:
        failure_type = str(debug.get("failure_type", "success"))
        retrieval_score = float(debug.get("retrieval_score", 0.0))
        grounding_score = float(debug.get("grounding_score", 0.0))
        confidence = float(debug.get("confidence", 0.0))
        answer_too_short = bool(debug.get("answer_too_short", False))

        if failure_type == "query_issue":
            return [
                "Rephrase query with more specific and meaningful terms",
                "Avoid repeating the same word multiple times",
            ]

        if failure_type == "missing_context":
            return [
                (
                    f"Missing context detected (confidence: {confidence:.2f}). "
                    "Add more domain-specific documents covering this query type."
                ),
                (
                    f"Current response confidence is {confidence:.2f}. "
                    "Expand document coverage so this kind of query is represented in the knowledge base."
                ),
            ]

        if failure_type == "retrieval_failure":
            return [
                (
                    f"Low retrieval score ({retrieval_score:.2f}). "
                    "Consider increasing top_k from 3 -> 8 to improve recall."
                ),
                (
                    f"Retrieved matches are weak (score: {retrieval_score:.2f}). "
                    "Apply reranking to improve relevance before answer generation."
                ),
            ]

        if failure_type == "grounding_failure":
            suggestions = [
                (
                    f"Low grounding score ({grounding_score:.2f}). "
                    "Improve the prompt to enforce stricter grounding to retrieved evidence."
                ),
                (
                    f"Grounding remains weak (score: {grounding_score:.2f}). "
                    "Reduce irrelevant context so the model focuses on the most useful chunks."
                ),
            ]
            if answer_too_short:
                suggestions = [
                    "Improve prompt to generate more detailed answers",
                    "Ensure context is fully utilized",
                ]
            return suggestions

        return ["No action needed"]
