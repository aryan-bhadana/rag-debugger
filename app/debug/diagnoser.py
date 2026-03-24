from __future__ import annotations

from typing import Dict, List


class Diagnoser:
    def diagnose(self, debug: Dict) -> List[str]:
        messages: List[str] = []

        retrieval_score = float(debug.get("retrieval_score", 0.0))
        grounding_score = float(debug.get("grounding_score", 0.0))
        missing_context = bool(debug.get("missing_context", False))
        answer_too_short = bool(debug.get("answer_too_short", False))
        confidence = float(debug.get("confidence", 0.0))
        failure_type = str(debug.get("failure_type", "success"))

        if failure_type == "query_issue":
            messages.append("Query appears poorly formed or repetitive, which affects retrieval quality")
            if confidence < 0.75:
                messages.append(f"Low confidence due to poor query quality (confidence: {confidence:.2f})")
            return messages

        if failure_type == "retrieval_failure" and retrieval_score < 0.5:
            messages.append(
                f"Low retrieval quality (score: {retrieval_score:.2f}) indicates a weak match between the query and retrieved documents"
            )

        if failure_type == "grounding_failure" and grounding_score < 0.65:
            messages.append(
                f"Answer is not well grounded in retrieved context (score: {grounding_score:.2f}), which raises hallucination risk"
            )
            if answer_too_short:
                messages.append("Answer is too short or lacks meaningful information")

        if failure_type == "missing_context" and missing_context:
            messages.append("Relevant information is missing from the knowledge base for this query")

        if confidence > 0.75:
            messages.append(
                f"High confidence response with strong retrieval and grounding (confidence: {confidence:.2f})"
            )
        elif 0.4 <= confidence <= 0.75:
            messages.append(
                f"Moderate confidence (score: {confidence:.2f}): response may be partially incomplete or imprecise"
            )
        else:
            messages.append(
                f"Low confidence (score: {confidence:.2f}): system is uncertain due to poor retrieval or grounding"
            )

        return messages
