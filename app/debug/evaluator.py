from __future__ import annotations

from collections import Counter
import re
from typing import Any, Dict, List

import numpy as np

from app.rag.embeddings import EmbeddingModel

SEMANTIC_MATCH_THRESHOLD = 0.65
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    a = np.asarray(vector_a, dtype="float32")
    b = np.asarray(vector_b, dtype="float32")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    similarity = float(np.dot(a, b) / denominator)
    return max(0.0, min(1.0, similarity))


def _extract_meaningful_terms(text: str) -> set[str]:
    terms = set(re.findall(r"\b\w+\b", text.lower()))
    return {term for term in terms if term not in STOPWORDS}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


class DebugEvaluator:
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model

    def _classify_failure(
        self,
        query_quality_issue: bool,
        retrieval_score: float,
        grounding_score: float,
        missing_context: bool,
        answer_too_short: bool,
    ) -> str:
        if query_quality_issue:
            return "query_issue"
        if missing_context:
            return "missing_context"
        if grounding_score < 0.65 or answer_too_short:
            return "grounding_failure"
        if retrieval_score < 0.5:
            return "retrieval_failure"
        return "success"

    def _is_low_quality_answer(self, answer: str) -> bool:
        answer_length = len(_tokenize(answer))
        return answer_length <= 2

    def _has_query_quality_issue(self, query: str) -> bool:
        tokens = _tokenize(query)
        total_tokens = len(tokens)
        if total_tokens == 0:
            return False

        counter = Counter(tokens)
        max_freq = max(counter.values())
        repeated_ratio = max_freq / total_tokens

        if total_tokens == 1:
            query_quality_issue = False
        else:
            query_quality_issue = repeated_ratio > 0.6

        print("TOKENS:", tokens)
        print("REPEATED_RATIO:", repeated_ratio)
        print("QUERY_QUALITY:", query_quality_issue)
        # single-word queries are valid; only repetition across multiple tokens is penalized
        return query_quality_issue

    def _compute_retrieval_score(self, chunks: List[Dict[str, Any]]) -> float:
        vector_distances = [
            float(chunk["vector_distance"])
            for chunk in chunks
            if chunk.get("vector_distance") is not None
        ]
        if not vector_distances:
            return 0.0

        avg_distance = float(np.mean(vector_distances))
        print("avg_distance:", avg_distance)
        raw_score = 1 / (1 + avg_distance)
        retrieval_score = min(1.0, raw_score * 1.5)
        return _clamp_score(retrieval_score)

    def _compute_grounding_score(self, answer: str, chunk_texts: List[str]) -> float:
        if not answer.strip() or not chunk_texts:
            return 0.0

        embeddings = self.embedding_model.encode([answer, *chunk_texts])
        answer_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]

        similarities = [_cosine_similarity(answer_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings]
        return max(similarities) if similarities else 0.0

    def _has_missing_context(self, query: str, chunk_texts: List[str]) -> bool:
        if not chunk_texts:
            return True

        query_terms = _extract_meaningful_terms(query)
        combined_context = " ".join(chunk_texts).lower()
        keyword_match = any(term in combined_context for term in query_terms) if query_terms else False

        embeddings = self.embedding_model.encode([query, *chunk_texts])
        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]
        max_similarity = max(
            (_cosine_similarity(query_embedding, chunk_embedding) for chunk_embedding in chunk_embeddings),
            default=0.0,
        )
        # stricter threshold to detect out-of-domain queries
        semantic_match = max_similarity > SEMANTIC_MATCH_THRESHOLD

        return not (keyword_match or semantic_match)

    def evaluate(self, query: str, chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        query_quality_issue = self._has_query_quality_issue(query)
        answer_too_short = self._is_low_quality_answer(answer)

        if not chunks:
            failure_type = self._classify_failure(query_quality_issue, 0.0, 0.0, True, answer_too_short)
            return {
                "retrieval_score": 0.0,
                "grounding_score": 0.0,
                "missing_context": True,
                "query_quality_issue": query_quality_issue,
                "answer_too_short": answer_too_short,
                "confidence": 0.0,
                "failure_type": failure_type,
            }

        chunk_texts = [chunk["chunk"]["text"] for chunk in chunks if chunk["chunk"]["text"].strip()]
        retrieval_score = self._compute_retrieval_score(chunks)
        grounding_score = self._compute_grounding_score(answer, chunk_texts)
        missing_context = self._has_missing_context(query, chunk_texts)
        if grounding_score == 0:
            missing_context = True

        confidence = (0.5 * retrieval_score) + (0.5 * grounding_score)
        if missing_context:
            confidence *= 0.5

        failure_type = self._classify_failure(
            query_quality_issue,
            retrieval_score,
            grounding_score,
            missing_context,
            answer_too_short,
        )

        return {
            "retrieval_score": round(_clamp_score(retrieval_score), 2),
            "grounding_score": round(_clamp_score(grounding_score), 2),
            "missing_context": missing_context,
            "query_quality_issue": query_quality_issue,
            "answer_too_short": answer_too_short,
            "confidence": round(_clamp_score(confidence), 2),
            "failure_type": failure_type,
        }
