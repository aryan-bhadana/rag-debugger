from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import VectorStore


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _normalize_scores(scores: Dict[str, float], invert: bool = False) -> Dict[str, float]:
    if not scores:
        return {}

    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return {key: 1.0 for key in scores}

    normalized: Dict[str, float] = {}
    for key, value in scores.items():
        score = (value - minimum) / (maximum - minimum)
        normalized[key] = 1.0 - score if invert else score

    return normalized


class BM25Retriever:
    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        tokenized_corpus = [_tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        for index in ranked_indices:
            results.append(
                {
                    "score": float(scores[index]),
                    "chunk": self.chunks[int(index)],
                }
            )

        results.sort(
            key=lambda item: (
                -item["score"],
                item["chunk"]["metadata"]["chunk_index"],
                item["chunk"]["id"],
            )
        )
        return results


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        embedding_model: EmbeddingModel,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        query_embedding = self.embedding_model.encode([query])[0]
        vector_results = self.vector_store.search(query_embedding, top_k=top_k)

        bm25_scores = {result["chunk"]["id"]: result["score"] for result in bm25_results}
        vector_scores = {result["chunk"]["id"]: result["score"] for result in vector_results}
        vector_distances = {result["chunk"]["id"]: result["score"] for result in vector_results}

        normalized_bm25 = _normalize_scores(bm25_scores)
        normalized_vector = _normalize_scores(vector_scores, invert=True)

        merged_results: Dict[str, Dict[str, Any]] = {}

        for result in bm25_results:
            chunk = result["chunk"]
            chunk_id = chunk["id"]
            merged_results[chunk_id] = {
                "chunk": chunk,
                "bm25_score": normalized_bm25.get(chunk_id, 0.0),
                "vector_score": 0.0,
                "vector_distance": None,
                "source": "bm25",
            }

        for result in vector_results:
            chunk = result["chunk"]
            chunk_id = chunk["id"]

            if chunk_id in merged_results:
                merged_results[chunk_id]["vector_score"] = normalized_vector.get(chunk_id, 0.0)
                merged_results[chunk_id]["vector_distance"] = vector_distances.get(chunk_id)
                merged_results[chunk_id]["source"] = "both"
            else:
                merged_results[chunk_id] = {
                    "chunk": chunk,
                    "bm25_score": 0.0,
                    "vector_score": normalized_vector.get(chunk_id, 0.0),
                    "vector_distance": vector_distances.get(chunk_id),
                    "source": "vector",
                }

        final_results: List[Dict[str, Any]] = []
        for item in merged_results.values():
            combined_score = 0.5 * item["bm25_score"] + 0.5 * item["vector_score"]
            final_results.append(
                {
                    "chunk": item["chunk"],
                    "score": combined_score,
                    "hybrid_score": combined_score,
                    "vector_distance": item["vector_distance"],
                    "source": item["source"],
                }
            )

        final_results.sort(
            key=lambda item: (
                -item["score"],
                item["chunk"]["metadata"]["chunk_index"],
                item["chunk"]["id"],
            )
        )
        return final_results[:top_k]
