from __future__ import annotations

from typing import Any, Dict, List

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: List[Dict[str, Any]] = []

    def add(self, embeddings: List[List[float]], chunks: List[Dict[str, Any]]) -> None:
        if not embeddings:
            return

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {vectors.shape}")

        if len(chunks) != len(vectors):
            raise ValueError("Embeddings and chunks must have the same length.")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        query_vector = np.asarray([query_embedding], dtype="float32")
        if query_vector.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dim}, got {query_vector.shape[1]}"
            )

        top_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, top_k)

        results: List[Dict[str, Any]] = []
        for score, index in zip(distances[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[index]
            results.append(
                {
                    "score": float(score),
                    "chunk": chunk,
                }
            )

        return results
