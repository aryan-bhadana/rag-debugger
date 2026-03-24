from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self) -> None:
        self.model_name = "all-MiniLM-L6-v2"
        self.model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._get_model().encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
