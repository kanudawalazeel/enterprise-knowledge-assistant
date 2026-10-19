
from typing import Any

from sentence_transformers import CrossEncoder


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Rerank retrieved document chunks against a user question."""

    def __init__(self, model_name: str = RERANKER_MODEL):
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker model loaded successfully.")

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Sort candidate chunks by cross-encoder relevance score."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not candidates:
            return []

        pairs = [
            (query, candidate["text"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = []

        for candidate, score in zip(candidates, scores):
            result = dict(candidate)
            result["metadata"] = dict(candidate.get("metadata") or {})
            result["rerank_score"] = float(score)
            reranked.append(result)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]
