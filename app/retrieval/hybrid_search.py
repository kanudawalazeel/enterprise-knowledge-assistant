
from typing import Any

from app.retrieval.bm25_search import BM25Searcher
from app.retrieval.vector_store import VectorStore


class HybridSearcher:
    """Combine BM25 and vector retrieval using Reciprocal Rank Fusion."""

    def __init__(
        self,
        bm25_searcher: BM25Searcher,
        vector_store: VectorStore,
        rrf_k: int = 60,
    ):
        self.bm25_searcher = bm25_searcher
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve candidates using keyword and semantic search."""

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("Search limits must be greater than zero.")

        vector_results = self.vector_store.search(
            query=query,
            top_k=candidate_k,
            document_id=document_id,
            file_type=file_type,
        )

        # BM25 currently searches its in-memory index, so filter its
        # results here using the same metadata criteria.
        bm25_results = self.bm25_searcher.search(
            query=query,
            top_k=candidate_k,
        )

        def matches_filters(
            candidate: dict[str, Any],
        ) -> bool:
            metadata = candidate.get("metadata", {})

            if document_id and metadata.get("document_id") != document_id:
                return False

            if file_type:
                candidate_type = str(
                    metadata.get("file_type", "")
                ).lower().lstrip(".")

                if candidate_type != file_type.lower().lstrip("."):
                    return False

            return True

        vector_results = [
            result for result in vector_results
            if matches_filters(result)
        ]

        bm25_results = [
            result for result in bm25_results
            if matches_filters(result)
        ]

        fused: dict[str, dict[str, Any]] = {}

        for results in (bm25_results, vector_results):
            for rank, candidate in enumerate(results, start=1):
                chunk_id = candidate["chunk_id"]

                if chunk_id not in fused:
                    fused[chunk_id] = {
                        **candidate,
                        "rrf_score": 0.0,
                        "retrieved_by": [],
                    }

                fused[chunk_id]["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                source = (
                    "bm25"
                    if results is bm25_results
                    else "vector"
                )

                if source not in fused[chunk_id]["retrieved_by"]:
                    fused[chunk_id]["retrieved_by"].append(source)

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return ranked[:top_k]
