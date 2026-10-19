
import re
from typing import Any

from rank_bm25 import BM25Okapi

from app.ingestion.chunker import DocumentChunk


class BM25Searcher:
    """Keyword search over document chunks using BM25."""

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build the BM25 index from the current chunks."""

        self.chunks = list(chunks)
        self.tokenized_corpus = [
            self.tokenize(chunk.text)
            for chunk in self.chunks
        ]

        if not self.chunks:
            self.bm25 = None
            return

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search keywords and optionally filter by metadata."""

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.bm25 is None:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            if scores[index] <= 0:
                continue

            if document_id and chunk.document_id != document_id:
                continue

            if file_type:
                normalized_type = file_type.lower().lstrip(".")

                if chunk.file_type.lower().lstrip(".") != normalized_type:
                    continue

            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        **chunk.metadata,
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "file_type": chunk.file_type,
                        "page": chunk.page,
                        "section": chunk.section,
                    },
                    "bm25_score": float(scores[index]),
                }
            )

            if len(results) >= top_k:
                break

        return results
