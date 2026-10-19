from pathlib import Path
from typing import Any

import chromadb

from app.ingestion.chunker import DocumentChunk
from app.retrieval.embeddings import EmbeddingService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
COLLECTION_NAME = "enterprise_documents"


class VectorStore:
    """Persistent vector database for document chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_directory: str | Path = DEFAULT_DB_PATH,
    ):
        self.embedding_service = embedding_service

        db_path = Path(persist_directory)
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _prepare_metadata(
        chunk: DocumentChunk,
    ) -> dict[str, Any]:
        metadata = dict(chunk.metadata)

        metadata.update(
            {
                "document_id": chunk.document_id,
                "source": chunk.source,
                "file_type": chunk.file_type,
            }
        )

        if chunk.page is not None:
            metadata["page"] = chunk.page

        if chunk.section is not None:
            metadata["section"] = chunk.section

        if chunk.last_updated is not None:
            metadata["last_updated"] = chunk.last_updated

        return {
            key: value
            for key, value in metadata.items()
            if value is not None
            and isinstance(value, (str, int, float, bool))
        }

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> int:
        """Insert or update chunks in the vector store."""

        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                self._prepare_metadata(chunk)
                for chunk in chunks
            ],
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 3,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search vectors, optionally restricted by metadata."""

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.collection.count() == 0:
            return []

        where: dict[str, Any] | None = None

        conditions = []

        if document_id:
            conditions.append({"document_id": document_id})

        if file_type:
            conditions.append(
                {"file_type": file_type.lower().lstrip(".")}
            )

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        query_embedding = self.embedding_service.embed_query(query)

        count = self.collection.count()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            where=where,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        matches = []

        for chunk_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

        return matches

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return self.collection.count()
