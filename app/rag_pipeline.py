
from pathlib import Path
from typing import Any

from app.ingestion.parsers import parse_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.registry import DocumentRegistry

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_search import BM25Searcher
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import Reranker

from generation.answer_generator import AnswerGenerator
from generation.schemas import AnswerResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "original"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".html", ".htm"}


class RAGPipeline:
    """Coordinate document ingestion, retrieval, and answer generation."""

    def __init__(self):
        print("Initializing RAG pipeline...")

        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore(self.embedding_service)
        self.bm25_searcher = BM25Searcher()

        self.registry = DocumentRegistry()
        self.chunks = self._load_documents()

        # Rebuild the keyword index from the currently registered documents.
        self.bm25_searcher.build_index(self.chunks)

        # Synchronize ChromaDB with the chunks loaded from the registry.
        self._synchronize_vector_store()

        self.hybrid_searcher = HybridSearcher(
            bm25_searcher=self.bm25_searcher,
            vector_store=self.vector_store,
        )

        self.reranker = Reranker()
        self.answer_generator = AnswerGenerator()

        print("RAG pipeline initialized successfully.")

    def _load_documents(self) -> list[Any]:
        """Load only documents recorded in the persistent registry."""

        if not DOCUMENTS_DIR.exists():
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

        records = self.registry.list_documents()
        all_chunks = []

        for record in records:
            document_id = record["document_id"]
            stored_filename = record.get("stored_filename")
            original_filename = record.get("filename", stored_filename)

            if not stored_filename:
                print(
                    f"Skipping registry record {document_id}: "
                    "stored filename is missing."
                )
                continue

            file_path = (DOCUMENTS_DIR / stored_filename).resolve()

            # Prevent registry metadata from pointing outside the document folder.
            if file_path.parent != DOCUMENTS_DIR.resolve():
                print(
                    f"Skipping document {document_id}: "
                    "invalid stored file path."
                )
                continue

            if not file_path.is_file():
                print(
                    f"Skipping document {document_id}: "
                    f"file not found ({stored_filename})."
                )
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                print(
                    f"Skipping document {document_id}: "
                    "unsupported file extension."
                )
                continue

            try:
                print(f"Loading registered document: {original_filename}")

                documents = parse_document(file_path)

                # Restore the public ID and original filename for stable
                # chunk IDs and meaningful citation metadata.
                for document in documents:
                    document.document_id = document_id
                    document.source = original_filename

                chunks = chunk_documents(
                    documents,
                    chunk_size=500,
                    chunk_overlap=80,
                )

                all_chunks.extend(chunks)

                print(
                    f"Loaded {len(chunks)} chunks from "
                    f"{original_filename}"
                )

            except Exception as exc:
                print(
                    f"Could not load {original_filename}: {exc}"
                )

        print(f"Total registered chunks loaded: {len(all_chunks)}")
        return all_chunks

    def _synchronize_vector_store(self) -> None:
        """Remove stale vectors and upsert the currently loaded chunks."""

        collection = self.vector_store.collection

        existing = collection.get(include=[])
        existing_ids = set(existing.get("ids", []))

        valid_ids = {chunk.chunk_id for chunk in self.chunks}
        stale_ids = list(existing_ids - valid_ids)

        if stale_ids:
            print(f"Removing {len(stale_ids)} stale vector records.")
            collection.delete(ids=stale_ids)

        if self.chunks:
            count = self.vector_store.add_chunks(self.chunks)
            print(f"Synchronized {count} chunks with ChromaDB.")

        print(f"Vector store contains {collection.count()} chunks.")

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
        file_type: str | None = None,
    ) -> AnswerResponse:
        """Answer a question using optionally filtered document evidence."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        if document_id and self.registry.get_document(document_id) is None:
            raise ValueError("The specified document_id is not registered.")

        if file_type:
            file_type = file_type.lower().lstrip(".")

            if file_type not in {"pdf", "docx", "md", "html", "htm"}:
                raise ValueError("Unsupported file_type filter.")

        if not self.chunks:
            return AnswerResponse(
                answer=(
                    "I could not find sufficient evidence in the "
                    "available documents to answer this question."
                ),
                citations=[],
                confidence=0.0,
                is_answerable=False,
            )

        candidates = self.hybrid_searcher.search(
            query=question,
            top_k=max(top_k, 5),
            candidate_k=20,
            document_id=document_id,
            file_type=file_type,
        )

        if not candidates:
            return self.answer_generator.generate(
                query=question,
                candidates=[],
            )

        ranked_candidates = self.reranker.rerank(
            query=question,
            candidates=candidates,
            top_k=top_k,
        )

        return self.answer_generator.generate(
            query=question,
            candidates=ranked_candidates,
        )
