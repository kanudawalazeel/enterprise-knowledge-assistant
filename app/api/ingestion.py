
from pathlib import Path
from uuid import uuid4

import pymupdf
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.parsers import parse_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.registry import DocumentRegistry

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "original"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".html", ".htm"}
MAX_FILE_SIZE = 20 * 1024 * 1024


def _safe_unlink(path: Path) -> None:
    """Attempt cleanup without masking the original exception."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"Warning: Could not remove temporary file {path.name}: {exc}")


@router.post("/ingest")
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
):
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Supported file types: PDF, DOCX, Markdown, and HTML.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 20 MB upload limit.",
        )

    # Validate PDF bytes before creating a file on disk.
    # This prevents malformed PDFs from leaving a locked upload behind.
    if extension == ".pdf":
        try:
            with pymupdf.open(stream=content, filetype="pdf"):
                pass
        except Exception as exc:
            print(f"Invalid PDF upload {filename}: {type(exc).__name__}")
            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded PDF is invalid or corrupted. "
                    "Please upload a valid PDF file."
                ),
            ) from exc

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    document_id = uuid4().hex
    stored_path = DOCUMENTS_DIR / f"{document_id}{extension}"

    registry = DocumentRegistry()
    pipeline = request.app.state.rag_pipeline

    chunks = []
    indexing_started = False

    try:
        stored_path.write_bytes(content)

        print(f"Parsing uploaded file: {filename}")
        documents = parse_document(stored_path)

        for document in documents:
            document.document_id = document_id
            document.source = filename

        chunks = chunk_documents(
            documents,
            chunk_size=500,
            chunk_overlap=80,
        )

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from the document.",
            )

        # Mark indexing as started before any index mutation,
        # so partial failures can trigger rollback.
        indexing_started = True

        pipeline.vector_store.add_chunks(chunks)
        pipeline.chunks.extend(chunks)
        pipeline.bm25_searcher.build_index(pipeline.chunks)

        registry.add_document(
            document_id=document_id,
            filename=filename,
            file_type=extension.lstrip("."),
            chunk_count=len(chunks),
            stored_filename=stored_path.name,
        )

        return {
            "message": "Document ingested successfully.",
            "document_id": document_id,
            "original_filename": filename,
            "parsed_records": len(documents),
            "chunks_created": len(chunks),
        }

    except HTTPException:
        if indexing_started:
            _rollback_ingestion(
                pipeline, registry, document_id, chunks
            )
        _safe_unlink(stored_path)
        raise

    except Exception as exc:
        print(
            f"Document ingestion failed for {filename}: "
            f"{type(exc).__name__}: {exc}"
        )

        if indexing_started:
            _rollback_ingestion(
                pipeline, registry, document_id, chunks
            )

        _safe_unlink(stored_path)

        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the server logs.",
        ) from exc

    finally:
        await file.close()


def _rollback_ingestion(pipeline, registry, document_id, chunks) -> None:
    """Best-effort rollback of vector, memory, BM25, and registry state."""
    try:
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if chunk_ids:
            pipeline.vector_store.collection.delete(ids=chunk_ids)

        pipeline.chunks = [
            chunk
            for chunk in pipeline.chunks
            if chunk.document_id != document_id
        ]

        pipeline.bm25_searcher.build_index(pipeline.chunks)
        registry.remove_document(document_id)

    except Exception as exc:
        print(
            f"Ingestion rollback failed for {document_id}: "
            f"{type(exc).__name__}: {exc}"
        )
