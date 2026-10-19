
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.chunker import chunk_documents
from app.ingestion.parsers import parse_document
from app.ingestion.registry import DocumentRegistry


router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "original"

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".md",
    ".html",
    ".htm",
}

MAX_FILE_SIZE = 20 * 1024 * 1024


@router.get("/documents")
def list_documents():
    """List all registered documents."""

    registry = DocumentRegistry()
    documents = registry.list_documents()

    return {
        "total_documents": len(documents),
        "documents": documents,
    }


@router.put("/documents/{document_id}")
async def update_document(
    document_id: str,
    request: Request,
    file: UploadFile = File(...),
):
    """Replace an existing document and refresh its indexes."""

    registry = DocumentRegistry()
    pipeline = request.app.state.rag_pipeline

    existing_record = registry.get_document(document_id)

    if existing_record is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found in the registry.",
        )

    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()

    if not filename or extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
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
            detail="File exceeds the 20 MB limit.",
        )

    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    new_stored_filename = f"{uuid4().hex}{extension}"
    new_path = DOCUMENTS_DIR / new_stored_filename

    old_stored_filename = existing_record.get("stored_filename")
    old_path = (
        DOCUMENTS_DIR / old_stored_filename
        if old_stored_filename
        else None
    )

    old_chunks = [
        chunk
        for chunk in pipeline.chunks
        if chunk.document_id == document_id
    ]

    new_chunks = []

    try:
        # Parse and chunk the replacement before modifying the active indexes.
        new_path.write_bytes(content)

        parsed_documents = parse_document(new_path)

        for document in parsed_documents:
            document.document_id = document_id
            document.source = filename

        new_chunks = chunk_documents(
            parsed_documents,
            chunk_size=500,
            chunk_overlap=80,
        )

        if not new_chunks:
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from the document.",
            )

        # Save the old vector records so an indexing failure can be rolled back.
        old_vector_records = pipeline.vector_store.collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas", "embeddings"],
        )

        # Replace vectors with the newly generated chunks.
        if old_vector_records["ids"]:
            pipeline.vector_store.collection.delete(
                ids=old_vector_records["ids"]
            )

        pipeline.vector_store.add_chunks(new_chunks)

        # Refresh the in-memory chunks and keyword index.
        pipeline.chunks = [
            chunk
            for chunk in pipeline.chunks
            if chunk.document_id != document_id
        ]
        pipeline.chunks.extend(new_chunks)
        pipeline.bm25_searcher.build_index(pipeline.chunks)

        # Update registry metadata while keeping the same document ID.
        registry.update_document(
            document_id=document_id,
            filename=filename,
            file_type=extension.lstrip("."),
            chunk_count=len(new_chunks),
            stored_filename=new_stored_filename,
        )

        # Only remove the old file after the new version is registered.
        if old_path and old_path.resolve().parent == DOCUMENTS_DIR.resolve():
            old_path.unlink(missing_ok=True)

        return {
            "message": "Document updated successfully.",
            "document_id": document_id,
            "original_filename": filename,
            "chunks_created": len(new_chunks),
        }

    except HTTPException:
        new_path.unlink(missing_ok=True)
        raise

    except Exception as exc:
        print(f"Document update failed: {exc}")

        # Best-effort rollback of vector records and in-memory indexes.
        try:
            current_records = pipeline.vector_store.collection.get(
                where={"document_id": document_id},
                include=[],
            )

            if current_records["ids"]:
                pipeline.vector_store.collection.delete(
                    ids=current_records["ids"]
                )

            if old_chunks:
                pipeline.vector_store.add_chunks(old_chunks)

            pipeline.chunks = [
                chunk
                for chunk in pipeline.chunks
                if chunk.document_id != document_id
            ]
            pipeline.chunks.extend(old_chunks)
            pipeline.bm25_searcher.build_index(pipeline.chunks)

            # Restore registry metadata if it was already changed.
            current_record = registry.get_document(document_id)

            if current_record != existing_record:
                registry.update_document(
                    document_id=document_id,
                    filename=existing_record["filename"],
                    file_type=existing_record["file_type"],
                    chunk_count=existing_record["chunk_count"],
                    stored_filename=existing_record["stored_filename"],
                )

        except Exception as rollback_exc:
            print(f"Document update rollback failed: {rollback_exc}")

        new_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Document update failed. Check the server logs.",
        ) from exc

    finally:
        await file.close()


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    request: Request,
):
    """Delete a document from storage, registry, and indexes."""

    pipeline = request.app.state.rag_pipeline
    registry = DocumentRegistry()

    record = registry.get_document(document_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found in the registry.",
        )

    matching_chunks = [
        chunk
        for chunk in pipeline.chunks
        if chunk.document_id == document_id
    ]

    pipeline.vector_store.collection.delete(
        where={"document_id": document_id}
    )

    pipeline.chunks = [
        chunk
        for chunk in pipeline.chunks
        if chunk.document_id != document_id
    ]

    pipeline.bm25_searcher.build_index(pipeline.chunks)

    stored_filename = record.get("stored_filename")

    if stored_filename:
        stored_path = (DOCUMENTS_DIR / stored_filename).resolve()

        if stored_path.parent == DOCUMENTS_DIR.resolve():
            stored_path.unlink(missing_ok=True)

    registry.remove_document(document_id)

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "chunks_removed": len(matching_chunks),
    }
