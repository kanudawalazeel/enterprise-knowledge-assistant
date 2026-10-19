
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.api.schemas import QueryRequest
from app.api.ingestion import router as ingestion_router
from app.api.document_management import router as document_management_router
from generation.schemas import AnswerResponse
from app.rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

RETRYABLE_ERROR_NAMES = {
    "ServerError",
    "ResourceExhausted",
    "DeadlineExceeded",
    "ServiceUnavailable",
    "InternalServerError",
    "TooManyRequests",
    "ConnectError",
    "ReadTimeout",
    "ConnectTimeout",
}


def is_temporary_service_error(exc: Exception) -> bool:
    """Identify temporary provider or network errors safely."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    error_name = type(exc).__name__

    if error_name in RETRYABLE_ERROR_NAMES:
        return True

    status_code = getattr(exc, "code", None)
    if not isinstance(status_code, int):
        status_code = getattr(exc, "status_code", None)

    return status_code in RETRYABLE_STATUS_CODES


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline when the application starts."""
    logger.info("Initializing Enterprise Knowledge Assistant...")

    app.state.rag_pipeline = RAGPipeline()

    logger.info("Enterprise Knowledge Assistant is ready.")

    yield

    logger.info("Shutting down Enterprise Knowledge Assistant...")


app = FastAPI(
    title="Enterprise Knowledge Assistant",
    description=(
        "An enterprise RAG application that answers questions "
        "from company documents with citations, hybrid search, "
        "reranking, and metadata filtering."
    ),
    version="0.4.0",
    lifespan=lifespan,
)


# Register API routers
app.include_router(
    ingestion_router,
    tags=["Documents"],
)

app.include_router(
    document_management_router,
    tags=["Documents"],
)


@app.get("/")
def home():
    """Return basic application information."""
    return {
        "message": "Welcome to the Enterprise Knowledge Assistant API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check():
    """Check whether the API and RAG pipeline are initialized."""
    pipeline = getattr(app.state, "rag_pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized.",
        )

    return {
        "status": "healthy",
        "pipeline_initialized": True,
        "indexed_chunks": len(pipeline.chunks),
    }


@app.post("/query", response_model=AnswerResponse)
def query_documents(request: QueryRequest):
    """
    Answer a question using retrieved document evidence.

    Optional filters:
    - document_id: Search within one registered document.
    - file_type: Restrict results to a file type such as pdf or docx.
    """
    pipeline = getattr(app.state, "rag_pipeline", None)

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized.",
        )

    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
            file_type=request.file_type,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except HTTPException:
        # Preserve any intentional HTTP errors raised by the pipeline.
        raise

    except Exception as exc:
        # Never expose raw provider exception messages to API clients.
        logger.error(
            "Query processing failed. Error type: %s",
            type(exc).__name__,
        )

        if is_temporary_service_error(exc):
            raise HTTPException(
                status_code=503,
                detail=(
                    "The answer generation service is temporarily "
                    "unavailable. Please try again later."
                ),
            ) from exc

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred while processing "
                "the query."
            ),
        ) from exc
