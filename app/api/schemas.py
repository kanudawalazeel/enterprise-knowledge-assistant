
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to answer using company documents",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of final retrieved chunks to use",
    )

    document_id: str | None = Field(
        default=None,
        description="Restrict retrieval to one registered document ID",
    )

    file_type: str | None = Field(
        default=None,
        description="Optional file type filter: pdf, docx, md, html, or htm",
    )
