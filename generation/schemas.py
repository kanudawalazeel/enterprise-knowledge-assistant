from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str
    page: int | None = None
    section: str | None = None
    chunk_id: str
    quote: str = Field(
        description="A short supporting quotation from the retrieved evidence."
    )


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    is_answerable: bool
