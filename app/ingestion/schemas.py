from pydantic import BaseModel, Field

class ParsedDocument(BaseModel):
    document_id: str
    source: str
    file_type: str
    text: str
    page: int | None = None
    section: str | None = None
    last_updated: str | None = None
    metadata: dict = Field(default_factory=dict)
