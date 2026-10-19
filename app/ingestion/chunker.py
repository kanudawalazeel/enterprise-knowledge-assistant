
import re
from typing import Any

import tiktoken
from pydantic import BaseModel, Field

from app.ingestion.schemas import ParsedDocument


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    file_type: str
    text: str
    page: int | None = None
    section: str | None = None
    last_updated: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ENCODING = tiktoken.get_encoding("cl100k_base")


def _split_into_sections(
    text: str,
) -> list[tuple[str | None, str]]:
    """Split Markdown documents into heading-based sections."""
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def save_current() -> None:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    for line in text.splitlines():
        heading_match = re.match(
            r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$",
            line,
        )

        if heading_match:
            save_current()
            current_heading = heading_match.group(2).strip()
            current_lines = [line.strip()]
        else:
            current_lines.append(line)

    save_current()

    if not sections and text.strip():
        sections.append((None, text.strip()))

    return sections


def _is_table_block(text: str) -> bool:
    """Detect a Markdown table containing a header separator."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 2:
        return False

    if not all("|" in line for line in lines):
        return False

    separator_pattern = re.compile(
        r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$"
    )

    return any(separator_pattern.match(line) for line in lines)


def _split_table_block(
    table_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split large Markdown tables by rows.
    Repeat the table header and separator in each resulting chunk.
    """
    lines = [
        line.rstrip()
        for line in table_text.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return [table_text.strip()]

    separator_index = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(
                r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$",
                line.strip(),
            )
        ),
        None,
    )

    if separator_index is None or separator_index == 0:
        return _split_long_text(
            table_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    header = lines[: separator_index + 1]
    rows = lines[separator_index + 1 :]

    header_text = "\n".join(header)
    header_tokens = len(ENCODING.encode(header_text))

    # Reserve space for the repeated table header.
    row_budget = max(1, chunk_size - header_tokens)

    result: list[str] = []
    current_rows: list[str] = []

    def save_table_rows() -> None:
        if current_rows:
            result.append(
                header_text + "\n" + "\n".join(current_rows)
            )

    for row in rows:
        candidate_rows = current_rows + [row]
        candidate_text = (
            header_text + "\n" + "\n".join(candidate_rows)
        )
        candidate_tokens = len(ENCODING.encode(candidate_text))

        if candidate_tokens <= chunk_size:
            current_rows.append(row)
            continue

        if current_rows:
            save_table_rows()
            previous_rows = current_rows
            current_rows = []

            # Keep the last row as lightweight overlap between chunks.
            if chunk_overlap > 0 and previous_rows:
                last_row = previous_rows[-1]
                if len(
                    ENCODING.encode(header_text + "\n" + last_row)
                ) < chunk_size:
                    current_rows = [last_row]

        current_text = header_text + "\n" + row

        if len(ENCODING.encode(current_text)) <= chunk_size:
            current_rows.append(row)
        else:
            # A single row is too large; split it safely as text.
            if current_rows:
                save_table_rows()
                current_rows = []

            result.extend(
                _split_long_text(
                    row,
                    chunk_size=row_budget,
                    chunk_overlap=chunk_overlap,
                )
            )

    save_table_rows()

    return result or [table_text.strip()]


def _split_long_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split text while preserving paragraphs and table structures."""
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
    ]

    result: list[str] = []
    current_parts: list[str] = []

    def current_text() -> str:
        return "\n\n".join(current_parts).strip()

    def flush_current() -> None:
        if current_parts:
            result.append(current_text())
            current_parts.clear()

    for block in blocks:
        if _is_table_block(block):
            flush_current()

            table_tokens = len(ENCODING.encode(block))

            if table_tokens <= chunk_size:
                result.append(block)
            else:
                result.extend(
                    _split_table_block(
                        block,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
            continue

        candidate = (
            f"{current_text()}\n\n{block}"
            if current_parts
            else block
        )

        if len(ENCODING.encode(candidate)) <= chunk_size:
            current_parts.append(block)
            continue

        flush_current()

        block_tokens = ENCODING.encode(block)

        if len(block_tokens) <= chunk_size:
            current_parts.append(block)
            continue

        step = chunk_size - chunk_overlap

        for start in range(0, len(block_tokens), step):
            end = min(start + chunk_size, len(block_tokens))
            piece = ENCODING.decode(
                block_tokens[start:end]
            ).strip()

            if piece:
                result.append(piece)

            if end == len(block_tokens):
                break

    flush_current()

    return result


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    """Create chunks with section metadata and table preservation."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    text = document.text.strip()

    if not text:
        return []

    sections = _split_into_sections(text)
    chunks: list[DocumentChunk] = []

    for heading, section_text in sections:
        pieces = _split_long_text(
            section_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for piece in pieces:
            chunk_number = len(chunks) + 1

            page_part = (
                f"_page_{document.page}"
                if document.page is not None
                else ""
            )

            chunk_id = (
                f"{document.document_id}"
                f"{page_part}_chunk_{chunk_number:04d}"
            )

            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_number": chunk_number,
                    "token_count": len(ENCODING.encode(piece)),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "contains_table": _is_table_block(piece),
                }
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    file_type=document.file_type,
                    text=piece,
                    page=document.page,
                    section=heading or document.section,
                    last_updated=document.last_updated,
                    metadata=metadata,
                )
            )

    return chunks


def chunk_documents(
    documents: list[ParsedDocument],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    """Chunk multiple parsed documents."""
    all_chunks: list[DocumentChunk] = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return all_chunks
