
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "data" / "document_registry.json"


class DocumentRegistry:
    """Manage persistent metadata for indexed documents."""

    def __init__(
        self,
        registry_path: str | Path = REGISTRY_PATH,
    ):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = Lock()
        self._documents: dict[str, dict[str, Any]] = {}

        self._load()

    def _load(self) -> None:
        """Load document metadata from the JSON registry."""

        if not self.registry_path.exists():
            self._documents = {}
            return

        try:
            data = json.loads(
                self.registry_path.read_text(encoding="utf-8")
            )

            if not isinstance(data, dict):
                raise ValueError(
                    "Registry must contain a JSON object."
                )

            self._documents = data

        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not read document registry: "
                f"{self.registry_path}"
            ) from exc

    def _save(self) -> None:
        """Save registry metadata atomically."""

        temporary_path = self.registry_path.with_suffix(".tmp")

        temporary_path.write_text(
            json.dumps(
                self._documents,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        temporary_path.replace(self.registry_path)

    def list_documents(self) -> list[dict[str, Any]]:
        """Return metadata for all registered documents."""

        with self._lock:
            return [
                {
                    "document_id": document_id,
                    **metadata,
                }
                for document_id, metadata
                in self._documents.items()
            ]

    def get_document(
        self,
        document_id: str,
    ) -> dict[str, Any] | None:
        """Return metadata for one document."""

        with self._lock:
            metadata = self._documents.get(document_id)

            if metadata is None:
                return None

            return {
                "document_id": document_id,
                **metadata,
            }

    def add_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        chunk_count: int,
        stored_filename: str,
    ) -> dict[str, Any]:
        """Register a newly indexed document."""

        with self._lock:
            if document_id in self._documents:
                raise ValueError(
                    f"Document ID already exists: {document_id}"
                )

            metadata = {
                "filename": filename,
                "file_type": file_type,
                "chunk_count": chunk_count,
                "stored_filename": stored_filename,
                "ingested_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            self._documents[document_id] = metadata

            try:
                self._save()
            except Exception:
                del self._documents[document_id]
                raise

            return {
                "document_id": document_id,
                **metadata,
            }

    def update_document(
        self,
        document_id: str,
        filename: str,
        file_type: str,
        chunk_count: int,
        stored_filename: str,
    ) -> dict[str, Any]:
        """Update an existing document without changing its ID."""

        with self._lock:
            existing = self._documents.get(document_id)

            if existing is None:
                raise ValueError(
                    f"Document ID not found: {document_id}"
                )

            updated_metadata = {
                **existing,
                "filename": filename,
                "file_type": file_type,
                "chunk_count": chunk_count,
                "stored_filename": stored_filename,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            self._documents[document_id] = updated_metadata

            try:
                self._save()
            except Exception:
                self._documents[document_id] = existing
                raise

            return {
                "document_id": document_id,
                **updated_metadata,
            }

    def remove_document(
        self,
        document_id: str,
    ) -> bool:
        """Remove a document from the registry."""

        with self._lock:
            metadata = self._documents.pop(
                document_id,
                None,
            )

            if metadata is None:
                return False

            try:
                self._save()
            except Exception:
                self._documents[document_id] = metadata
                raise

            return True
