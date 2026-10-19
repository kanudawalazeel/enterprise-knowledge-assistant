from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """Generate semantic embeddings for documents and queries."""

    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Convert one user question into an embedding vector."""
        return self.embed_texts([query])[0]
