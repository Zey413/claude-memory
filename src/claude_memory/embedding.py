"""Optional embedding support for semantic search.

Requires the ``embeddings`` extra: ``pip install claude-memory[embeddings]``
(sentence-transformers + numpy).  Everything in this module is designed to
degrade gracefully when those packages are not installed.
"""
from __future__ import annotations

import logging

__all__ = [
    "is_available",
    "EmbeddingEngine",
]

logger = logging.getLogger(__name__)

_MODEL_ID = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


def is_available() -> bool:
    """Check if embedding dependencies are installed."""
    try:
        import numpy  # noqa: F401
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


class EmbeddingEngine:
    """Lazy-loading embedding engine using sentence-transformers.

    Uses a singleton pattern so the (heavy) model is loaded at most once.
    The model itself is loaded lazily on the first ``encode`` call.
    """

    _instance: EmbeddingEngine | None = None  # Singleton

    def __init__(self) -> None:
        self._model = None  # Lazy load

    @classmethod
    def get_instance(cls) -> EmbeddingEngine:
        """Return the shared singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_ID)

    # -- public API -----------------------------------------------------------

    def encode(self, text: str):
        """Encode *text* to a 384-dim normalised vector.

        Returns a ``numpy.ndarray`` of shape ``(384,)``.
        """
        self._load_model()
        return self._model.encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]):
        """Encode multiple texts efficiently.

        Returns a ``numpy.ndarray`` of shape ``(len(texts), 384)``.
        """
        self._load_model()
        return self._model.encode(texts, normalize_embeddings=True, batch_size=32)

    @staticmethod
    def cosine_similarity(a, b) -> float:
        """Compute cosine similarity between two normalised vectors."""
        import numpy as np
        return float(np.dot(a, b))  # Already L2-normalised

    @staticmethod
    def serialize(vector) -> bytes:
        """Serialize a numpy array to bytes for SQLite BLOB storage."""
        import numpy as np
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def deserialize(data: bytes):
        """Deserialize bytes back to a numpy array."""
        import numpy as np
        return np.frombuffer(data, dtype=np.float32)
