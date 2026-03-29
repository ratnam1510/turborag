from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray


FloatMatrix = NDArray[np.float32]
FloatVector = NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Protocol satisfied by any object that can embed a single text string.

    This is the minimal contract expected by :class:`~turborag.hybrid.HybridRetriever`.
    :class:`SentenceTransformerEmbedder` implements this alongside the richer
    ``embed_query`` / ``embed_documents`` surface used by adapters and LangChain.
    """

    def embed(self, text: str) -> FloatVector: ...


class SentenceTransformerEmbedder:
    """Optional embedder wrapper for local sentence-transformers models.

    Satisfies both the minimal :class:`Embedder` protocol (``embed``) **and** the
    ``embed_query`` / ``embed_documents`` surface expected by LangChain and the
    compatibility adapters.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerEmbedder requires installing turborag[embed] or sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(model_name)

    # -- Minimal Embedder protocol ------------------------------------------

    def embed(self, text: str) -> FloatVector:
        """Embed a single text string (satisfies :class:`Embedder` protocol)."""
        return self.embed_query(text)

    # -- Richer surface used by adapters / LangChain -------------------------

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> FloatVector:
        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(vector, dtype=np.float32)
