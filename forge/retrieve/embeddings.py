"""Embedding generation using sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Generate embeddings for text using sentence-transformers."""

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text.

        Returns:
            1-D numpy float32 array.
        """
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)
