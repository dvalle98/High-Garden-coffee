"""Utilidades para persistir y consultar la base de conocimiento FAISS."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np


@dataclass
class SearchHit:
    """Resultado individual de una búsqueda vectorial."""

    text: str
    metadata: Dict[str, str]
    score: float


class LocalKnowledgeBase:
    """Wrapper ligero sobre un índice FAISS serializable."""

    def __init__(self, index: faiss.Index, texts: Sequence[str], metadata: Sequence[Dict[str, str]]):
        self.index = index
        self.texts = list(texts)
        self.metadata = list(metadata)

    @classmethod
    def build(cls, embeddings: np.ndarray, texts: Sequence[str], metadata: Sequence[Dict[str, str]]) -> "LocalKnowledgeBase":
        if embeddings.ndim != 2:
            raise ValueError("La matriz de embeddings debe ser 2D (n_chunks, dim).")
        if embeddings.shape[0] != len(texts) or len(texts) != len(metadata):
            raise ValueError("Embeddings, textos y metadatos deben tener el mismo largo.")

        matrix = embeddings.astype("float32")
        faiss.normalize_L2(matrix)

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(index=index, texts=texts, metadata=metadata)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index": faiss.serialize_index(self.index),
            "texts": self.texts,
            "metadata": self.metadata,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "LocalKnowledgeBase":
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)
        index = faiss.deserialize_index(payload["index"])
        texts = payload["texts"]
        metadata = payload["metadata"]
        return cls(index=index, texts=texts, metadata=metadata)

    def search(self, query_vector: Sequence[float], top_k: int = 4) -> List[SearchHit]:
        if self.index.ntotal == 0:
            return []
        top_k = max(1, top_k)
        vector = np.array([query_vector], dtype="float32")
        faiss.normalize_L2(vector)
        distances, indices = self.index.search(vector, top_k)

        hits: List[SearchHit] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            hits.append(
                SearchHit(
                    text=self.texts[idx],
                    metadata=self.metadata[idx],
                    score=float(score),
                )
            )
        return hits
