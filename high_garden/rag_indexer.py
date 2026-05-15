"""Construye el vector store FAISS usado por el chatbot (pipeline RAG)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Any

import nbformat
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from rag_store import LocalKnowledgeBase

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_OUTPUT = BASE_DIR / "vector_store.faiss"

NOTEBOOKS = [
    BASE_DIR / "01_eda_storytelling.ipynb",
    BASE_DIR / "02_forecasting.ipynb",
    BASE_DIR / "03_segmentation_1.ipynb",
]

TEXT_SOURCES = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "RESUMEN_EJECUTIVO.md",
]


def _path_label(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_notebook(path: Path) -> str:
    nb = nbformat.read(path, as_version=4)
    blocks: List[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") in {"markdown", "raw"}:
            blocks.append(cell.get("source", ""))
    return "\n\n".join(blocks).strip()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_corpus() -> List[Dict[str, str]]:
    corpus: List[Dict[str, str]] = []
    for path in TEXT_SOURCES:
        if path.exists():
            corpus.append({"source": _path_label(path), "text": _read_text_file(path)})
    for path in NOTEBOOKS:
        if path.exists():
            corpus.append({"source": _path_label(path), "text": _read_notebook(path)})
    return corpus


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser positivo")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo")
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_size debe ser mayor que chunk_overlap")

    chunks: List[str] = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def chunk_corpus(corpus: List[Dict[str, str]], chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for doc in corpus:
        if not doc["text"]:
            continue
        parts = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, part in enumerate(parts):
            chunks.append(
                {
                    "text": part,
                    "metadata": {"source": doc["source"], "chunk_id": str(idx)},
                }
            )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el vector store FAISS para RAG.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Ruta destino del archivo .faiss")
    parser.add_argument("--chunk-size", type=int, default=1100, help="Tamaño del chunk en caracteres")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap entre chunks en caracteres")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Modelo de embeddings OpenAI a utilizar",
    )
    args = parser.parse_args()

    load_dotenv()
    corpus = load_corpus()
    if not corpus:
        raise RuntimeError("No se encontraron fuentes para construir la base de conocimiento.")

    chunks = chunk_corpus(corpus, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    if not chunks:
        raise RuntimeError("No se generaron chunks de texto. Revisa las fuentes.")

    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    embed_model = OpenAIEmbeddings(model=args.embedding_model)
    embeddings = embed_model.embed_documents(texts)
    matrix = np.array(embeddings, dtype="float32")

    kb = LocalKnowledgeBase.build(matrix, texts, metadata)
    kb.save(args.output)

    print("Vector store construido correctamente ✅")
    print(f"- Fuentes procesadas: {len(corpus)}")
    print(f"- Total de chunks: {len(chunks)}")
    print(f"- Archivo generado: {args.output}")


if __name__ == "__main__":
    main()
