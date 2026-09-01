"""Persistent Chroma vector store for document chunks and embeddings."""

from __future__ import annotations

from langchain_chroma import Chroma

from config.settings import settings
from rag.embeddings import get_embeddings

_store: Chroma | None = None
COLLECTION_NAME = "research_docs"


class VectorStoreError(RuntimeError):
    pass


def get_vector_store() -> Chroma:
    global _store
    if _store is None:
        try:
            _store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(settings.chroma_dir),
                embedding_function=get_embeddings(),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                "ChromaDB could not be opened. Delete data/chroma and try again "
                f"if the store is corrupted. Details: {exc}"
            ) from exc
    return _store


def reset_vector_store() -> None:
    """Remove all stored chunks so documents can be ingested again."""
    try:
        store = get_vector_store()
        data = store.get()
        ids = data.get("ids") or []
        if ids:
            store.delete(ids=ids)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Could not clear ChromaDB: {exc}") from exc
