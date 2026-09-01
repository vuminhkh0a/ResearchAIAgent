"""Content digestion: extract → split → embed → load into Chroma."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from loaders.document_loader import load_documents
from rag.vector_store import VectorStoreError, get_vector_store


def _load_index() -> dict[str, str]:
    path = settings.ingested_index_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_index(index: dict[str, str]) -> None:
    settings.ingested_index_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ingested_index_path.write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(path.read_bytes())
    return digest.hexdigest()


def ingest_file(path: str | Path) -> dict:
    """Parse a file, chunk it, embed chunks, and store them in Chroma.

    Skips work when the same filename + content hash was already ingested.
    """
    file_path = Path(path)
    fingerprint = file_fingerprint(file_path)
    index = _load_index()
    if index.get(file_path.name) == fingerprint:
        return {
            "filename": file_path.name,
            "chunks": 0,
            "skipped": True,
            "message": f"`{file_path.name}` is already in the vector store.",
        }

    documents = load_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    prepared: list[Document] = []
    for i, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)
        metadata["filename"] = file_path.name
        metadata["chunk_index"] = i
        metadata["fingerprint"] = fingerprint
        prepared.append(Document(page_content=chunk.page_content, metadata=metadata))

    if not prepared:
        raise VectorStoreError(f"No chunks produced from `{file_path.name}`.")

    try:
        store = get_vector_store()
        store.add_documents(prepared)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to write chunks to ChromaDB: {exc}") from exc

    index[file_path.name] = fingerprint
    _save_index(index)
    return {
        "filename": file_path.name,
        "chunks": len(prepared),
        "skipped": False,
        "message": f"Ingested `{file_path.name}` into {len(prepared)} chunks.",
    }


def list_ingested_files() -> list[str]:
    return sorted(_load_index().keys())


def clear_ingested_index() -> None:
    if settings.ingested_index_path.exists():
        settings.ingested_index_path.unlink()
