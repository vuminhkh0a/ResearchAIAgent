"""Retrieve top-k chunks for a question embedding, with a token budget."""

from __future__ import annotations

from rag.vector_store import VectorStoreError, get_vector_store
from config.settings import settings
from utils.tokens import estimate_tokens, trim_to_tokens


def retrieve_chunks(query: str, k: int | None = None) -> list[dict]:
    if not query.strip():
        return []

    top_k = k or settings.retrieval_k
    try:
        store = get_vector_store()
        docs = store.similarity_search(query, k=top_k)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"ChromaDB retrieval failed: {exc}") from exc

    results: list[dict] = []
    used_tokens = 0
    for doc in docs:
        text = trim_to_tokens(doc.page_content.strip(), settings.max_context_tokens)
        chunk_tokens = estimate_tokens(text)
        if used_tokens + chunk_tokens > settings.max_context_tokens:
            remaining = settings.max_context_tokens - used_tokens
            if remaining < 50:
                break
            text = trim_to_tokens(text, remaining)
            chunk_tokens = estimate_tokens(text)
        used_tokens += chunk_tokens
        results.append(
            {
                "text": text,
                "filename": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
            }
        )
        if used_tokens >= settings.max_context_tokens:
            break
    return results


def format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant chunks were found in the uploaded documents."

    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.get("page")
        page_label = f"page {page}" if page not in (None, "") else "page n/a"
        lines.append(
            f"[{i}] {chunk['filename']} ({page_label}, chunk {chunk.get('chunk_index', '?')})\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(lines)
