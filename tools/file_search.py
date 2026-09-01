"""RAG tool: embed the query, search Chroma, return cited chunks."""

from __future__ import annotations

from langchain_core.tools import tool

from rag.ingestion import list_ingested_files
from rag.retriever import format_chunks, retrieve_chunks
from rag.vector_store import VectorStoreError


@tool
def file_search(query: str) -> str:
    """Search uploaded documents for passages relevant to the query.

    Use this when the user asks about papers, files, or content they uploaded.
    Returns chunk text plus filename and page metadata for citations.
    """
    if not list_ingested_files():
        return (
            "No documents have been uploaded yet. Ask the user to upload a PDF, "
            "DOCX, or TXT file before searching local documents."
        )
    try:
        chunks = retrieve_chunks(query)
    except VectorStoreError as exc:
        return str(exc)
    if not chunks:
        return (
            "No relevant passages were found in the uploaded documents for this query. "
            "Try a more specific question or a different document."
        )
    return format_chunks(chunks)
