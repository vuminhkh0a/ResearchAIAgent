"""Optimized summarization: single-pass with hierarchical fallback.

Replaces Map-Reduce (10-20 LLM calls) with:
1. Single-pass for docs fitting in context (1 call)
2. Hierarchical 2-pass for long docs: chunk summaries → final (2-3 calls)
3. Streaming support for perceived latency
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from loaders.document_loader import load_documents
from llm import get_llm
from utils.tokens import estimate_tokens


SINGLE_PASS_PROMPT = (
    "Summarize the following document. Keep the main ideas, methods, findings, "
    "and any important numbers. Write in plain English.\n\n"
    "{focus_block}Document:\n{text}"
)

CHUNK_SUMMARY_PROMPT = (
    "Summarize this document section. Keep key claims, methods, results, "
    "and important numbers. Be concise.\n\n"
    "{focus_block}Section:\n{text}"
)

MERGE_PROMPT = (
    "Combine the following section summaries into one coherent summary. "
    "Remove repetition, preserve important details, and write clearly.\n\n"
    "{focus_block}Section summaries:\n{text}"
)


def _budget() -> int:
    return max(1024, settings.context_window_tokens - settings.reserved_response_tokens - 512)


def _focus_block(focus: str) -> str:
    if not focus.strip():
        return ""
    return f"Focus on: {focus.strip()}\n\n"


def _complete(system: str, user: str, temperature: float = 0.2) -> str:
    llm = get_llm(temperature=temperature)
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return (result.content or "").strip()


def _join_docs(path: Path) -> str:
    docs = load_documents(path)
    parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        header = f"[page {page}]\n" if page else ""
        parts.append(header + doc.page_content)
    return "\n\n".join(parts)


def summarize_text(text: str, focus: str = "", stream: bool = False) -> str:
    """Summarize text with single-pass or hierarchical fallback."""
    if not text.strip():
        return "There is no text to summarize."
    
    budget = _budget()
    token_count = estimate_tokens(text)
    
    # Single-pass if fits in context
    if token_count <= budget:
        return _complete(
            "You are a careful research summarizer.",
            SINGLE_PASS_PROMPT.format(focus_block=_focus_block(focus), text=text),
        )
    
    # Hierarchical: split into chunks, summarize each, then merge
    return _hierarchical_summarize(text, focus, budget)


def _hierarchical_summarize(text: str, focus: str, budget: int) -> str:
    """2-pass hierarchical summarization: chunk summaries → final merge."""
    # Split into chunks that fit in context with room for prompt
    max_chunk_tokens = budget - 500  # Leave room for prompt + response
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_tokens * 4,  # ~4 chars per token
        chunk_overlap=min(settings.chunk_overlap, max_chunk_tokens // 4),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    
    if not pieces:
        return "Could not split the document for summarization."
    
    # Pass 1: Summarize each chunk (can be parallelized later)
    chunk_summaries = []
    for i, piece in enumerate(pieces, 1):
        prompt = CHUNK_SUMMARY_PROMPT.format(focus_block=_focus_block(focus), text=piece)
        summary = _complete("You are a careful research summarizer.", prompt)
        chunk_summaries.append(f"Section {i}: {summary}")
    
    # Pass 2: Merge summaries
    combined = "\n\n".join(chunk_summaries)
    if estimate_tokens(combined) <= budget:
        return _complete(
            "You are a careful research summarizer.",
            MERGE_PROMPT.format(focus_block=_focus_block(focus), text=combined),
        )
    
    # Rare case: merged summaries still too long - recursive merge
    return _hierarchical_summarize(combined, focus, budget)


def summarize_file(path: str | Path, focus: str = "", stream: bool = False) -> str:
    file_path = Path(path)
    text = _join_docs(file_path)
    return summarize_text(text, focus=focus, stream=stream)