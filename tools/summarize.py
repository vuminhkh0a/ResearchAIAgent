"""Agent-facing summarization tool. Parsing and splitting stay internal."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from config.settings import settings
from rag.ingestion import list_ingested_files
from summarization.map_reduce import summarize_file


@tool
def summarize(filename: str = "", focus: str = "") -> str:
    """Summarize one uploaded document using Map-Reduce when it is long.

    Pass the exact filename when several files are uploaded. Leave filename
    empty only if a single document is available. Optional focus steers the
    summary (for example: "methods" or "limitations").
    """
    available = list_ingested_files()
    if not available:
        return "No documents have been uploaded. Ask the user to upload a file first."

    target = filename.strip()
    if not target:
        if len(available) == 1:
            target = available[0]
        else:
            listed = ", ".join(available)
            return (
                "Several documents are uploaded. Call summarize again with one "
                f"filename from: {listed}"
            )

    path = settings.upload_dir / target
    if not path.exists():
        listed = ", ".join(available)
        return f"`{target}` was not found in uploads. Available files: {listed}"

    try:
        summary = summarize_file(path, focus=focus)
    except Exception as exc:  # noqa: BLE001
        return f"Summarization failed: {exc}"
    return f"Summary of {target}:\n\n{summary}"
