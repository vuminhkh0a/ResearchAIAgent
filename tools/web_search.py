"""Lightweight web search. Default provider: DuckDuckGo. Optional: Tavily."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from config.settings import settings


@tool
def web_search(query: str) -> str:
    """Search the public web for recent or external information.

    Use this for current events, papers not in the uploads, or facts that
    are unlikely to be in the local documents. Returns titles, snippets, and URLs.
    """
    if not query.strip():
        return "Provide a non-empty search query."
    try:
        if settings.search_provider == "tavily":
            return _search_tavily(query)
        return _search_duckduckgo(query)
    except Exception as exc:  # noqa: BLE001
        return (
            "Web search failed. Check your network or SEARCH_PROVIDER settings. "
            f"Details: {exc}"
        )


def _search_duckduckgo(query: str) -> str:
    from ddgs import DDGS

    rows = list(
        DDGS().text(query, max_results=settings.search_max_results)
    )
    if not rows:
        return "No web search results were found."
    lines = []
    for i, row in enumerate(rows, start=1):
        title = row.get("title") or "Untitled"
        href = row.get("href") or row.get("url") or ""
        body = row.get("body") or row.get("snippet") or ""
        lines.append(f"[{i}] {title}\n{body}\nURL: {href}")
    return "\n\n".join(lines)


def _search_tavily(query: str) -> str:
    if not settings.tavily_api_key:
        return (
            "SEARCH_PROVIDER is tavily but TAVILY_API_KEY is missing. "
            "Set the key in .env or switch SEARCH_PROVIDER=duckduckgo."
        )
    response = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": settings.search_max_results,
        },
        timeout=20.0,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        return "No web search results were found."
    lines = []
    for i, row in enumerate(results, start=1):
        title = row.get("title") or "Untitled"
        url = row.get("url") or ""
        content = row.get("content") or ""
        lines.append(f"[{i}] {title}\n{content}\nURL: {url}")
    return "\n\n".join(lines)
