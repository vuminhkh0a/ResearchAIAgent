"""Ollama availability checks with user-facing error messages."""

from __future__ import annotations

import httpx

from config.settings import settings


class OllamaError(RuntimeError):
    pass


def check_ollama() -> tuple[bool, str]:
    """Return (ok, message) after checking the Ollama server and Qwen3 model."""
    tags_url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        response = httpx.get(tags_url, timeout=5.0)
        response.raise_for_status()
    except httpx.HTTPError:
        return (
            False,
            "Ollama is not available. Install Ollama, start it, then pull a Qwen3 "
            f"model (default: `{settings.ollama_model}`). Server: {settings.ollama_base_url}",
        )

    names: list[str] = []
    for item in response.json().get("models", []):
        name = item.get("name") or item.get("model") or ""
        if name:
            names.append(name)

    wanted = settings.ollama_model
    matched = any(
        name == wanted or name.startswith(wanted + "-") or name.startswith(wanted + ":")
        for name in names
    )
    if not matched and wanted not in names:
        available = ", ".join(names) if names else "(none)"
        return (
            False,
            f"Qwen3 model `{wanted}` was not found in Ollama. "
            f"Run `ollama pull {wanted}`. Installed models: {available}",
        )
    return True, f"Ollama is ready · {wanted}"


def require_ollama() -> None:
    ok, message = check_ollama()
    if not ok:
        raise OllamaError(message)
