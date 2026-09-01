"""Convert exceptions into short messages suitable for the Streamlit UI."""

from __future__ import annotations


def user_message(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) > 500:
        text = text[:500] + "..."
    return text
