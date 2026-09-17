"""Construct compact context, select once, validate, and add runtime metadata."""

from __future__ import annotations

from typing import Any

from .source_selection import build_selection_context, validate_selection
from .source_selector import SourceSelector


def create_source_selection(plan: dict[str, Any], sources: dict[str, Any],
                            selector: SourceSelector,
                            transcript: dict[str, Any] | None = None) -> dict[str, Any]:
    context = build_selection_context(plan, sources, transcript)
    result = selector.select(context)
    document = result.selection | {
        "provider": result.provider,
        "metadata": {"provider": result.provider, "model": result.model,
                     "invocation_count": result.invocation_count,
                     "elapsed_seconds": result.elapsed_seconds},
    }
    return validate_selection(document, plan, sources)
