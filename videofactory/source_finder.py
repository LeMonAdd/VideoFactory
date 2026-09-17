"""Turn DirectorPlan visual queries into unselected source candidates."""

from __future__ import annotations

from typing import Any, Callable

from . import SCHEMA_VERSION
from .source_models import validate_sources_document
from .source_provider import SourceProvider, SourceSearchError


def normalized_query(query: str) -> str:
    return " ".join(query.split()).casefold()


def find_sources(plan: dict[str, Any], project_name: str, provider: SourceProvider,
                 limit: int = 5, existing_sources: list[dict] | None = None,
                 report: Callable[[str], None] | None = None) -> dict[str, Any]:
    if not 1 <= limit <= 20:
        raise ValueError("Source limit must be between 1 and 20")
    requests: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    fatal_error: str | None = None
    for shot in plan["shots"]:
        visual_type = shot["visual_type"]
        query = shot["visual_query"]
        media_type = "VIDEO" if visual_type == "B_ROLL" else "IMAGE" if visual_type == "IMAGE" else None
        entry: dict[str, Any] = {
            "shot_id": shot["id"], "visual_type": visual_type, "visual_query": query,
            "media_type": media_type, "status": "SKIPPED", "candidates": [],
            "error": None, "reused_from_shot_id": None,
        }
        if media_type is None or (media_type == "IMAGE" and provider.name == "pexels"):
            requests.append(entry)
            continue
        key = (media_type, normalized_query(query))
        if key in cache:
            previous = cache[key]
            entry.update(status=previous["status"], error=previous["error"],
                         reused_from_shot_id=previous["shot_id"],
                         candidates=[{**item, "query": query} for item in previous["candidates"]])
            if report:
                report(f"reused query for shot {shot['id']}")
        elif fatal_error:
            entry.update(status="ERROR", error=fatal_error)
        else:
            if report:
                report(f"shot {shot['id']}: {query}")
            try:
                candidates = provider.search(query, media_type, "landscape", limit)
                entry.update(status="FOUND" if candidates else "NO_RESULTS",
                             candidates=[candidate.to_dict() for candidate in candidates])
            except SourceSearchError as exc:
                entry.update(status="ERROR", error=str(exc))
                if exc.fatal:
                    fatal_error = str(exc)
            cache[key] = entry
        requests.append(entry)
    document = {"schema_version": SCHEMA_VERSION, "project_name": project_name,
                "provider": provider.name, "sources": existing_sources or [], "requests": requests}
    return validate_sources_document(document, project_name)
