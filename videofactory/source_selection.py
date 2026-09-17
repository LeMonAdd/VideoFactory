"""Validate selections against the actual per-shot source candidates."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from jsonschema import Draft202012Validator

from .paths import ROOT
from .source_models import validate_sources_document

SELECTION_SCHEMA = ROOT / "config" / "source_selection.schema.json"


def page_title(page_url: str | None) -> str | None:
    """Expose a URL slug as a clue, never as verified visual content."""
    if not page_url:
        return None
    parsed = urlsplit(page_url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname != "www.pexels.com":
        return None
    slug = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    words = [word for word in slug.split("-") if word and not word.isdigit()]
    return " ".join(words) or None


def narration_for(shot: dict[str, Any], transcript: dict[str, Any] | None) -> str | None:
    if not transcript:
        return None
    parts = [segment.get("text", "").strip() for segment in transcript.get("segments", [])
             if float(segment["start"]) < shot["end"] and float(segment["end"]) > shot["start"]]
    text = " ".join(part for part in parts if part)
    return text[:500] or None


def build_selection_context(plan: dict[str, Any], sources: dict[str, Any],
                            transcript: dict[str, Any] | None = None) -> dict[str, Any]:
    project_name = plan["project_name"]
    validate_sources_document(sources, project_name)
    requests = {request["shot_id"]: request for request in sources["requests"]}
    if set(requests) != {shot["id"] for shot in plan["shots"]}:
        raise ValueError("sources.json requests must cover every director shot")
    shots = []
    for shot in plan["shots"]:
        request = requests[shot["id"]]
        if (request["visual_type"] != shot["visual_type"] or
                request["visual_query"] != shot["visual_query"]):
            raise ValueError(f"sources.json request does not match director shot {shot['id']}")
        candidates = [{
            "candidate_id": candidate["candidate_id"],
            "provider": candidate["provider"],
            "provider_asset_id": candidate["provider_asset_id"],
            "page_url": candidate["page_url"],
            "page_title_clue": page_title(candidate["page_url"]),
            "width": candidate["width"], "height": candidate["height"],
            "duration": candidate["duration"], "orientation": candidate["orientation"],
            "creator": candidate["creator"],
        } for candidate in request["candidates"]]
        shots.append({
            "shot_id": shot["id"], "visual_type": shot["visual_type"],
            "visual_query": shot["visual_query"],
            "shot_duration": round(shot["end"] - shot["start"], 3),
            "narration_context": narration_for(shot, transcript),
            "search_status": request["status"], "candidates": candidates,
        })
    return {"schema_version": 1, "project_name": project_name, "shots": shots}


def validate_selection(document: dict[str, Any], plan: dict[str, Any],
                       sources: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(SELECTION_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.path)) or "root"
        raise ValueError(f"Invalid source_selection.json at {location}: {first.message}")
    if document["project_name"] != plan["project_name"]:
        raise ValueError("Source selection project_name does not match director plan")
    if document["metadata"]["provider"] != document["provider"]:
        raise ValueError("Source selection provider metadata does not match")
    if not math.isfinite(document["metadata"]["elapsed_seconds"]):
        raise ValueError("Source selection elapsed_seconds must be finite")
    context = build_selection_context(plan, sources)
    shots = {shot["shot_id"]: shot for shot in context["shots"]}
    seen: set[int] = set()
    for selection in document["selections"]:
        shot_id = selection["shot_id"]
        if shot_id in seen:
            raise ValueError(f"Duplicate selection for shot {shot_id}")
        seen.add(shot_id)
        if shot_id not in shots:
            raise ValueError(f"Selection references unknown shot {shot_id}")
        shot = shots[shot_id]
        status, candidate_id = selection["status"], selection["candidate_id"]
        confidence = selection["confidence"]
        if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
            raise ValueError(f"Invalid confidence for shot {shot_id}")
        if not selection["reason"].strip():
            raise ValueError(f"Selection for shot {shot_id} needs a reason")
        if selection["refined_query"] is not None and not selection["refined_query"].strip():
            raise ValueError(f"Empty refined_query for shot {shot_id}")
        if status != "NO_SUITABLE_CANDIDATE" and selection["refined_query"] is not None:
            raise ValueError(f"Only rejected shot {shot_id} may suggest a refined_query")
        if shot["visual_type"] in {"A_ROLL", "GRAPHIC"}:
            if status != "SKIPPED" or candidate_id is not None:
                raise ValueError(f"{shot['visual_type']} shot {shot_id} must be SKIPPED")
        elif status == "SKIPPED":
            if shot["visual_type"] != "IMAGE" or shot["search_status"] != "SKIPPED" or candidate_id is not None:
                raise ValueError(f"Visual shot {shot_id} cannot be SKIPPED")
        elif status == "NO_SUITABLE_CANDIDATE":
            if candidate_id is not None:
                raise ValueError(f"Rejected shot {shot_id} must have candidate_id=null")
        elif status == "SELECTED":
            if not candidate_id:
                raise ValueError(f"SELECTED shot {shot_id} requires candidate_id")
            candidates = [item for item in shot["candidates"] if item["candidate_id"] == candidate_id]
            if len(candidates) != 1:
                raise ValueError(f"Candidate {candidate_id} is not available for shot {shot_id}")
            candidate = candidates[0]
            if shot["visual_type"] == "B_ROLL":
                duration = candidate["duration"]
                if duration is None or not math.isfinite(duration) or duration + 0.001 < shot["shot_duration"]:
                    raise ValueError(f"Candidate {candidate_id} is too short or has unknown duration for shot {shot_id}")
        else:
            raise ValueError(f"Invalid selection status for shot {shot_id}")
    if seen != set(shots):
        raise ValueError("Source selection needs exactly one entry per director shot")
    return document
