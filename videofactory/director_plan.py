"""Validate untrusted editorial intent and align cuts to spoken words."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .paths import ROOT

SCHEMA_PATH = ROOT / "config" / "director_plan.schema.json"
MIN_SHOT_SECONDS = 1.0
SNAP_TOLERANCE_SECONDS = 0.3


def load_schema(path: Path = SCHEMA_PATH) -> dict[str, Any]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


def validate_plan(plan: Any, project_name: str, duration: float, mode: str,
                  schema_path: Path = SCHEMA_PATH) -> dict[str, Any]:
    schema = load_schema(schema_path)
    errors = sorted(Draft202012Validator(schema).iter_errors(plan), key=lambda error: list(map(str, error.path)))
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.path)) or "root"
        raise ValueError(f"Invalid director plan at {location}: {first.message}")
    if plan["project_name"] != project_name:
        raise ValueError("Director plan project_name does not match the requested project")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Transcript duration must be positive")
    previous_end = -1.0
    seen_ids: set[int] = set()
    for shot in plan["shots"]:
        start, end = shot["start"], shot["end"]
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"Shot {shot['id']} has non-finite timing")
        if end - start < MIN_SHOT_SECONDS:
            raise ValueError(f"Shot {shot['id']} is shorter than {MIN_SHOT_SECONDS:.1f}s")
        if end > duration + 0.000001:
            raise ValueError(f"Shot {shot['id']} exceeds transcript duration")
        if start < previous_end - 0.001:
            raise ValueError(f"Shot {shot['id']} overlaps or is out of order")
        if shot["id"] in seen_ids:
            raise ValueError(f"Duplicate shot id: {shot['id']}")
        seen_ids.add(shot["id"])
        previous_end = end
        visual_type, query = shot["visual_type"], shot["visual_query"]
        if visual_type in {"B_ROLL", "IMAGE"} and (not isinstance(query, str) or not query.strip()):
            raise ValueError(f"Shot {shot['id']} requires a visual_query")
        if visual_type == "A_ROLL" and query is not None:
            raise ValueError(f"A_ROLL shot {shot['id']} must have visual_query=null")
        if mode == "VOICEOVER" and visual_type == "A_ROLL":
            raise ValueError("VOICEOVER director plans cannot use A_ROLL")
    return plan


def transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    return [word for segment in transcript["segments"] for word in segment.get("words", [])]


def snap_boundary(time: float, words: list[dict[str, Any]],
                  tolerance: float = SNAP_TOLERANCE_SECONDS) -> float:
    """Snap nearby cut times; reject cuts deep inside a timed word."""
    if not words:
        return time
    boundaries = [float(word[edge]) for word in words for edge in ("start", "end")]
    nearest = min(boundaries, key=lambda boundary: abs(boundary - time))
    if abs(nearest - time) <= tolerance:
        return round(nearest, 3)
    for word in words:
        if float(word["start"]) < time < float(word["end"]):
            raise ValueError(f"Cut at {time:.3f}s falls inside a spoken word and is too far to snap")
    return time


def align_plan_to_words(plan: dict[str, Any], transcript: dict[str, Any]) -> dict[str, Any]:
    aligned = copy.deepcopy(plan)
    words = transcript_words(transcript)
    if not words:
        return aligned
    for shot in aligned["shots"]:
        shot["start"] = snap_boundary(float(shot["start"]), words)
        shot["end"] = snap_boundary(float(shot["end"]), words)
    return aligned
