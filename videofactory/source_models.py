"""Normalized source candidates and source-search document validation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
from jsonschema import Draft202012Validator

from .paths import ROOT

SOURCE_SEARCH_SCHEMA = ROOT / "config" / "source_search.schema.json"


def positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def nonnegative_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def orientation_for(width: int | None, height: int | None) -> str | None:
    if width is None or height is None:
        return None
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"


@dataclass(frozen=True)
class CandidateFile:
    url: str
    width: int | None = None
    height: int | None = None
    quality: str | None = None
    file_type: str | None = None


@dataclass(frozen=True)
class SourceCandidate:
    candidate_id: str
    provider: str
    provider_asset_id: str
    media_type: str
    query: str
    page_url: str | None = None
    preview_url: str | None = None
    creator: str | None = None
    creator_url: str | None = None
    license: str | None = None
    license_url: str | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    orientation: str | None = None
    files: list[CandidateFile] = field(default_factory=list)
    local_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_sources_document(document: dict[str, Any], project_name: str) -> dict[str, Any]:
    schema = json.loads(SOURCE_SEARCH_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        location = ".".join(map(str, first.path)) or "root"
        raise ValueError(f"Invalid sources.json at {location}: {first.message}")
    if document["project_name"] != project_name:
        raise ValueError("sources.json project_name does not match")
    seen: set[int] = set()
    for request in document["requests"]:
        shot_id, status, candidates = request["shot_id"], request["status"], request["candidates"]
        if shot_id in seen:
            raise ValueError(f"Duplicate source request for shot {shot_id}")
        seen.add(shot_id)
        if status == "FOUND" and not candidates:
            raise ValueError(f"FOUND source request for shot {shot_id} has no candidates")
        if status != "FOUND" and candidates:
            raise ValueError(f"{status} source request for shot {shot_id} must have no candidates")
        if status == "ERROR" and not request["error"]:
            raise ValueError(f"ERROR source request for shot {shot_id} needs an error message")
        if status != "ERROR" and request["error"] is not None:
            raise ValueError(f"Non-error source request for shot {shot_id} has an error message")
        if request["visual_type"] in {"B_ROLL", "IMAGE"} and not (request["visual_query"] or "").strip():
            raise ValueError(f"Source request for shot {shot_id} needs a visual_query")
        if request["visual_type"] in {"A_ROLL", "GRAPHIC"} and status != "SKIPPED":
            raise ValueError(f"{request['visual_type']} shot {shot_id} must be skipped")
        expected_media_type = "VIDEO" if request["visual_type"] == "B_ROLL" else "IMAGE" if request["visual_type"] == "IMAGE" else None
        if request["media_type"] != expected_media_type:
            raise ValueError(f"Source request for shot {shot_id} has wrong media_type")
        for candidate in candidates:
            if (candidate["provider"] != document["provider"] or
                    candidate["query"] != request["visual_query"] or
                    candidate["media_type"] != expected_media_type):
                raise ValueError(f"Candidate mismatch for shot {shot_id}")
    return document
