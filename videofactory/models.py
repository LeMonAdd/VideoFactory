"""Stable project document helpers and schema checks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION

VISUAL_TYPES = {"A_ROLL", "B_ROLL", "IMAGE", "GRAPHIC"}


def write_json(path: Path, data: dict[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path.name}: missing or unsupported schema_version")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version")
    return data


def normalize_transcript(raw: dict[str, Any], media_duration: float) -> dict[str, Any]:
    if not math.isfinite(media_duration) or media_duration <= 0:
        raise ValueError("Input media duration must be positive")
    segments: list[dict[str, Any]] = []
    for item in raw.get("segments", []):
        start, end = float(item["start"]), float(item["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= media_duration + 0.25):
            raise ValueError(f"Invalid transcript segment: {item}")
        segment: dict[str, Any] = {
            "start": round(start, 3),
            "end": round(min(end, media_duration), 3),
            "text": str(item.get("text", "")).strip(),
        }
        if item.get("words"):
            segment["words"] = [
                {"start": round(float(word["start"]), 3), "end": round(float(word["end"]), 3),
                 "text": str(word.get("word", word.get("text", ""))).strip()}
                for word in item["words"]
            ]
        segments.append(segment)
    segments.sort(key=lambda value: value["start"])
    if not segments:
        raise ValueError("Transcript has no timed segments")
    for previous, current in zip(segments, segments[1:]):
        if current["start"] < previous["end"] - 0.05:
            raise ValueError("Transcript segments overlap")
    return {
        "schema_version": SCHEMA_VERSION,
        "language": raw.get("language") or "und",
        "duration": round(media_duration, 3),
        "segments": segments,
    }
