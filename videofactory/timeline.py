"""Create and check the authoritative, replayable timeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .models import VISUAL_TYPES


def build_timeline(scenes: dict[str, Any], sources: dict[str, Any], narration_source: Path,
                   mode: str, settings: dict[str, Any]) -> dict[str, Any]:
    by_id = {item["id"]: item for item in sources["sources"]}
    events: list[dict[str, Any]] = []
    for scene in scenes["scenes"]:
        kind = scene["visual_type"]
        if kind not in VISUAL_TYPES:
            raise ValueError(f"Unsupported visual type: {kind}")
        source = by_id.get(scene["source_id"]) if scene["source_id"] else None
        if kind != "GRAPHIC" and source is None:
            raise ValueError(f"Missing source for {scene['id']}")
        if source is not None and source["kind"] != kind:
            raise ValueError(f"Wrong source kind for {scene['id']}")
        length = round(float(scene["end"]) - float(scene["start"]), 3)
        if length <= 0:
            raise ValueError(f"Non-positive scene duration: {scene['id']}")
        source_in = float(scene["start"]) if kind == "A_ROLL" else 0.0
        if kind == "B_ROLL" and source["duration"] is not None:
            source_out = min(length, float(source["duration"]))
        else:
            source_out = source_in + length
        events.append({
            "scene_id": scene["id"], "start": scene["start"], "end": scene["end"],
            "visual_type": kind, "source_id": scene["source_id"],
            "source_path": source["path"] if source else None,
            "source_in": round(source_in, 3) if source else None,
            "source_out": round(source_out, 3) if source else None,
            "loop": kind == "B_ROLL",
            "fit": "cover",
            "graphic_color": "0x202630" if kind == "GRAPHIC" else None,
            "audio_behavior": "original_narration_continuous",
        })
    timeline = {
        "schema_version": SCHEMA_VERSION, "mode": mode,
        "duration": events[-1]["end"], "output_settings": settings,
        "audio": {"source_path": str(narration_source.resolve()),
                  "stream_index": 0, "behavior": "continuous"},
        "events": events,
    }
    validate_timeline(timeline)
    return timeline


def validate_timeline(timeline: dict[str, Any]) -> None:
    events = timeline.get("events", [])
    if not events:
        raise ValueError("Timeline has no visual events")
    cursor = 0.0
    for event in events:
        if abs(float(event["start"]) - cursor) > 0.01 or event["end"] <= event["start"]:
            raise ValueError(f"Timeline gap, overlap, or invalid duration at {event['scene_id']}")
        if event["visual_type"] not in VISUAL_TYPES:
            raise ValueError(f"Invalid visual type in timeline: {event['visual_type']}")
        if event["visual_type"] != "GRAPHIC" and not event.get("source_path"):
            raise ValueError(f"Timeline event lacks source path: {event['scene_id']}")
        if event.get("fit") != "cover":
            raise ValueError(f"Unsupported fit mode in timeline: {event['scene_id']}")
        cursor = float(event["end"])
    if abs(cursor - float(timeline["duration"])) > 0.01:
        raise ValueError("Timeline duration does not match visual events")
    if not timeline.get("audio", {}).get("source_path"):
        raise ValueError("Timeline lacks narration audio source")
