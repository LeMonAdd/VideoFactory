"""Replaceable deterministic scene planner."""

from __future__ import annotations

from typing import Any

from . import SCHEMA_VERSION


def plan_scenes(transcript: dict[str, Any], sources: dict[str, Any], mode: str,
                minimum_scene_seconds: float = 3.0) -> dict[str, Any]:
    segments = _split_long_segments(transcript["segments"])
    if not segments:
        raise ValueError("Cannot plan scenes from an empty transcript")
    groups: list[list[dict[str, Any]]] = []
    group: list[dict[str, Any]] = []
    scene_start = 0.0
    for segment in segments:
        group.append(segment)
        if segment["end"] - scene_start >= minimum_scene_seconds:
            groups.append(group)
            scene_start = segment["end"]
            group = []
    if group:
        if groups and transcript["duration"] - scene_start < minimum_scene_seconds:
            groups[-1].extend(group)
        else:
            groups.append(group)
    broll = [item for item in sources["sources"] if item["kind"] == "B_ROLL"]
    images = [item for item in sources["sources"] if item["kind"] == "IMAGE"]
    scenes: list[dict[str, Any]] = []
    start = 0.0
    broll_index = 0
    for index, items in enumerate(groups):
        end = float(items[-1]["end"]) if index < len(groups) - 1 else float(transcript["duration"])
        if end <= start:
            continue
        if mode == "TALKING_HEAD" and index % 2 == 0:
            visual_type, source_id = "A_ROLL", "source_aroll"
        elif broll:
            visual_type, source_id = "B_ROLL", broll[broll_index % len(broll)]["id"]
            broll_index += 1
        elif images:
            visual_type, source_id = "IMAGE", images[index % len(images)]["id"]
        elif mode == "TALKING_HEAD":
            visual_type, source_id = "A_ROLL", "source_aroll"
        else:
            visual_type, source_id = "GRAPHIC", None
        narration = " ".join(item["text"] for item in items if item["text"]).strip()
        scenes.append({
            "id": f"scene_{index + 1:03d}", "start": round(start, 3),
            "end": round(end, 3), "narration": narration,
            "visual_type": visual_type, "visual_query": narration[:120] or None,
            "source_id": source_id,
        })
        start = end
    if not scenes:
        raise ValueError("Scene planner produced no scenes")
    return {"schema_version": SCHEMA_VERSION, "scenes": scenes}


def _split_long_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep periodic cuts possible even if Whisper returns a long segment."""
    expanded: list[dict[str, Any]] = []
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        length = end - start
        if length <= 8:
            expanded.append(segment)
            continue
        pieces = max(2, round(length / 4))
        words = segment["text"].split()
        for index in range(pieces):
            left = round(start + index * length / pieces, 3)
            right = round(start + (index + 1) * length / pieces, 3)
            first = round(index * len(words) / pieces)
            last = round((index + 1) * len(words) / pieces)
            expanded.append({"start": left, "end": right,
                             "text": " ".join(words[first:last])})
    return expanded
