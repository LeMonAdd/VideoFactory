"""Resolve editorial intent to local media without inventing asset matches."""

from __future__ import annotations

from typing import Any

from . import SCHEMA_VERSION


def _narration(transcript: dict[str, Any], start: float, end: float) -> str:
    words = [
        word["text"]
        for segment in transcript["segments"]
        for word in segment.get("words", [])
        if start <= (float(word["start"]) + float(word["end"])) / 2 < end
    ]
    if words:
        return " ".join(words)
    return " ".join(
        segment["text"] for segment in transcript["segments"]
        if not segment.get("words") and segment["start"] >= start - 0.01
        and segment["end"] <= end + 0.01
    )


def _resolve(visual_type: str, query: str | None, sources: dict[str, Any],
             mode: str) -> tuple[str, str | None, str]:
    if visual_type == "A_ROLL":
        return "A_ROLL", "source_aroll", "resolved"
    if visual_type == "GRAPHIC":
        return "GRAPHIC", None, "resolved_placeholder"
    # Only explicit, user-curated sidecar descriptions count as semantic matches.
    normalized = (query or "").strip().casefold()
    matches = [
        item for item in sources["sources"]
        if item["kind"] == visual_type
        and normalized in {phrase.strip().casefold() for phrase in item.get("visual_queries", [])}
    ]
    if matches:
        return visual_type, matches[0]["id"], "resolved_local"
    if mode == "TALKING_HEAD":
        return "A_ROLL", "source_aroll", "unresolved_local_asset"
    return "GRAPHIC", None, "unresolved_local_asset"


def convert_plan_to_scenes(plan: dict[str, Any], transcript: dict[str, Any],
                           sources: dict[str, Any], mode: str) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    cursor = 0.0

    def append(start: float, end: float, intent: dict[str, Any] | None) -> None:
        if end - start <= 0.001:
            return
        requested = intent["visual_type"] if intent else ("A_ROLL" if mode == "TALKING_HEAD" else "GRAPHIC")
        query = intent["visual_query"] if intent else None
        visual_type, source_id, status = _resolve(requested, query, sources, mode)
        scenes.append({
            "id": f"scene_{len(scenes) + 1:03d}",
            "start": round(start, 3), "end": round(end, 3),
            "narration": _narration(transcript, start, end),
            "visual_type": visual_type, "visual_query": query,
            "source_id": source_id, "requested_visual_type": requested,
            "resolution_status": status if intent else "filled_gap",
            "reason": intent["reason"] if intent else "Uncovered interval; no narration invented",
            "director_shot_id": intent["id"] if intent else None,
        })

    for shot in plan["shots"]:
        start, end = float(shot["start"]), float(shot["end"])
        if start > cursor + 0.001:
            append(cursor, start, None)
        append(start, end, shot)
        cursor = end
    if float(transcript["duration"]) > cursor + 0.001:
        append(cursor, float(transcript["duration"]), None)
    return {"schema_version": SCHEMA_VERSION, "scenes": scenes}
