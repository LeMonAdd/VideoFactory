"""Map a saved layered edit onto the canonical speech-shortened time axis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .clip_planner import TIMING_TOLERANCE
from .edit_timeline import validate_edit_timeline
from .paths import ROOT
from .speech_edit import validate_retime_map, validate_saved_speech_edit_plan

SCHEMA = ROOT / "config" / "retimed_edit_timeline.schema.json"


def _ms(value: float) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Retimed edit timing must be finite")
    return round(value * 1000)


def _sec(value: int) -> float:
    return value / 1000


def retime_overlay(overlay: dict[str, Any], clip: dict[str, Any],
                   keeps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Intersect one original B-roll overlay with each canonical keep range."""
    if overlay["visual_type"] != "B_ROLL" or clip["status"] != "PLANNED":
        raise ValueError("Only planned B-roll overlays can be retimed")
    pieces: list[dict[str, Any]] = []
    original_start = _ms(overlay["timeline_start"])
    original_end = _ms(overlay["timeline_end"])
    original_source_in = _ms(overlay["source_in"])
    for keep in keeps:
        keep_start, keep_end = _ms(keep["source_start"]), _ms(keep["source_end"])
        start, end = max(original_start, keep_start), min(original_end, keep_end)
        if end <= start:
            continue
        source_in = original_source_in + start - original_start
        source_out = source_in + end - start
        edited_start = _ms(keep["edited_start"]) + start - keep_start
        edited_end = edited_start + end - start
        if source_out > _ms(clip["asset_duration"]) + _ms(TIMING_TOLERANCE):
            raise ValueError(f"Retimed B-roll source range exceeds asset duration for shot {overlay['shot_id']}")
        pieces.append({"shot_id": overlay["shot_id"], "segment_index": len(pieces) + 1,
                       "candidate_id": clip["candidate_id"], "visual_type": "B_ROLL",
                       "source_path": overlay["source_path"], "original_timeline_start": _sec(start),
                       "original_timeline_end": _sec(end),
                       "timeline_start": _sec(edited_start), "timeline_end": _sec(edited_end),
                       "source_in": _sec(source_in), "source_out": _sec(source_out),
                       "source_audio": False, "audio_policy": "MUTE_SOURCE"})
    return pieces


def build_retimed_edit_timeline(project: dict[str, Any], transcript: dict[str, Any],
                                director_plan: dict[str, Any], clip_plan: dict[str, Any],
                                timeline: dict[str, Any], speech_plan: dict[str, Any],
                                retime_map: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    validate_edit_timeline(timeline, project, director_plan, clip_plan, project_dir)
    validate_saved_speech_edit_plan(speech_plan, project, transcript, timeline)
    validate_retime_map(retime_map, speech_plan)
    if project["mode"] != "TALKING_HEAD":
        raise ValueError("VOICEOVER speech rendering requires a renderable primary visual")
    keeps = retime_map["segments"]
    clips = {clip["shot_id"]: clip for clip in clip_plan["clips"]}
    overlays: list[dict[str, Any]] = []
    for overlay in timeline["visual_overlays"]:
        if overlay["visual_type"] != "B_ROLL":
            raise ValueError("Speech renderer cannot render an unresolved GRAPHIC overlay")
        overlays.extend(retime_overlay(overlay, clips[overlay["shot_id"]], keeps))
    overlays.sort(key=lambda item: (item["timeline_start"], item["shot_id"], item["segment_index"]))
    result = {"schema_version": SCHEMA_VERSION, "project_name": project["project_name"],
              "mode": "TALKING_HEAD", "source_duration": speech_plan["source_duration"],
              "duration": speech_plan["edited_duration"], "retime_source": "retime_map.json",
              "base_video": {"role": "PRIMARY", "source_path": timeline["base_video"]["source_path"],
                             "keep_segments": keeps},
              "base_audio": {"role": "PRIMARY", "source_path": timeline["base_audio"]["source_path"],
                             "keep_segments": keeps},
              "visual_overlays": overlays}
    return validate_retimed_edit_timeline(result, project, clip_plan, retime_map)


def validate_retimed_edit_timeline(document: dict[str, Any], project: dict[str, Any],
                                   clip_plan: dict[str, Any], retime_map: dict[str, Any]) -> dict[str, Any]:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid retimed_edit_timeline.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    if (document["project_name"] != project["project_name"] or document["mode"] != project["mode"] or
            _ms(document["source_duration"]) != _ms(retime_map["source_duration"]) or
            _ms(document["duration"]) != _ms(retime_map["edited_duration"])):
        raise ValueError("Retimed edit project or duration mismatch")
    for layer in ("base_video", "base_audio"):
        if (document[layer]["source_path"] != project["source_media"] or
                document[layer]["keep_segments"] != retime_map["segments"]):
            raise ValueError(f"Retimed {layer} differs from canonical keep segments")
    clips = {clip["shot_id"]: clip for clip in clip_plan["clips"]}
    prior_end = 0
    indexes: dict[int, int] = {}
    for overlay in document["visual_overlays"]:
        start, end = _ms(overlay["timeline_start"]), _ms(overlay["timeline_end"])
        source_in, source_out = _ms(overlay["source_in"]), _ms(overlay["source_out"])
        original_start, original_end = _ms(overlay["original_timeline_start"]), _ms(overlay["original_timeline_end"])
        shot = overlay["shot_id"]
        clip = clips.get(shot)
        indexes[shot] = indexes.get(shot, 0) + 1
        if (clip is None or clip["status"] != "PLANNED" or
                overlay["candidate_id"] != clip["candidate_id"] or
                overlay["source_path"] != clip["relative_path"] or
                overlay["segment_index"] != indexes[shot] or
                start < prior_end or end <= start or end > _ms(document["duration"]) or
                original_end - original_start != end - start or
                source_out - source_in != end - start or
                source_in < _ms(clip["source_in"]) or source_out > _ms(clip["source_out"]) or
                overlay["source_audio"] is not False or overlay["audio_policy"] != "MUTE_SOURCE"):
            raise ValueError(f"Invalid retimed B-roll overlay for shot {shot}")
        prior_end = end
    return document
