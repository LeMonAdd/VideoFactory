"""Layered future-renderer input with continuous primary narration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .clip_planner import TIMING_TOLERANCE, resolve_project_media, validate_clip_plan
from .paths import ROOT

EDIT_SCHEMA = ROOT / "config" / "edit_timeline.schema.json"


def build_edit_timeline(project: dict[str, Any], duration: float, plan: dict[str, Any],
                        clip_plan: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    mode = project["mode"]
    validate_clip_plan(clip_plan, plan, mode, duration, project_dir)
    primary = Path(project["source_media"])
    if not primary.is_file():
        raise FileNotFoundError(f"Primary media is missing: {primary}")
    primary_path = str(primary.resolve())
    base = {"source": "PRIMARY", "source_path": primary_path,
            "timeline_start": 0, "timeline_end": duration,
            "source_in": 0, "source_out": duration}
    overlays = []
    resolutions = []
    for clip in clip_plan["clips"]:
        start, end = clip["timeline_start"], clip["timeline_end"]
        resolutions.append({"shot_id": clip["shot_id"], "timeline_start": start,
                            "timeline_end": end, "visual_type": clip["resolved_visual_type"],
                            "status": clip["status"]})
        if clip["status"] == "PLANNED":
            overlays.append({"shot_id": clip["shot_id"], "visual_type": "B_ROLL",
                             "timeline_start": start, "timeline_end": end,
                             "source_path": clip["relative_path"], "source_in": clip["source_in"],
                             "source_out": clip["source_out"], "source_audio": False,
                             "audio_policy": "MUTE_SOURCE"})
        elif clip["resolved_visual_type"] == "GRAPHIC":
            overlays.append({"shot_id": clip["shot_id"], "visual_type": "GRAPHIC",
                             "timeline_start": start, "timeline_end": end,
                             "source_path": None, "source_in": None, "source_out": None,
                             "source_audio": False, "audio_policy": "MUTE_SOURCE"})
    document = {"schema_version": SCHEMA_VERSION, "project_name": plan["project_name"],
                "mode": mode, "duration": duration,
                "base_video": base if mode == "TALKING_HEAD" else None,
                "base_audio": base | {"audio_policy": "KEEP_PRIMARY_CONTINUOUS"},
                "uncovered_visual": "PRIMARY" if mode == "TALKING_HEAD" else "GRAPHIC_PLACEHOLDER",
                "shot_resolutions": resolutions, "visual_overlays": overlays}
    return validate_edit_timeline(document, project, plan, clip_plan, project_dir)


def validate_edit_timeline(document: dict[str, Any], project: dict[str, Any],
                           plan: dict[str, Any], clip_plan: dict[str, Any],
                           project_dir: Path) -> dict[str, Any]:
    schema = json.loads(EDIT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid edit_timeline.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    mode = project["mode"]
    duration = document["duration"]
    if (document["project_name"] != plan["project_name"] or project["project_name"] != plan["project_name"]
            or document["mode"] != mode or not math.isfinite(duration)):
        raise ValueError("Edit timeline project, mode, or duration mismatch")
    validate_clip_plan(clip_plan, plan, mode, duration, project_dir)
    primary = Path(project["source_media"])
    if not primary.is_file():
        raise FileNotFoundError(f"Primary media is missing: {primary}")
    primary_path = str(primary.resolve())
    for name in ("base_audio", "base_video"):
        layer = document[name]
        if name == "base_video" and mode == "VOICEOVER":
            if layer is not None:
                raise ValueError("VOICEOVER base_video must be null")
            continue
        if layer is None or layer["source_path"] != primary_path:
            raise ValueError(f"{name} must use the primary source")
        if any(abs(layer[key] - expected) > TIMING_TOLERANCE for key, expected in
               (("timeline_start", 0), ("source_in", 0),
                ("timeline_end", duration), ("source_out", duration))):
            raise ValueError(f"{name} must cover the entire project")
    expected_uncovered = "PRIMARY" if mode == "TALKING_HEAD" else "GRAPHIC_PLACEHOLDER"
    if document["uncovered_visual"] != expected_uncovered:
        raise ValueError("Wrong uncovered visual policy")
    if len(document["shot_resolutions"]) != len(clip_plan["clips"]):
        raise ValueError("Edit timeline must resolve every director shot")
    for resolution, clip in zip(document["shot_resolutions"], clip_plan["clips"]):
        if (resolution["shot_id"] != clip["shot_id"] or resolution["status"] != clip["status"] or
                resolution["visual_type"] != clip["resolved_visual_type"] or
                abs(resolution["timeline_start"] - clip["timeline_start"]) > TIMING_TOLERANCE or
                abs(resolution["timeline_end"] - clip["timeline_end"]) > TIMING_TOLERANCE):
            raise ValueError("Edit timeline shot resolution mismatch")
    expected_overlays = [clip for clip in clip_plan["clips"]
                         if clip["status"] == "PLANNED" or clip["resolved_visual_type"] == "GRAPHIC"]
    if len(document["visual_overlays"]) != len(expected_overlays):
        raise ValueError("Edit timeline overlays do not match the clip plan")
    cursor = 0.0
    for overlay, clip in zip(document["visual_overlays"], expected_overlays):
        start, end = overlay["timeline_start"], overlay["timeline_end"]
        if (overlay["shot_id"] != clip["shot_id"] or overlay["visual_type"] != clip["resolved_visual_type"]
                or not all(math.isfinite(value) for value in (start, end))
                or start < cursor - TIMING_TOLERANCE or start < 0 or end <= start
                or end > duration + TIMING_TOLERANCE
                or abs(start - clip["timeline_start"]) > TIMING_TOLERANCE
                or abs(end - clip["timeline_end"]) > TIMING_TOLERANCE):
            raise ValueError(f"Invalid visual overlay for shot {clip['shot_id']}")
        cursor = end
        if overlay["visual_type"] == "B_ROLL":
            if (overlay["source_path"] != clip["relative_path"] or
                    overlay["source_in"] != clip["source_in"] or
                    overlay["source_out"] != clip["source_out"] or
                    abs((overlay["source_out"] - overlay["source_in"]) - (end - start)) > TIMING_TOLERANCE):
                raise ValueError(f"B-roll overlay range mismatch for shot {clip['shot_id']}")
            resolve_project_media(project_dir, overlay["source_path"])
        elif any(overlay[key] is not None for key in ("source_path", "source_in", "source_out")):
            raise ValueError(f"Graphic overlay must not reference media for shot {clip['shot_id']}")
        if overlay["source_audio"] is not False or overlay["audio_policy"] != "MUTE_SOURCE":
            raise ValueError(f"Overlay audio must be muted for shot {clip['shot_id']}")
    return document
