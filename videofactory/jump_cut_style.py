"""Deterministic static framing for canonical primary keep segments."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .paths import ROOT

STYLE_SCHEMA = ROOT / "config" / "jump_cut_style_plan.schema.json"
STYLED_SCHEMA = ROOT / "config" / "styled_edit_timeline.schema.json"


def validate_punch_in_scale(scale: float) -> float:
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or
            not math.isfinite(scale) or not 1 < scale <= 1.20):
        raise ValueError("Punch-in scale must be finite, greater than 1.0, and at most 1.20")
    return float(scale)


def _schema_validate(document: dict[str, Any], path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid {path.name} at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")


def _style_segments(retimed: dict[str, Any], punch_in_scale: float) -> list[dict[str, Any]]:
    return [{"segment_index": index, "edited_start": keep["edited_start"],
             "edited_end": keep["edited_end"],
             "scale": 1.0 if index % 2 else punch_in_scale}
            for index, keep in enumerate(retimed["base_video"]["keep_segments"], 1)]


def build_jump_cut_style_plan(retimed: dict[str, Any], punch_in_scale: float = 1.08) -> dict[str, Any]:
    scale = validate_punch_in_scale(punch_in_scale)
    result = {"schema_version": SCHEMA_VERSION, "project_name": retimed["project_name"],
              "source": "retimed_edit_timeline.json", "mode": "alternate_static_punch_in",
              "normal_scale": 1.0, "punch_in_scale": scale, "anchor": "center",
              "segments": _style_segments(retimed, scale)}
    return validate_jump_cut_style_plan(result, retimed)


def validate_jump_cut_style_plan(document: dict[str, Any], retimed: dict[str, Any]) -> dict[str, Any]:
    _schema_validate(document, STYLE_SCHEMA)
    scale = validate_punch_in_scale(document["punch_in_scale"])
    if (document["project_name"] != retimed["project_name"] or retimed["mode"] != "TALKING_HEAD" or
            document["segments"] != _style_segments(retimed, scale)):
        raise ValueError("Jump-cut style segments differ from canonical keep segments")
    return document


def build_styled_edit_timeline(retimed: dict[str, Any],
                               style_plan: dict[str, Any]) -> dict[str, Any]:
    validate_jump_cut_style_plan(style_plan, retimed)
    video_segments = [dict(keep, scale=style["scale"], anchor="center")
                      for keep, style in zip(retimed["base_video"]["keep_segments"], style_plan["segments"])]
    result = {"schema_version": SCHEMA_VERSION, "project_name": retimed["project_name"],
              "mode": retimed["mode"], "source_duration": retimed["source_duration"],
              "duration": retimed["duration"], "retime_source": "retime_map.json",
              "style_source": "jump_cut_style_plan.json",
              "base_video": {"role": "PRIMARY", "source_path": retimed["base_video"]["source_path"],
                             "segments": video_segments},
              "base_audio": copy.deepcopy(retimed["base_audio"]),
              "visual_overlays": copy.deepcopy(retimed["visual_overlays"])}
    return validate_styled_edit_timeline(result, retimed, style_plan)


def validate_styled_edit_timeline(document: dict[str, Any], retimed: dict[str, Any],
                                  style_plan: dict[str, Any]) -> dict[str, Any]:
    _schema_validate(document, STYLED_SCHEMA)
    validate_jump_cut_style_plan(style_plan, retimed)
    expected_video = [dict(keep, scale=style["scale"], anchor="center")
                      for keep, style in zip(retimed["base_video"]["keep_segments"], style_plan["segments"])]
    if (document["project_name"] != retimed["project_name"] or
            document["mode"] != retimed["mode"] or
            document["source_duration"] != retimed["source_duration"] or
            document["duration"] != retimed["duration"] or
            document["base_video"]["source_path"] != retimed["base_video"]["source_path"] or
            document["base_video"]["segments"] != expected_video or
            document["base_audio"] != retimed["base_audio"] or
            document["visual_overlays"] != retimed["visual_overlays"]):
        raise ValueError("Styled edit timeline changed canonical timing, audio, or B-roll")
    return document
