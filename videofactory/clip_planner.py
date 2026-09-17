"""Deterministic, technically valid source windows for selected B-roll."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .paths import ROOT
from .source_downloader import sha256_file, validate_manifest
from .source_selection import validate_selection

CLIP_SCHEMA = ROOT / "config" / "clip_plan.schema.json"
TIMING_TOLERANCE = 0.003


def resolve_project_media(project_dir: Path, relative_path: str) -> Path:
    """Allow existing regular media files only beneath the real project directory."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("media", "broll"):
        raise ValueError(f"Unsafe project media path: {relative_path}")
    if project_dir.is_symlink():
        raise ValueError("Project directory must not be a symlink")
    cursor = project_dir
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Project media path crosses a symlink: {relative_path}")
    resolved_root = project_dir.resolve()
    resolved = cursor.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise FileNotFoundError(f"Downloaded media is missing or outside project: {relative_path}")
    return resolved


def _fallback(shot: dict[str, Any], mode: str, reason: str,
              candidate_id: str | None = None) -> dict[str, Any]:
    if mode == "TALKING_HEAD":
        status, visual = "A_ROLL_FALLBACK", "A_ROLL"
    else:
        status, visual = "GRAPHIC_FALLBACK", "GRAPHIC"
    return {"status": status, "resolved_visual_type": visual, "candidate_id": candidate_id,
            "relative_path": None, "source_in": None, "source_out": None,
            "source_duration": None, "asset_duration": None,
            "audio_policy": "KEEP_PRIMARY_CONTINUOUS", "source_overlap": False,
            "reason": reason}


def _allocate(windows: list[dict[str, Any]], asset_duration: float,
              preferred_margin: float) -> None:
    """Spread non-overlapping uses across footage; overlap only when required."""
    ordered = sorted(windows, key=lambda item: (item["timeline_start"], item["shot_id"]))
    required = [item["timeline_duration"] for item in ordered]
    total = sum(required)
    if total <= asset_duration:
        margin = min(preferred_margin, max(0.0, (asset_duration - total) / 2))
        gap = max(0.0, asset_duration - total - 2 * margin) / (len(ordered) + 1)
        starts = []
        cursor = margin + gap
        for length in required:
            starts.append(cursor)
            cursor += length + gap
    else:
        starts = [(asset_duration - length) * (index / (len(ordered) - 1))
                  if len(ordered) > 1 else (asset_duration - length) / 2
                  for index, length in enumerate(required)]
    allocated: list[tuple[float, float]] = []
    for item, raw_start in zip(ordered, starts):
        length = item["timeline_duration"]
        start = math.floor((min(max(raw_start, 0.0), asset_duration - length) + 1e-9) * 1000) / 1000
        end = round(start + length, 3)
        overlap = any(start < old_end - TIMING_TOLERANCE and end > old_start + TIMING_TOLERANCE
                      for old_start, old_end in allocated)
        item.update(source_in=start, source_out=end, source_duration=length,
                    source_overlap=overlap,
                    reason="Allocated a valid source range; repeated use overlaps because distinct ranges do not fit"
                    if overlap else "Allocated a distinct valid source range for repeated use"
                    if len(ordered) > 1 else "Allocated a valid source range")
        allocated.append((start, end))


def build_clip_plan(project_dir: Path, mode: str, duration: float, plan: dict[str, Any],
                    sources: dict[str, Any], selection: dict[str, Any],
                    manifest: dict[str, Any], source_margin_seconds: float = 0.5) -> dict[str, Any]:
    if mode not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Invalid project mode")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Project duration must be positive")
    if not math.isfinite(source_margin_seconds) or source_margin_seconds < 0:
        raise ValueError("Source safety margin must be finite and non-negative")
    validate_selection(selection, plan, sources)
    validate_manifest(manifest, plan["project_name"])
    decisions = {item["shot_id"]: item for item in selection["selections"]}
    assets = {item["candidate_id"]: item for item in manifest["assets"]}
    clips: list[dict[str, Any]] = []
    repeated: dict[str, list[dict[str, Any]]] = {}
    for shot in plan["shots"]:
        start, end = float(shot["start"]), float(shot["end"])
        if start < 0 or end <= start or end > duration + TIMING_TOLERANCE:
            raise ValueError(f"Director shot {shot['id']} is outside project duration")
        visual_type = shot["visual_type"]
        decision = decisions[shot["id"]]
        clip = {"shot_id": shot["id"], "visual_type": visual_type,
                "timeline_start": start, "timeline_end": end,
                "timeline_duration": round(end - start, 3)}
        if visual_type == "A_ROLL":
            clip.update(status="SKIPPED", resolved_visual_type="A_ROLL", candidate_id=None,
                        relative_path=None, source_in=None, source_out=None,
                        source_duration=None, asset_duration=None,
                        audio_policy="KEEP_PRIMARY_CONTINUOUS", source_overlap=False,
                        reason="Presenter remains visible on the continuous primary layer")
        elif visual_type == "GRAPHIC":
            clip.update(status="SKIPPED", resolved_visual_type="GRAPHIC", candidate_id=None,
                        relative_path=None, source_in=None, source_out=None,
                        source_duration=None, asset_duration=None,
                        audio_policy="KEEP_PRIMARY_CONTINUOUS", source_overlap=False,
                        reason="Graphic placeholder remains an explicit visual overlay")
        elif decision["status"] != "SELECTED":
            clip.update(_fallback(shot, mode, "No suitable downloaded visual was selected"))
        else:
            candidate_id = decision["candidate_id"]
            asset = assets.get(candidate_id)
            if asset is None:
                clip.update(_fallback(shot, mode, "Selected asset has no download manifest entry", candidate_id))
            else:
                if shot["id"] not in asset["shot_ids"]:
                    raise ValueError(f"Download manifest shot mapping disagrees for shot {shot['id']}")
                path = resolve_project_media(project_dir, asset["relative_path"])
                if sha256_file(path) != asset["sha256"]:
                    raise ValueError(f"Downloaded media hash differs from manifest: {candidate_id}")
                asset_duration = float(asset["duration"])
                if asset_duration < clip["timeline_duration"]:
                    clip.update(_fallback(shot, mode, "Downloaded asset is shorter than the visual shot", candidate_id))
                else:
                    clip.update(status="PLANNED", resolved_visual_type="B_ROLL",
                                candidate_id=candidate_id, relative_path=asset["relative_path"],
                                source_in=None, source_out=None, source_duration=None,
                                asset_duration=asset_duration, audio_policy="MUTE_SOURCE",
                                source_overlap=False, reason="Source window pending allocation")
                    repeated.setdefault(candidate_id, []).append(clip)
        clips.append(clip)
    for candidate_id, group in repeated.items():
        _allocate(group, group[0]["asset_duration"], source_margin_seconds)
    result = {"schema_version": SCHEMA_VERSION, "project_name": plan["project_name"],
              "source_margin_seconds": source_margin_seconds, "clips": clips}
    return validate_clip_plan(result, plan, mode, duration, project_dir)


def validate_clip_plan(document: dict[str, Any], plan: dict[str, Any], mode: str,
                       duration: float, project_dir: Path) -> dict[str, Any]:
    schema = json.loads(CLIP_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid clip_plan.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    if document["project_name"] != plan["project_name"] or project_dir.name != plan["project_name"]:
        raise ValueError("Clip plan project_name mismatch")
    if not math.isfinite(document["source_margin_seconds"]):
        raise ValueError("Clip plan source margin must be finite")
    if len(document["clips"]) != len(plan["shots"]):
        raise ValueError("Clip plan must resolve every director shot")
    cursor = 0.0
    for shot, clip in zip(plan["shots"], document["clips"]):
        if clip["shot_id"] != shot["id"] or clip["visual_type"] != shot["visual_type"]:
            raise ValueError("Clip plan shot order or identity mismatch")
        start, end = clip["timeline_start"], clip["timeline_end"]
        if not all(math.isfinite(value) for value in (start, end, clip["timeline_duration"])):
            raise ValueError(f"Non-finite clip timing for shot {shot['id']}")
        if (start < cursor - TIMING_TOLERANCE or start < 0 or end <= start or
                end > duration + TIMING_TOLERANCE or abs(start - shot["start"]) > TIMING_TOLERANCE or
                abs(end - shot["end"]) > TIMING_TOLERANCE or
                abs((end - start) - clip["timeline_duration"]) > TIMING_TOLERANCE):
            raise ValueError(f"Invalid timeline range for shot {shot['id']}")
        cursor = end
        if clip["status"] == "PLANNED":
            if shot["visual_type"] != "B_ROLL":
                raise ValueError(f"Only B-roll can have a planned source clip: shot {shot['id']}")
            if clip["resolved_visual_type"] != "B_ROLL" or clip["audio_policy"] != "MUTE_SOURCE":
                raise ValueError(f"Invalid B-roll policy for shot {shot['id']}")
            if not clip["candidate_id"] or not clip["relative_path"]:
                raise ValueError(f"Planned B-roll shot {shot['id']} lacks a source")
            source_values = (clip["source_in"], clip["source_out"], clip["source_duration"], clip["asset_duration"])
            if any(value is None or not math.isfinite(value) for value in source_values):
                raise ValueError(f"Planned B-roll shot {shot['id']} lacks valid timing")
            source_in, source_out, source_duration, asset_duration = source_values
            if (source_in < 0 or source_out <= source_in or source_out > asset_duration + 1e-9 or
                    abs((source_out - source_in) - clip["timeline_duration"]) > TIMING_TOLERANCE or
                    abs(source_duration - clip["timeline_duration"]) > TIMING_TOLERANCE):
                raise ValueError(f"Invalid source range for shot {shot['id']}")
            resolve_project_media(project_dir, clip["relative_path"])
        else:
            if clip["status"] == "SKIPPED" and shot["visual_type"] not in {"A_ROLL", "GRAPHIC"}:
                raise ValueError(f"Visual shot {shot['id']} cannot be SKIPPED")
            if clip["status"] in {"A_ROLL_FALLBACK", "GRAPHIC_FALLBACK"} and shot["visual_type"] not in {"B_ROLL", "IMAGE"}:
                raise ValueError(f"Non-visual shot {shot['id']} cannot use fallback status")
            if any(clip[key] is not None for key in ("relative_path", "source_in", "source_out", "source_duration", "asset_duration")):
                raise ValueError(f"Non-B-roll shot {shot['id']} has source timing")
            if clip["audio_policy"] != "KEEP_PRIMARY_CONTINUOUS" or clip["source_overlap"]:
                raise ValueError(f"Invalid fallback policy for shot {shot['id']}")
            expected = "A_ROLL" if mode == "TALKING_HEAD" else "GRAPHIC"
            if clip["status"] == "SKIPPED":
                expected = "A_ROLL" if shot["visual_type"] == "A_ROLL" else "GRAPHIC"
            if clip["resolved_visual_type"] != expected:
                raise ValueError(f"Invalid fallback visual for shot {shot['id']}")
    return document
