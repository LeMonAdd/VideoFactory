"""Publish a static-primary-framing render without touching V3D or V4B outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .edit_renderer import inspect_render_inputs, validate_edited_output
from .ffmpeg_utils import run_command
from .jump_cut_style import build_styled_edit_timeline, validate_jump_cut_style_plan
from .models import read_json, write_json
from .paths import ROOT
from .retimed_edit import build_retimed_edit_timeline
from .source_downloader import sha256_file
from .speech_renderer import build_speech_render_command

MANIFEST_SCHEMA = ROOT / "config" / "punch_in_render_manifest.schema.json"


def validate_punch_in_render_manifest(document: dict[str, Any], project_name: str) -> dict[str, Any]:
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid punch_in_render_manifest.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    if document["project_name"] != project_name:
        raise ValueError("Punch-in render manifest project_name mismatch")
    return document


class PunchInRenderer:
    def render(self, project_dir: Path, output_dir: Path, project: dict[str, Any],
               transcript: dict[str, Any], director_plan: dict[str, Any],
               clip_plan: dict[str, Any], timeline: dict[str, Any],
               download_manifest: dict[str, Any], speech_plan: dict[str, Any],
               retime_map: dict[str, Any], retimed: dict[str, Any],
               style_plan: dict[str, Any], styled: dict[str, Any],
               encoder: str = "libx264", overwrite: bool = False,
               before_render: Callable[[], None] | None = None,
               before_validate: Callable[[], None] | None = None) -> dict[str, Any]:
        final = output_dir / "punch_in_draft.mp4"
        temporary = output_dir / "punch_in_draft.mp4.part.mp4"
        if final.exists() and not overwrite:
            raise FileExistsError(f"Punch-in draft already exists: {final}; use --overwrite-render")
        if temporary.exists():
            raise FileExistsError(f"Temporary punch-in draft already exists: {temporary}")
        if output_dir.is_symlink() or final.is_symlink() or temporary.is_symlink():
            raise ValueError("Punch-in render output path must not be a symlink")
        canonical = build_retimed_edit_timeline(project, transcript, director_plan, clip_plan,
                                                timeline, speech_plan, retime_map, project_dir)
        if retimed != canonical:
            raise ValueError("Saved retimed edit timeline differs from canonical transformation")
        validate_jump_cut_style_plan(style_plan, retimed)
        if styled != build_styled_edit_timeline(retimed, style_plan):
            raise ValueError("Saved styled edit timeline differs from canonical style plan")
        inputs = {"speech_edit_plan.json": speech_plan, "retime_map.json": retime_map,
                  "retimed_edit_timeline.json": retimed, "jump_cut_style_plan.json": style_plan,
                  "styled_edit_timeline.json": styled}
        for filename, document in inputs.items():
            if read_json(project_dir / filename) != document:
                raise ValueError(f"Saved {filename} changed before punch-in rendering")
        input_hashes = {filename: sha256_file(project_dir / filename) for filename in inputs}
        primary_info, visual_sources = inspect_render_inputs(
            project, timeline, clip_plan, director_plan, download_manifest, project_dir)
        scales = [item["scale"] for item in style_plan["segments"]]
        command = build_speech_render_command(retimed, project_dir, temporary, primary_info,
                                              visual_sources, encoder, primary_scales=scales)
        output_dir.mkdir(parents=True, exist_ok=True)
        if before_render:
            before_render()
        published = False
        completed = False
        try:
            run_command(command)
            if before_validate:
                before_validate()
            facts = validate_edited_output(temporary, retimed["duration"])
            if any(sha256_file(project_dir / filename) != digest for filename, digest in input_hashes.items()):
                raise ValueError("Punch-in planning artifacts changed during rendering")
            manifest = validate_punch_in_render_manifest({
                "schema_version": SCHEMA_VERSION, "project_name": project_dir.name,
                "input_retimed_edit_timeline": "retimed_edit_timeline.json",
                "input_jump_cut_style_plan": "jump_cut_style_plan.json",
                "input_styled_edit_timeline": "styled_edit_timeline.json",
                "output_path": str(final), "encoder": encoder,
                "source_duration": retimed["source_duration"],
                "total_removed_duration": speech_plan["total_removed_duration"],
                "style_mode": style_plan["mode"], "normal_scale": style_plan["normal_scale"],
                "punch_in_scale": style_plan["punch_in_scale"],
                "styled_segment_count": len(style_plan["segments"]),
                "punch_in_segment_count": sum(item["scale"] > 1 for item in style_plan["segments"]),
                "original_broll_overlay_count": len(retimed["visual_overlays"]),
                "styled_broll_overlay_count": len(styled["visual_overlays"]),
                **facts, "metadata_stripped": True,
                "file_size_bytes": temporary.stat().st_size, "sha256": sha256_file(temporary),
                "jump_cut_style_plan_sha256": input_hashes["jump_cut_style_plan.json"],
                "styled_edit_timeline_sha256": input_hashes["styled_edit_timeline.json"],
                "retimed_edit_timeline_sha256": input_hashes["retimed_edit_timeline.json"],
            }, project_dir.name)
            if overwrite:
                os.replace(temporary, final)
            else:
                os.link(temporary, final)
                temporary.unlink()
            published = True
            write_json(project_dir / "punch_in_render_manifest.json", manifest)
            completed = True
            return manifest
        finally:
            if temporary.exists():
                temporary.unlink()
            if published and not completed and not overwrite and final.exists():
                final.unlink()
