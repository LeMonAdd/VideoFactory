"""Compose a validated V3C timeline while keeping primary narration continuous."""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .clip_planner import TIMING_TOLERANCE, resolve_project_media
from .edit_timeline import validate_edit_timeline
from .ffmpeg_utils import FFMPEG, run_command
from .media_probe import duration as probe_duration
from .media_probe import probe
from .models import write_json
from .paths import ROOT
from .source_downloader import sha256_file, validate_manifest

WIDTH, HEIGHT, FPS = 1920, 1080, 30
RENDER_SCHEMA = ROOT / "config" / "render_manifest.schema.json"


def _streams(info: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [item for item in info.get("streams", []) if item.get("codec_type") == kind]


def _number(value: float) -> str:
    """Only validated numeric values enter the filter graph."""
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError("Edit timing must be finite numeric values")
    return f"{value:.3f}"


def inspect_render_inputs(project: dict[str, Any], timeline: dict[str, Any],
                          clip_plan: dict[str, Any], plan: dict[str, Any],
                          manifest: dict[str, Any], project_dir: Path) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    validate_edit_timeline(timeline, project, plan, clip_plan, project_dir)
    validate_manifest(manifest, project["project_name"])
    planned = {clip["shot_id"]: clip for clip in clip_plan["clips"] if clip["status"] == "PLANNED"}
    assets = {asset["candidate_id"]: asset for asset in manifest["assets"]}
    for overlay in timeline["visual_overlays"]:
        if overlay["visual_type"] != "B_ROLL":
            continue
        shot_id = overlay["shot_id"]
        clip = planned.get(shot_id)
        if clip is None:
            raise ValueError(f"No planned B-roll clip for shot {shot_id}")
        candidate_id = clip["candidate_id"]
        asset = assets.get(candidate_id)
        if asset is None:
            raise ValueError(f"Download manifest lacks candidate {candidate_id} for shot {shot_id}")
        if asset["relative_path"] != clip["relative_path"] or shot_id not in asset["shot_ids"]:
            raise ValueError(f"Download manifest mapping differs for shot {shot_id}")
        source = resolve_project_media(project_dir, clip["relative_path"])
        if sha256_file(source) != asset["sha256"]:
            raise ValueError(f"Downloaded media hash differs from manifest: {candidate_id}")
    if project["mode"] != "TALKING_HEAD":
        raise ValueError("VOICEOVER edit rendering requires a renderable base visual and is not yet supported by V3D")
    if any(item["visual_type"] == "GRAPHIC" for item in timeline["visual_overlays"]):
        raise ValueError("V3D cannot render an unresolved GRAPHIC overlay")
    primary = Path(timeline["base_video"]["source_path"])
    info = probe(primary)
    if not _streams(info, "video"):
        raise ValueError("Primary source has no usable video stream")
    audio = _streams(info, "audio")
    if not audio:
        raise ValueError("Primary source has no usable narration audio stream")
    if not isinstance(audio[0].get("index"), int):
        raise ValueError("ffprobe did not identify the primary audio stream index")
    if probe_duration(info) + TIMING_TOLERANCE < timeline["duration"]:
        raise ValueError("Primary source is shorter than edit timeline duration")
    visual_sources: dict[Path, dict[str, Any]] = {}
    for overlay in timeline["visual_overlays"]:
        source = resolve_project_media(project_dir, overlay["source_path"])
        if source not in visual_sources:
            asset_info = probe(source)
            if not _streams(asset_info, "video"):
                raise ValueError(f"B-roll source has no video stream: {source.name}")
            visual_sources[source] = asset_info
        if probe_duration(visual_sources[source]) + TIMING_TOLERANCE < overlay["source_out"]:
            raise ValueError(f"B-roll source is too short for shot {overlay['shot_id']}")
    return info, visual_sources


def build_edit_render_command(project: dict[str, Any], timeline: dict[str, Any],
                              project_dir: Path, destination: Path,
                              primary_info: dict[str, Any],
                              visual_sources: dict[Path, dict[str, Any]],
                              encoder: str = "libx264") -> list[str]:
    if encoder not in {"libx264", "h264_videotoolbox"}:
        raise ValueError(f"Unsupported H.264 encoder: {encoder}")
    if project["mode"] != "TALKING_HEAD" or timeline["mode"] != "TALKING_HEAD":
        raise ValueError("VOICEOVER edit rendering requires a renderable base visual and is not yet supported by V3D")
    duration = _number(timeline["duration"])
    audio = _streams(primary_info, "audio")
    if not audio or not isinstance(audio[0].get("index"), int):
        raise ValueError("Primary source has no usable narration audio stream")
    overlays = timeline["visual_overlays"]
    if any(item["visual_type"] != "B_ROLL" for item in overlays):
        raise ValueError("V3D cannot render an unresolved GRAPHIC overlay")
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-i", timeline["base_video"]["source_path"]]
    sources = list(visual_sources)
    input_index = {source: index + 1 for index, source in enumerate(sources)}
    for source in sources:
        command.extend(["-i", str(source)])
    # FFmpeg keeps autorotation enabled for iPhone MOVs. Normalize the displayed frame.
    filters = [
        f"[0:v:0]setpts=PTS-STARTPTS,fps={FPS},"
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setsar=1,format=yuv420p,"
        f"trim=duration={duration},setpts=PTS-STARTPTS[base]"
    ]
    uses: dict[Path, list[int]] = {source: [] for source in sources}
    for number, overlay in enumerate(overlays):
        source = resolve_project_media(project_dir, overlay["source_path"])
        if source not in input_index:
            raise ValueError(f"B-roll source was not inspected for shot {overlay['shot_id']}")
        uses[source].append(number)
    branch: dict[int, str] = {}
    for source, numbers in uses.items():
        index = input_index[source]
        if len(numbers) == 1:
            branch[numbers[0]] = f"[{index}:v:0]"
        else:
            labels = [f"[src{index}_{number}]" for number in numbers]
            filters.append(f"[{index}:v:0]split={len(numbers)}{''.join(labels)}")
            branch.update({number: label for number, label in zip(numbers, labels)})
    previous = "base"
    for number, overlay in enumerate(overlays):
        start = _number(overlay["timeline_start"])
        end = _number(overlay["timeline_end"])
        source_in = _number(overlay["source_in"])
        source_out = _number(overlay["source_out"])
        if abs((overlay["source_out"] - overlay["source_in"]) -
               (overlay["timeline_end"] - overlay["timeline_start"])) > TIMING_TOLERANCE:
            raise ValueError(f"B-roll duration mismatch for shot {overlay['shot_id']}")
        filters.append(
            f"{branch[number]}trim=start={source_in}:end={source_out},"
            f"setpts=PTS-STARTPTS,fps={FPS},"
            f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT},setsar=1,format=yuv420p,"
            f"setpts=PTS+{start}/TB[shot{number}]"
        )
        output = f"layer{number}" if number < len(overlays) - 1 else "vout"
        filters.append(
            f"[{previous}][shot{number}]overlay=x=0:y=0:"
            f"enable='gte(t,{start})*lt(t,{end})':"
            f"repeatlast=0:eof_action=pass:shortest=0[{output}]"
        )
        previous = output
    if not overlays:
        filters.append("[base]null[vout]")
    command.extend(["-filter_complex", ";".join(filters),
                    "-map", "[vout]", "-map", f"0:{audio[0]['index']}",
                    "-af", f"atrim=duration={duration},asetpts=PTS-STARTPTS,aresample=48000",
                    "-c:v", encoder])
    if encoder == "libx264":
        command.extend(["-preset", "veryfast", "-crf", "22"])
    else:
        command.extend(["-b:v", "8M"])
    command.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    "-r", str(FPS), "-pix_fmt", "yuv420p", "-t", duration,
                    "-map_metadata", "-1", "-map_metadata:s:v:0", "-1",
                    "-map_metadata:s:a:0", "-1", "-map_chapters", "-1", "-dn", "-sn",
                    "-movflags", "+faststart", str(destination)])
    return command


def validate_edited_output(path: Path, expected_duration: float) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("Edited render is missing or empty")
    info = probe(path)
    videos, audios = _streams(info, "video"), _streams(info, "audio")
    if len(videos) != 1 or len(audios) != 1 or any(
        item.get("codec_type") not in {"video", "audio"} for item in info.get("streams", [])
    ):
        raise ValueError("Edited render must contain exactly one video and one audio stream, with no data streams")
    video, audio = videos[0], audios[0]
    if (video.get("codec_name") != "h264" or audio.get("codec_name") != "aac" or
            video.get("width") != WIDTH or video.get("height") != HEIGHT):
        raise ValueError("Edited render has unexpected codec or resolution")
    if video.get("pix_fmt") is not None and video["pix_fmt"] != "yuv420p":
        raise ValueError("Edited render has unexpected pixel format")
    try:
        fps = float(Fraction(video.get("avg_frame_rate", "0/1")))
        duration = probe_duration(info)
        audio_duration = float(audio.get("duration", "nan"))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Edited render has invalid frame rate or duration") from exc
    if not math.isfinite(fps) or abs(fps - FPS) > 0.1 or abs(duration - expected_duration) > 0.15:
        raise ValueError("Edited render frame rate or duration differs from timeline")
    if not math.isfinite(audio_duration) or audio_duration < expected_duration - 0.15:
        raise ValueError("Edited narration audio does not cover the timeline duration")
    return {"width": WIDTH, "height": HEIGHT, "fps": fps, "duration": duration,
            "video_codec": "h264", "audio_codec": "aac",
            "video_streams": len(videos), "audio_streams": len(audios)}


def validate_render_manifest(document: dict[str, Any], project_name: str) -> dict[str, Any]:
    schema = json.loads(RENDER_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid render_manifest.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    if document["project_name"] != project_name:
        raise ValueError("Render manifest project_name mismatch")
    return document


class EditRenderer:
    """Render to a temporary MP4, validate it, then publish a final draft."""

    def render(self, project_dir: Path, output_dir: Path, project: dict[str, Any],
               timeline: dict[str, Any], clip_plan: dict[str, Any], plan: dict[str, Any],
               manifest: dict[str, Any],
               encoder: str = "libx264", overwrite: bool = False,
               report: Callable[[str], None] | None = None,
               before_render: Callable[[], None] | None = None,
               before_validate: Callable[[], None] | None = None) -> dict[str, Any]:
        final = output_dir / "edited_draft.mp4"
        temporary = output_dir / "edited_draft.mp4.part.mp4"
        old_draft = output_dir / "draft.mp4"
        if final.exists() and not overwrite:
            raise FileExistsError(f"Edited draft already exists: {final}; use --overwrite-render")
        if temporary.exists():
            raise FileExistsError(f"Temporary edited draft already exists: {temporary}")
        if final.is_symlink() or temporary.is_symlink() or output_dir.is_symlink():
            raise ValueError("Edited render output path must not be a symlink")
        if old_draft == final or old_draft == temporary:
            raise ValueError("Edited draft path conflicts with V1 draft")
        primary_info, visual_sources = inspect_render_inputs(project, timeline, clip_plan, plan, manifest, project_dir)
        if report:
            for overlay in timeline["visual_overlays"]:
                report(f"overlay shot {overlay['shot_id']}: "
                       f"{overlay['timeline_start']:.2f}-{overlay['timeline_end']:.2f} <- "
                       f"{Path(overlay['source_path']).name} "
                       f"[{overlay['source_in']:.2f}-{overlay['source_out']:.2f}]")
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_edit_render_command(project, timeline, project_dir, temporary,
                                            primary_info, visual_sources, encoder)
        if before_render:
            before_render()
        completed = False
        published = False
        try:
            run_command(command)
            if before_validate:
                before_validate()
            facts = validate_edited_output(temporary, timeline["duration"])
            manifest = validate_render_manifest({
                "schema_version": SCHEMA_VERSION, "project_name": project_dir.name,
                "input_edit_timeline": "edit_timeline.json", "output_path": str(final),
                "encoder": encoder, **facts, "broll_overlay_count": len(timeline["visual_overlays"]),
                "metadata_stripped": True, "file_size_bytes": temporary.stat().st_size,
                "sha256": sha256_file(temporary),
            }, project_dir.name)
            if not overwrite:
                os.link(temporary, final)
                temporary.unlink()
            else:
                os.replace(temporary, final)
            published = True
            write_json(project_dir / "render_manifest.json", manifest)
            completed = True
            return manifest
        finally:
            if temporary.exists():
                temporary.unlink()
            if published and not completed and not overwrite and final.exists():
                final.unlink()
