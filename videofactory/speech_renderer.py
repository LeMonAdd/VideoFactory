"""Render the frozen V4A keep map from original media into a separate draft."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator

from . import SCHEMA_VERSION
from .clip_planner import resolve_project_media
from .edit_renderer import (_number, _streams, inspect_render_inputs,
                            validate_edited_output, WIDTH, HEIGHT, FPS)
from .ffmpeg_utils import FFMPEG, run_command
from .models import read_json, write_json
from .paths import ROOT
from .retimed_edit import build_retimed_edit_timeline
from .source_downloader import sha256_file

MANIFEST_SCHEMA = ROOT / "config" / "speech_render_manifest.schema.json"


def build_speech_render_command(retimed: dict[str, Any], project_dir: Path, destination: Path,
                                primary_info: dict[str, Any],
                                visual_sources: dict[Path, dict[str, Any]],
                                encoder: str = "libx264",
                                primary_scales: list[float] | None = None) -> list[str]:
    if encoder not in {"libx264", "h264_videotoolbox"}:
        raise ValueError(f"Unsupported H.264 encoder: {encoder}")
    videos, audios = _streams(primary_info, "video"), _streams(primary_info, "audio")
    if not videos or not audios or not isinstance(videos[0].get("index"), int) or not isinstance(audios[0].get("index"), int):
        raise ValueError("Primary source lacks identified video or audio stream")
    if retimed["base_video"]["keep_segments"] != retimed["base_audio"]["keep_segments"]:
        raise ValueError("Primary video and audio keep segments differ")
    keeps = retimed["base_video"]["keep_segments"]
    if primary_scales is not None:
        if len(primary_scales) != len(keeps) or any(
            isinstance(scale, bool) or not isinstance(scale, (int, float)) or
            not math.isfinite(scale) or not 1 <= scale <= 1.20 for scale in primary_scales
        ):
            raise ValueError("Invalid primary segment framing scales")
    command = [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-i", retimed["base_video"]["source_path"]]
    sources = list(visual_sources)
    input_index = {source: index + 1 for index, source in enumerate(sources)}
    for source in sources:
        command.extend(["-i", str(source)])
    filters: list[str] = []
    count = len(keeps)
    if count > 1:
        filters.append(f"[0:{videos[0]['index']}]split={count}" + "".join(f"[vsrc{i}]" for i in range(count)))
        filters.append(f"[0:{audios[0]['index']}]asplit={count}" + "".join(f"[asrc{i}]" for i in range(count)))
    for i, keep in enumerate(keeps):
        video_input = f"[vsrc{i}]" if count > 1 else f"[0:{videos[0]['index']}]"
        audio_input = f"[asrc{i}]" if count > 1 else f"[0:{audios[0]['index']}]"
        start, end = _number(keep["source_start"]), _number(keep["source_end"])
        video_filter = (f"{video_input}trim=start={start}:end={end},setpts=PTS-STARTPTS,"
                        f"fps={FPS},scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                        f"crop={WIDTH}:{HEIGHT},setsar=1,format=yuv420p")
        scale = primary_scales[i] if primary_scales is not None else 1.0
        if scale > 1:
            # -2 derives an even height from the width while preserving aspect ratio.
            punch_width = math.ceil(WIDTH * scale / 2) * 2
            video_filter += (f",scale={punch_width}:-2,"
                             f"crop={WIDTH}:{HEIGHT}:(iw-{WIDTH})/2:(ih-{HEIGHT})/2,"
                             "setsar=1,format=yuv420p")
        filters.append(f"{video_filter}[vkeep{i}]")
        filters.append(f"{audio_input}atrim=start={start}:end={end},"
                       f"asetpts=PTS-STARTPTS,aresample=48000[akeep{i}]")
    if count > 1:
        filters.append("".join(f"[vkeep{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0[base]")
        filters.append("".join(f"[akeep{i}]" for i in range(count)) + f"concat=n={count}:v=0:a=1[aout]")
    else:
        filters.extend(["[vkeep0]null[base]", "[akeep0]anull[aout]"])
    overlays = retimed["visual_overlays"]
    uses: dict[Path, list[int]] = {}
    for number, overlay in enumerate(overlays):
        source = resolve_project_media(project_dir, overlay["source_path"])
        if source not in input_index:
            raise ValueError(f"B-roll source was not inspected for shot {overlay['shot_id']}")
        uses.setdefault(source, []).append(number)
    branches: dict[int, str] = {}
    for source, numbers in uses.items():
        index = input_index[source]
        if len(numbers) == 1:
            branches[numbers[0]] = f"[{index}:v:0]"
        else:
            labels = [f"[src{index}_{number}]" for number in numbers]
            filters.append(f"[{index}:v:0]split={len(numbers)}{''.join(labels)}")
            branches.update(zip(numbers, labels))
    previous = "base"
    for number, overlay in enumerate(overlays):
        start, end = _number(overlay["timeline_start"]), _number(overlay["timeline_end"])
        source_in, source_out = _number(overlay["source_in"]), _number(overlay["source_out"])
        if abs((overlay["source_out"] - overlay["source_in"]) -
               (overlay["timeline_end"] - overlay["timeline_start"])) > 0.003:
            raise ValueError(f"B-roll piece duration mismatch for shot {overlay['shot_id']}")
        filters.append(f"{branches[number]}trim=start={source_in}:end={source_out},"
                       f"setpts=PTS-STARTPTS,fps={FPS},"
                       f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
                       f"crop={WIDTH}:{HEIGHT},setsar=1,format=yuv420p,"
                       f"setpts=PTS+{start}/TB[shot{number}]")
        output = f"layer{number}" if number < len(overlays) - 1 else "vout"
        filters.append(f"[{previous}][shot{number}]overlay=x=0:y=0:"
                       f"enable='gte(t,{start})*lt(t,{end})':"
                       f"repeatlast=0:eof_action=pass:shortest=0[{output}]")
        previous = output
    if not overlays:
        filters.append("[base]null[vout]")
    duration = _number(retimed["duration"])
    command.extend(["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]",
                    "-c:v", encoder])
    if encoder == "libx264":
        command.extend(["-preset", "veryfast", "-crf", "22"])
    else:
        command.extend(["-b:v", "8M"])
    command.extend(["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-r", str(FPS),
                    "-pix_fmt", "yuv420p", "-t", duration,
                    "-map_metadata", "-1", "-map_metadata:s:v:0", "-1",
                    "-map_metadata:s:a:0", "-1", "-map_chapters", "-1", "-dn", "-sn",
                    "-movflags", "+faststart", str(destination)])
    return command


def validate_speech_render_manifest(document: dict[str, Any], project_name: str) -> dict[str, Any]:
    schema = json.loads(MANIFEST_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(document))
    if errors:
        first = errors[0]
        raise ValueError(f"Invalid speech_render_manifest.json at {'.'.join(map(str, first.path)) or 'root'}: {first.message}")
    if document["project_name"] != project_name:
        raise ValueError("Speech render manifest project_name mismatch")
    return document


class SpeechEditRenderer:
    def render(self, project_dir: Path, output_dir: Path, project: dict[str, Any],
               transcript: dict[str, Any], director_plan: dict[str, Any],
               clip_plan: dict[str, Any], timeline: dict[str, Any],
               download_manifest: dict[str, Any], speech_plan: dict[str, Any],
               retime_map: dict[str, Any], retimed: dict[str, Any],
               encoder: str = "libx264", overwrite: bool = False,
               before_render: Callable[[], None] | None = None,
               before_validate: Callable[[], None] | None = None) -> dict[str, Any]:
        final = output_dir / "speech_edited_draft.mp4"
        temporary = output_dir / "speech_edited_draft.mp4.part.mp4"
        if final.exists() and not overwrite:
            raise FileExistsError(f"Speech-edited draft already exists: {final}; use --overwrite-render")
        if temporary.exists():
            raise FileExistsError(f"Temporary speech-edited draft already exists: {temporary}")
        if output_dir.is_symlink() or final.is_symlink() or temporary.is_symlink():
            raise ValueError("Speech render output path must not be a symlink")
        expected = build_retimed_edit_timeline(project, transcript, director_plan, clip_plan,
                                               timeline, speech_plan, retime_map, project_dir)
        if retimed != expected:
            raise ValueError("Saved retimed edit timeline differs from canonical transformation")
        for filename, document in (("speech_edit_plan.json", speech_plan),
                                   ("retime_map.json", retime_map),
                                   ("retimed_edit_timeline.json", retimed)):
            if read_json(project_dir / filename) != document:
                raise ValueError(f"Saved {filename} changed before speech rendering")
        input_hashes = {filename: sha256_file(project_dir / filename) for filename in
                        ("speech_edit_plan.json", "retime_map.json", "retimed_edit_timeline.json")}
        primary_info, visual_sources = inspect_render_inputs(
            project, timeline, clip_plan, director_plan, download_manifest, project_dir)
        command = build_speech_render_command(retimed, project_dir, temporary,
                                              primary_info, visual_sources, encoder)
        output_dir.mkdir(parents=True, exist_ok=True)
        if before_render:
            before_render()
        published = False
        completed = False
        try:
            run_command(command)
            if before_validate:
                before_validate()
            facts = validate_edited_output(temporary, speech_plan["edited_duration"])
            if any(sha256_file(project_dir / filename) != digest for filename, digest in input_hashes.items()):
                raise ValueError("Speech planning artifacts changed during rendering")
            manifest = validate_speech_render_manifest({
                "schema_version": SCHEMA_VERSION, "project_name": project_dir.name,
                "input_edit_timeline": "edit_timeline.json",
                "input_speech_edit_plan": "speech_edit_plan.json",
                "input_retime_map": "retime_map.json",
                "input_retimed_edit_timeline": "retimed_edit_timeline.json",
                "output_path": str(final), "encoder": encoder,
                "source_duration": speech_plan["source_duration"],
                "edited_duration": speech_plan["edited_duration"],
                "total_removed_duration": speech_plan["total_removed_duration"],
                "keep_segment_count": len(retime_map["segments"]),
                "original_broll_overlay_count": len(timeline["visual_overlays"]),
                "retimed_broll_overlay_segment_count": len(retimed["visual_overlays"]),
                **facts, "metadata_stripped": True,
                "file_size_bytes": temporary.stat().st_size, "sha256": sha256_file(temporary),
                "speech_edit_plan_sha256": input_hashes["speech_edit_plan.json"],
                "retime_map_sha256": input_hashes["retime_map.json"],
                "retimed_edit_timeline_sha256": input_hashes["retimed_edit_timeline.json"],
            }, project_dir.name)
            if overwrite:
                os.replace(temporary, final)
            else:
                os.link(temporary, final)
                temporary.unlink()
            published = True
            write_json(project_dir / "speech_render_manifest.json", manifest)
            completed = True
            return manifest
        finally:
            if temporary.exists():
                temporary.unlink()
            if published and not completed and not overwrite and final.exists():
                final.unlink()
