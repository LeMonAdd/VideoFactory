"""VideoFactory command-line orchestration."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION
from .asset_manager import discover_assets
from .audio import extract_narration
from .clip_planner import build_clip_plan, validate_clip_plan
from .director_provider import CodexDirector, RuleBasedDirector
from .director_plan import validate_plan
from .director_scenes import convert_plan_to_scenes
from .director_service import create_director_plan
from .input_handler import inspect_input
from .edit_timeline import build_edit_timeline, validate_edit_timeline
from .edit_renderer import EditRenderer
from .retimed_edit import build_retimed_edit_timeline
from .speech_renderer import SpeechEditRenderer
from .jump_cut_style import (build_jump_cut_style_plan, build_styled_edit_timeline,
                             validate_punch_in_scale)
from .punch_in_renderer import PunchInRenderer
from .caption_export import CaptionExporter
from .media_probe import duration as probe_duration
from .media_probe import probe, stream
from .models import read_json, write_json
from .paths import ROOT, ProjectPaths, safe_project_name
from .progress import ProgressReporter
from .renderer import render
from .source_downloader import (DEFAULT_MAX_DOWNLOAD_MB, SelectedMediaDownloader,
                                max_download_bytes, selected_assets, validate_manifest)
from .source_finder import find_sources
from .source_provider import LocalSourceProvider, PexelsSourceProvider
from .source_selection import build_selection_context, validate_selection
from .source_selection_service import create_source_selection
from .source_selector import CodexSourceSelector, RuleBasedSourceSelector
from .source_models import validate_sources_document
from .speech_edit import (SpeechEditPlanner, build_retime_map, validate_inputs,
                          validate_parameters, validate_retime_map,
                          validate_saved_speech_edit_plan)
from .silence_detector import LocalAudioSilenceDetector, validate_noise_db
from .timeline import build_timeline
from .transcriber import FixtureTranscriber, RealMLXTranscriber
from .validator import validate_output

DEFAULT_SETTINGS = {"width": 1920, "height": 1080, "fps": 30,
                    "video_codec": "h264", "audio_codec": "aac"}


def default_whisper_model() -> str:
    config = read_json(ROOT / "config" / "transcription.json")
    model = config.get("default_whisper_model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("config/transcription.json needs a non-empty default_whisper_model")
    return model


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Local VideoFactory editing pipeline")
    source = result.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Talking-head video with narration")
    source.add_argument("--voice", type=Path, help="Standalone voiceover audio")
    result.add_argument("--project", required=True, help="Safe project name")
    result.add_argument("--director", choices=["rule", "codex"], default="rule",
                        help="Editorial provider (default: rule; Codex is opt-in)")
    result.add_argument("--director-model", help="Optional Codex model; omit for the user's Codex default")
    result.add_argument("--director-timeout", type=float, default=300,
                        help="Codex subprocess timeout in seconds (default: 300)")
    result.add_argument("--plan-only", action="store_true",
                        help="Create director_plan.json from an existing project transcript")
    result.add_argument("--find-sources", action="store_true",
                        help="Search candidates for an existing director plan; no download or render")
    result.add_argument("--source-provider", choices=["local", "pexels"],
                        help="Candidate provider for --find-sources; Pexels is opt-in")
    result.add_argument("--source-limit", type=int, default=5,
                        help="Maximum candidates per visual query (1-20; default: 5)")
    result.add_argument("--select-sources", action="store_true",
                        help="Select or reject existing candidates without search, download, or render")
    result.add_argument("--selector", choices=["rule", "codex"], default="rule",
                        help="Candidate selector (default: rule; Codex is opt-in)")
    result.add_argument("--selector-model", help="Optional Codex selector model")
    result.add_argument("--selector-timeout", type=float, default=300,
                        help="Codex selector subprocess timeout in seconds (default: 300)")
    result.add_argument("--download-sources", action="store_true",
                        help="Download only selected unique source videos for an existing project")
    result.add_argument("--max-download-mb", type=float, default=DEFAULT_MAX_DOWNLOAD_MB,
                        help="Hard per-asset download limit in MiB (default: 500)")
    result.add_argument("--build-edit-timeline", action="store_true",
                        help="Plan local B-roll source ranges and a layered edit timeline; no render")
    result.add_argument("--render-edit", action="store_true",
                        help="Render the saved layered edit timeline to edited_draft.mp4")
    result.add_argument("--plan-speech-edits", action="store_true",
                        help="Plan conservative speech-gap cuts and a canonical time map without rendering")
    result.add_argument("--render-speech-edits", action="store_true",
                        help="Render the saved speech-shortened layered edit to speech_edited_draft.mp4")
    result.add_argument("--render-punch-ins", action="store_true",
                        help="Render static alternating primary framing to punch_in_draft.mp4")
    result.add_argument("--export-captions", action="store_true",
                        help="Export retimed SRT and WebVTT sidecar captions without rendering")
    result.add_argument("--overwrite-captions", action="store_true",
                        help="Replace existing generated caption sidecars")
    result.add_argument("--punch-in-scale", type=float, default=1.08,
                        help="Static framing scale on alternating primary segments (default: 1.08)")
    result.add_argument("--pause-threshold-seconds", type=float, default=1.0,
                        help="Minimum detected audio silence to consider for shortening (default: 1.0)")
    result.add_argument("--pause-keep-seconds", type=float, default=0.25,
                        help="Natural pause to retain around a cut (default: 0.25)")
    result.add_argument("--silence-noise-db", type=float, default=-35.0,
                        help="FFmpeg silencedetect threshold in dB (default: -35)")
    result.add_argument("--source-margin-seconds", type=float, default=0.5,
                        help="Preferred head/tail margin for source clips (default: 0.5)")
    result.add_argument("--fixture-transcript", type=Path,
                        help="Deterministic JSON transcript; bypasses MLX Whisper")
    result.add_argument("--whisper-model", "--mlx-model", dest="whisper_model",
                        default=default_whisper_model(),
                        help="Local MLX model path or Hugging Face model ID (default: project config)")
    result.add_argument("--language", help="Whisper language code, for example ru; omit to auto-detect")
    result.add_argument("--assets-dir", type=Path,
                        help="Local asset root with broll/ and images/ (default: assets/)")
    result.add_argument("--encoder", choices=["libx264", "h264_videotoolbox"],
                        default="libx264", help="H.264 encoder (default: libx264)")
    result.add_argument("--overwrite-render", action="store_true",
                        help="Replace an existing generated output for the selected render mode")
    return result


def director_provider(args: argparse.Namespace) -> RuleBasedDirector | CodexDirector:
    if args.director == "codex":
        return CodexDirector(model=args.director_model, timeout_seconds=args.director_timeout)
    return RuleBasedDirector()


def run_plan_only(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(3)
    progress.start(1, "Loading transcript")
    if not (paths.project / "project.json").is_file() or not (paths.project / "transcript.json").is_file():
        raise FileNotFoundError(f"Existing project and transcript are required under {paths.project}")
    project = read_json(paths.project / "project.json")
    transcript = read_json(paths.project / "transcript.json")
    if project.get("project_name") != name or project.get("mode") not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Existing project.json has a mismatched name or invalid mode")
    progress.complete(1, "Transcript loading")
    progress.start(2, "Creating director plan")
    plan = create_director_plan(name, project["mode"], transcript, director_provider(args))
    progress.complete(2, "Director plan creation")
    progress.start(3, "Validating and saving director_plan.json")
    destination = paths.project / "director_plan.json"
    write_json(destination, plan)
    progress.complete(3, "Director plan validation and save")
    return destination


def run_find_sources(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(4)
    progress.start(1, "Loading director plan")
    for filename in ("project.json", "transcript.json", "director_plan.json"):
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    project = read_json(paths.project / "project.json")
    transcript = read_json(paths.project / "transcript.json")
    plan = read_json(paths.project / "director_plan.json")
    if project.get("project_name") != name or project.get("mode") not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Existing project.json has a mismatched name or invalid mode")
    validate_plan(plan, name, float(transcript["duration"]), project["mode"])
    progress.complete(1, "Director plan loading")
    provider = (PexelsSourceProvider() if args.source_provider == "pexels" else
                LocalSourceProvider(ROOT, args.assets_dir.resolve() if args.assets_dir else None))
    existing_sources: list[dict] = []
    destination = paths.project / "sources.json"
    if destination.is_file():
        existing = read_json(destination)
        if not isinstance(existing.get("sources"), list):
            raise ValueError("Existing sources.json needs a sources list")
        existing_sources = existing["sources"]
    progress.start(2, "Searching source candidates")
    document = find_sources(plan, name, provider, args.source_limit, existing_sources,
                            report=lambda message: print(f"      {message}", flush=True))
    progress.complete(2, "Source candidate search")
    progress.start(3, "Validating sources")
    # find_sources validates before returning; retain a distinct visible stage.
    progress.complete(3, "Source validation")
    progress.start(4, "Saving sources.json")
    write_json(destination, document)
    progress.complete(4, "Source save")
    return destination


def run_select_sources(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(4)
    progress.start(1, "Loading director plan")
    for filename in ("project.json", "transcript.json", "director_plan.json"):
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    project = read_json(paths.project / "project.json")
    transcript = read_json(paths.project / "transcript.json")
    plan = read_json(paths.project / "director_plan.json")
    if project.get("project_name") != name or project.get("mode") not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Existing project.json has a mismatched name or invalid mode")
    validate_plan(plan, name, float(transcript["duration"]), project["mode"])
    progress.complete(1, "Director plan loading")
    progress.start(2, "Loading source candidates")
    source_path = paths.project / "sources.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Existing project needs {source_path}")
    sources = read_json(source_path)
    validate_sources_document(sources, name)
    build_selection_context(plan, sources, transcript)
    progress.complete(2, "Source candidate loading")
    progress.start(3, "Selecting candidates")
    selector = (CodexSourceSelector(model=args.selector_model, timeout_seconds=args.selector_timeout)
                if args.selector == "codex" else RuleBasedSourceSelector())
    selection = create_source_selection(plan, sources, selector, transcript)
    progress.complete(3, "Candidate selection")
    progress.start(4, "Validating and saving source_selection.json")
    validate_selection(selection, plan, sources)
    destination = paths.project / "source_selection.json"
    write_json(destination, selection)
    progress.complete(4, "Source selection validation and save")
    return destination


def run_download_sources(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(5)
    progress.start(1, "Loading source candidates")
    source_path = paths.project / "sources.json"
    if not source_path.is_file():
        raise FileNotFoundError(f"Existing project needs {source_path}")
    sources = read_json(source_path)
    validate_sources_document(sources, name)
    progress.complete(1, "Source candidate loading")
    progress.start(2, "Loading source selection")
    selection_path = paths.project / "source_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Existing project needs {selection_path}")
    selection = read_json(selection_path)
    plan_path = paths.project / "director_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Existing project needs {plan_path}")
    plan = read_json(plan_path)
    progress.complete(2, "Source selection loading")
    progress.start(3, "Resolving selected assets")
    selected_assets(plan, sources, selection)
    downloader = SelectedMediaDownloader(max_download_mb=args.max_download_mb)
    progress.complete(3, "Selected asset resolution")
    progress.start(4, "Downloading and validating media")
    def begin_save() -> None:
        progress.complete(4, "Media download and validation")
        progress.start(5, "Saving download_manifest.json")
    downloader.run(paths.project, plan, sources, selection,
                   report=lambda message: print(f"      {message}", flush=True), before_save=begin_save)
    progress.complete(5, "Download manifest save")
    return paths.project / "download_manifest.json"


def run_build_edit_timeline(args: argparse.Namespace) -> tuple[Path, Path]:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(6)
    progress.start(1, "Loading project metadata")
    project_path = paths.project / "project.json"
    if not project_path.is_file():
        raise FileNotFoundError(f"Existing project needs {project_path}")
    project = read_json(project_path)
    mode = project.get("mode")
    if project.get("project_name") != name or mode not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Existing project.json has a mismatched name or invalid mode")
    primary = Path(project["source_media"])
    info = probe(primary)
    if stream(info, "audio") is None or (mode == "TALKING_HEAD" and stream(info, "video") is None):
        raise ValueError("Primary source lacks required narration audio or talking-head video")
    duration = round(probe_duration(info), 3)
    progress.complete(1, "Project metadata loading")
    progress.start(2, "Loading director plan")
    plan_path = paths.project / "director_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Existing project needs {plan_path}")
    plan = read_json(plan_path)
    validate_plan(plan, name, duration, mode)
    progress.complete(2, "Director plan loading")
    progress.start(3, "Loading source selection")
    selection_path = paths.project / "source_selection.json"
    sources_path = paths.project / "sources.json"
    for path in (selection_path, sources_path):
        if not path.is_file():
            raise FileNotFoundError(f"Existing project needs {path}")
    selection = read_json(selection_path)
    sources = read_json(sources_path)
    validate_selection(selection, plan, sources)
    progress.complete(3, "Source selection loading")
    progress.start(4, "Loading downloaded media manifest")
    manifest_path = paths.project / "download_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Existing project needs {manifest_path}")
    manifest = read_json(manifest_path)
    validate_manifest(manifest, name)
    progress.complete(4, "Download manifest loading")
    progress.start(5, "Planning source clips")
    clip_plan = build_clip_plan(paths.project, mode, duration, plan, sources, selection,
                                manifest, args.source_margin_seconds)
    for clip in clip_plan["clips"]:
        if clip["status"] == "PLANNED":
            print(f"      shot {clip['shot_id']} -> B-roll {clip['candidate_id']} "
                  f"[{clip['source_in']:.2f}-{clip['source_out']:.2f}]", flush=True)
        elif clip["status"] in {"A_ROLL_FALLBACK", "GRAPHIC_FALLBACK"}:
            print(f"      shot {clip['shot_id']} -> {clip['resolved_visual_type']} fallback", flush=True)
    progress.complete(5, "Source clip planning")
    progress.start(6, "Validating and saving edit timeline")
    edit_timeline = build_edit_timeline(project, duration, plan, clip_plan, paths.project)
    validate_clip_plan(clip_plan, plan, mode, duration, paths.project)
    validate_edit_timeline(edit_timeline, project, plan, clip_plan, paths.project)
    clip_path = paths.project / "clip_plan.json"
    edit_path = paths.project / "edit_timeline.json"
    write_json(clip_path, clip_plan)
    write_json(edit_path, edit_timeline)
    progress.complete(6, "Edit timeline validation and save")
    return clip_path, edit_path


def run_render_edit(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(5)
    progress.start(1, "Loading edit timeline")
    required = ("project.json", "director_plan.json", "clip_plan.json", "edit_timeline.json",
                "download_manifest.json")
    for filename in required:
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    project = read_json(paths.project / "project.json")
    plan = read_json(paths.project / "director_plan.json")
    clips = read_json(paths.project / "clip_plan.json")
    timeline = read_json(paths.project / "edit_timeline.json")
    manifest = validate_manifest(read_json(paths.project / "download_manifest.json"), name)
    if project.get("project_name") != name or project.get("mode") not in {"TALKING_HEAD", "VOICEOVER"}:
        raise ValueError("Existing project.json has a mismatched name or invalid mode")
    progress.complete(1, "Edit timeline loading")
    progress.start(2, "Validating render inputs")
    validate_plan(plan, name, timeline["duration"], project["mode"])
    validate_edit_timeline(timeline, project, plan, clips, paths.project)
    if project["mode"] != "TALKING_HEAD":
        raise ValueError("VOICEOVER edit rendering requires a renderable base visual and is not yet supported by V3D")
    progress.complete(2, "Render input validation")
    progress.start(3, "Building FFmpeg render graph")
    def begin_render() -> None:
        progress.complete(3, "FFmpeg render graph creation")
        progress.start(4, "Rendering edited draft")
    def begin_save() -> None:
        progress.complete(4, "Edited draft rendering")
        progress.start(5, "Validating and saving render result")
    EditRenderer().render(paths.project, paths.output, project, timeline, clips, plan, manifest,
                          encoder=args.encoder, overwrite=args.overwrite_render,
                          report=lambda message: print(f"      {message}", flush=True),
                          before_render=begin_render, before_validate=begin_save)
    progress.complete(5, "Render result validation and save")
    return paths.output / "edited_draft.mp4"


def run_plan_speech_edits(args: argparse.Namespace) -> tuple[Path, Path]:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(5)
    progress.start(1, "Loading transcript")
    required = ("project.json", "transcript.json", "edit_timeline.json")
    for filename in required:
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    project = read_json(paths.project / "project.json")
    transcript = read_json(paths.project / "transcript.json")
    timeline = read_json(paths.project / "edit_timeline.json")
    progress.complete(1, "Transcript loading")
    progress.start(2, "Validating word timestamps")
    validate_inputs(project, transcript, timeline)
    progress.complete(2, "Word timestamp validation")
    progress.start(3, "Detecting removable pauses")
    silences = LocalAudioSilenceDetector().detect(
        Path(project["source_media"]), timeline["duration"],
        args.pause_threshold_seconds, args.silence_noise_db)
    plan = SpeechEditPlanner().plan(project, transcript, timeline, silences,
                                    args.pause_threshold_seconds, args.pause_keep_seconds,
                                    args.silence_noise_db)
    progress.complete(3, "Pause detection")
    progress.start(4, "Building deterministic time map")
    retime = build_retime_map(plan)
    progress.complete(4, "Time map construction")
    progress.start(5, "Validating and saving speech edit plan")
    plan_path = paths.project / "speech_edit_plan.json"
    map_path = paths.project / "retime_map.json"
    write_json(plan_path, plan)
    write_json(map_path, retime)
    progress.complete(5, "Speech edit plan validation and save")
    for cut in plan["cuts"]:
        print(f"pause {cut['id']}: {cut['source_start']:.2f}-{cut['source_end']:.2f} "
              f"removed {cut['removed_duration']:.2f}s", flush=True)
    print(f"Original duration: {plan['source_duration']:.2f}s\n"
          f"Removed: {plan['total_removed_duration']:.2f}s\n"
          f"Edited duration: {plan['edited_duration']:.2f}s", flush=True)
    return plan_path, map_path


def run_render_speech_edits(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    final = paths.output / "speech_edited_draft.mp4"
    if final.exists() and not args.overwrite_render:
        raise FileExistsError(f"Speech-edited draft already exists: {final}; use --overwrite-render")
    progress = ProgressReporter(6)
    progress.start(1, "Loading speech edit artifacts")
    required = ("project.json", "transcript.json", "director_plan.json", "clip_plan.json",
                "edit_timeline.json", "download_manifest.json", "speech_edit_plan.json",
                "retime_map.json")
    for filename in required:
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    project = read_json(paths.project / "project.json")
    transcript = read_json(paths.project / "transcript.json")
    director_plan = read_json(paths.project / "director_plan.json")
    clip_plan = read_json(paths.project / "clip_plan.json")
    timeline = read_json(paths.project / "edit_timeline.json")
    download_manifest = validate_manifest(read_json(paths.project / "download_manifest.json"), name)
    speech_plan = read_json(paths.project / "speech_edit_plan.json")
    retime_map = read_json(paths.project / "retime_map.json")
    progress.complete(1, "Speech edit artifact loading")
    progress.start(2, "Validating canonical retime map")
    validate_plan(director_plan, name, timeline["duration"], project["mode"])
    validate_saved_speech_edit_plan(speech_plan, project, transcript, timeline)
    validate_retime_map(retime_map, speech_plan)
    progress.complete(2, "Canonical retime map validation")
    progress.start(3, "Building retimed edit timeline")
    retimed = build_retimed_edit_timeline(project, transcript, director_plan, clip_plan,
                                          timeline, speech_plan, retime_map, paths.project)
    for number, keep in enumerate(retime_map["segments"], 1):
        print(f"      keep segment {number}: {keep['source_start']:.3f}-{keep['source_end']:.3f} "
              f"-> {keep['edited_start']:.3f}-{keep['edited_end']:.3f}", flush=True)
    for overlay in retimed["visual_overlays"]:
        print(f"      B-roll shot {overlay['shot_id']} piece {overlay['segment_index']}: "
              f"{overlay['original_timeline_start']:.3f}-{overlay['original_timeline_end']:.3f} "
              f"-> {overlay['timeline_start']:.3f}-{overlay['timeline_end']:.3f}", flush=True)
    write_json(paths.project / "retimed_edit_timeline.json", retimed)
    progress.complete(3, "Retimed edit timeline construction")
    progress.start(4, "Building FFmpeg render graph")
    def begin_render() -> None:
        progress.complete(4, "FFmpeg render graph construction")
        progress.start(5, "Rendering speech-edited draft")
    def begin_validate() -> None:
        progress.complete(5, "Speech-edited draft rendering")
        progress.start(6, "Validating and saving render result")
    SpeechEditRenderer().render(paths.project, paths.output, project, transcript,
                                director_plan, clip_plan, timeline, download_manifest,
                                speech_plan, retime_map, retimed, encoder=args.encoder,
                                overwrite=args.overwrite_render, before_render=begin_render,
                                before_validate=begin_validate)
    progress.complete(6, "Speech render validation and save")
    return final


def run_render_punch_ins(args: argparse.Namespace) -> Path:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    final = paths.output / "punch_in_draft.mp4"
    if final.exists() and not args.overwrite_render:
        raise FileExistsError(f"Punch-in draft already exists: {final}; use --overwrite-render")
    progress = ProgressReporter(6)
    progress.start(1, "Loading retimed edit artifacts")
    required = ("project.json", "transcript.json", "director_plan.json", "clip_plan.json",
                "edit_timeline.json", "download_manifest.json", "speech_edit_plan.json",
                "retime_map.json", "retimed_edit_timeline.json")
    for filename in required:
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    project = read_json(paths.project / "project.json")
    transcript = read_json(paths.project / "transcript.json")
    director_plan = read_json(paths.project / "director_plan.json")
    clip_plan = read_json(paths.project / "clip_plan.json")
    timeline = read_json(paths.project / "edit_timeline.json")
    download_manifest = validate_manifest(read_json(paths.project / "download_manifest.json"), name)
    speech_plan = read_json(paths.project / "speech_edit_plan.json")
    retime_map = read_json(paths.project / "retime_map.json")
    retimed = read_json(paths.project / "retimed_edit_timeline.json")
    validate_plan(director_plan, name, timeline["duration"], project["mode"])
    canonical = build_retimed_edit_timeline(project, transcript, director_plan, clip_plan,
                                             timeline, speech_plan, retime_map, paths.project)
    if retimed != canonical:
        raise ValueError("Saved retimed edit timeline differs from canonical transformation")
    progress.complete(1, "Retimed edit artifact loading")
    progress.start(2, "Building jump-cut style plan")
    style_plan = build_jump_cut_style_plan(retimed, args.punch_in_scale)
    progress.complete(2, "Jump-cut style plan construction")
    progress.start(3, "Building styled edit timeline")
    styled = build_styled_edit_timeline(retimed, style_plan)
    for segment in style_plan["segments"]:
        print(f"      A-roll segment {segment['segment_index']}: "
              f"{segment['edited_start']:.3f}-{segment['edited_end']:.3f} "
              f"scale {segment['scale']:.2f}", flush=True)
    for overlay in styled["visual_overlays"]:
        print(f"      B-roll shot {overlay['shot_id']} piece {overlay['segment_index']}: "
              f"{overlay['timeline_start']:.3f}-{overlay['timeline_end']:.3f}", flush=True)
    write_json(paths.project / "jump_cut_style_plan.json", style_plan)
    write_json(paths.project / "styled_edit_timeline.json", styled)
    progress.complete(3, "Styled edit timeline construction")
    progress.start(4, "Building FFmpeg render graph")
    def begin_render() -> None:
        progress.complete(4, "FFmpeg render graph construction")
        progress.start(5, "Rendering punch-in draft")
    def begin_validate() -> None:
        progress.complete(5, "Punch-in draft rendering")
        progress.start(6, "Validating and saving render result")
    PunchInRenderer().render(paths.project, paths.output, project, transcript, director_plan,
                             clip_plan, timeline, download_manifest, speech_plan, retime_map,
                             retimed, style_plan, styled, encoder=args.encoder,
                             overwrite=args.overwrite_render, before_render=begin_render,
                             before_validate=begin_validate)
    progress.complete(6, "Punch-in render validation and save")
    return final


def run_export_captions(args: argparse.Namespace) -> dict:
    name = safe_project_name(args.project)
    paths = ProjectPaths(ROOT, name)
    progress = ProgressReporter(5)
    progress.start(1, "Loading caption artifacts")
    required = ("transcript.json", "retime_map.json", "retimed_edit_timeline.json")
    for filename in required:
        if not (paths.project / filename).is_file():
            raise FileNotFoundError(f"Existing project needs {paths.project / filename}")
    transcript = read_json(paths.project / "transcript.json")
    retime_map = read_json(paths.project / "retime_map.json")
    retimed = read_json(paths.project / "retimed_edit_timeline.json")
    project_path = paths.project / "project.json"
    project = read_json(project_path) if project_path.is_file() else {}
    if project and project.get("project_name") != name:
        raise ValueError("Existing project.json has a mismatched name")
    transcription = project.get("transcription", {})
    project_language = transcription.get("language") if isinstance(transcription, dict) else None
    progress.complete(1, "Caption artifact loading")
    progress.start(2, "Validating final retime timeline")
    from .caption_timeline import validate_caption_inputs
    validate_caption_inputs(name, transcript, retime_map, retimed)
    progress.complete(2, "Final retime timeline validation")
    progress.start(3, "Building caption timeline")
    from .caption_timeline import build_caption_timeline
    build_caption_timeline(name, transcript, retime_map, retimed,
                           project_language=project_language)
    progress.complete(3, "Caption timeline construction")
    progress.start(4, "Exporting SRT and WebVTT")
    manifest = CaptionExporter().export(paths.project, paths.output, name, transcript,
                                        retime_map, retimed, overwrite=args.overwrite_captions,
                                        project_language=project_language)
    progress.complete(4, "Caption sidecar export")
    progress.start(5, "Validating and saving caption export")
    progress.complete(5, "Caption export validation and save")
    return manifest


def run(args: argparse.Namespace) -> dict:
    name = safe_project_name(args.project)
    mode = "TALKING_HEAD" if args.video else "VOICEOVER"
    paths = ProjectPaths(ROOT, name)
    if paths.draft.exists() and not args.overwrite_render:
        raise FileExistsError(f"Draft already exists: {paths.draft}; use --overwrite-render")
    progress = ProgressReporter(8)
    progress.start(1, "Inspecting source media")
    source, info, media_duration = inspect_input(args.video or args.voice, mode)
    progress.complete(1, "Source inspection")
    paths.create()
    project = {
        "schema_version": SCHEMA_VERSION, "project_name": name, "mode": mode,
        "source_media": str(source), "created_at": datetime.now(timezone.utc).isoformat(),
        "output_settings": DEFAULT_SETTINGS,
        "transcription": {
            "backend": "fixture" if args.fixture_transcript else "mlx_whisper",
            "model": None if args.fixture_transcript else args.whisper_model,
            "language": args.language or "auto",
        },
        "director": {"provider": args.director, "model": args.director_model if args.director == "codex" else None},
    }
    write_json(paths.project / "project.json", project)
    progress.start(2, "Extracting narration")
    narration = extract_narration(source, paths.temp / "narration.wav")
    progress.complete(2, "Narration extraction")
    transcriber = (FixtureTranscriber(args.fixture_transcript) if args.fixture_transcript
                   else RealMLXTranscriber(paths.temp, args.whisper_model, args.language))
    label = "Loading fixture transcript" if args.fixture_transcript else f"Transcribing with {args.whisper_model}"
    progress.start(3, label)
    transcript = transcriber.transcribe(narration, media_duration)
    write_json(paths.project / "transcript.json", transcript)
    progress.complete(3, "Transcription")
    progress.start(4, "Creating director plan")
    plan = create_director_plan(name, mode, transcript, director_provider(args))
    write_json(paths.project / "director_plan.json", plan)
    progress.complete(4, "Director plan creation")
    progress.start(5, "Resolving local assets")
    sources = discover_assets(ROOT, mode, source if mode == "TALKING_HEAD" else None,
                              info if mode == "TALKING_HEAD" else None,
                              args.assets_dir.resolve() if args.assets_dir else None)
    write_json(paths.project / "sources.json", sources)
    scenes = convert_plan_to_scenes(plan, transcript, sources, mode)
    write_json(paths.project / "scenes.json", scenes)
    progress.complete(5, "Local asset resolution")
    progress.start(6, "Building timeline")
    timeline = build_timeline(scenes, sources, source, mode, DEFAULT_SETTINGS)
    write_json(paths.project / "timeline.json", timeline)
    progress.complete(6, "Timeline creation")
    progress.start(7, "Rendering draft")
    render(timeline, paths.draft, args.encoder, args.overwrite_render)
    progress.complete(7, "Draft rendering")
    progress.start(8, "Validating output")
    result = validate_output(paths.draft, DEFAULT_SETTINGS)
    progress.complete(8, "Output validation")
    return result


def main(argv: list[str] | None = None) -> int:
    command_parser = parser()
    args = command_parser.parse_args(argv)
    supplied_args = argv if argv is not None else sys.argv[1:]
    workflows = sum((args.plan_only, args.find_sources, args.select_sources,
                     args.download_sources, args.build_edit_timeline, args.render_edit,
                     args.plan_speech_edits, args.render_speech_edits, args.render_punch_ins,
                     args.export_captions))
    if workflows > 1:
        command_parser.error("Choose only one existing-project workflow")
    if args.find_sources:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--find-sources uses an existing project and cannot take source, fixture, or --plan-only arguments")
        if not args.source_provider:
            command_parser.error("--find-sources requires --source-provider local or pexels")
    elif args.select_sources:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--select-sources uses an existing project and cannot take source, fixture, or --plan-only arguments")
    elif args.download_sources:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--download-sources uses an existing project and cannot take source or fixture arguments")
    elif args.build_edit_timeline:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--build-edit-timeline uses an existing project and cannot take source or fixture arguments")
    elif args.render_edit:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--render-edit uses an existing project and cannot take source or fixture arguments")
    elif args.plan_speech_edits:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--plan-speech-edits uses an existing project and cannot take source or fixture arguments")
    elif args.render_speech_edits:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--render-speech-edits uses an existing project and cannot take source or fixture arguments")
    elif args.render_punch_ins:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--render-punch-ins uses an existing project and cannot take source or fixture arguments")
    elif args.export_captions:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--export-captions uses an existing project and cannot take source or fixture arguments")
    elif args.source_provider:
        command_parser.error("--source-provider requires --find-sources")
    elif args.plan_only:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--plan-only uses an existing project and cannot take source or fixture arguments")
    elif not (args.video or args.voice):
        command_parser.error("one of --video or --voice is required unless an existing-project workflow is used")
    if not 1 <= args.source_limit <= 20:
        command_parser.error("--source-limit must be between 1 and 20")
    if args.source_provider and not args.find_sources:
        command_parser.error("--source-provider requires --find-sources")
    if args.selector == "codex" and not args.select_sources:
        command_parser.error("--selector codex requires --select-sources")
    if (args.selector_model or args.selector_timeout != 300) and not args.select_sources:
        command_parser.error("--selector-model and --selector-timeout require --select-sources")
    if args.max_download_mb != DEFAULT_MAX_DOWNLOAD_MB and not args.download_sources:
        command_parser.error("--max-download-mb requires --download-sources")
    if args.source_margin_seconds != 0.5 and not args.build_edit_timeline:
        command_parser.error("--source-margin-seconds requires --build-edit-timeline")
    if not math.isfinite(args.source_margin_seconds) or args.source_margin_seconds < 0:
        command_parser.error("--source-margin-seconds must be finite and non-negative")
    if ((args.pause_threshold_seconds != 1.0 or args.pause_keep_seconds != 0.25 or
         args.silence_noise_db != -35.0)
            and not args.plan_speech_edits):
        command_parser.error("Silence and pause options require --plan-speech-edits")
    if any(item == "--punch-in-scale" or item.startswith("--punch-in-scale=")
           for item in supplied_args) and not args.render_punch_ins:
        command_parser.error("--punch-in-scale requires --render-punch-ins")
    if args.overwrite_captions and not args.export_captions:
        command_parser.error("--overwrite-captions requires --export-captions")
    try:
        validate_parameters(args.pause_threshold_seconds, args.pause_keep_seconds)
        validate_noise_db(args.silence_noise_db)
        validate_punch_in_scale(args.punch_in_scale)
    except ValueError as exc:
        command_parser.error(str(exc))
    try:
        if args.build_edit_timeline:
            clip_path, edit_path = run_build_edit_timeline(args)
            print(f"Clip plan saved: {clip_path}")
            print(f"Edit timeline saved: {edit_path}")
            return 0
        max_download_bytes(args.max_download_mb)
    except ValueError as exc:
        command_parser.error(str(exc))
    try:
        if args.export_captions:
            manifest = run_export_captions(args)
            print(f"Caption cues: {manifest['cue_count']}\nLanguage: {manifest['language']}\n"
                  f"Duration: {manifest['duration']:.3f}s\n"
                  f"SRT: {manifest['srt_path']}\nVTT: {manifest['vtt_path']}")
            return 0
        if args.render_punch_ins:
            print(f"Punch-in draft rendered and validated: {run_render_punch_ins(args)}")
            return 0
        if args.render_speech_edits:
            print(f"Speech-edited draft rendered and validated: {run_render_speech_edits(args)}")
            return 0
        if args.plan_speech_edits:
            plan_path, map_path = run_plan_speech_edits(args)
            print(f"Speech edit plan saved: {plan_path}\nRetime map saved: {map_path}")
            return 0
        if args.render_edit:
            print(f"Edited draft rendered and validated: {run_render_edit(args)}")
            return 0
        if args.download_sources:
            print(f"Download manifest saved: {run_download_sources(args)}")
            return 0
        if args.select_sources:
            print(f"Source selection saved: {run_select_sources(args)}")
            return 0
        if args.find_sources:
            print(f"Source candidates saved: {run_find_sources(args)}")
            return 0
        if args.plan_only:
            print(f"Director plan saved: {run_plan_only(args)}")
            return 0
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"VideoFactory error: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered and validated: {result['path']}")
    print(f"{result['duration']:.3f}s, {result['width']}x{result['height']}, "
          f"{result['fps']:.3f} fps, {result['video_codec']}/{result['audio_codec']}")
    return 0
