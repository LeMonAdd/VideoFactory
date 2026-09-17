"""VideoFactory command-line orchestration."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION
from .asset_manager import discover_assets
from .audio import extract_narration
from .director_provider import CodexDirector, RuleBasedDirector
from .director_plan import validate_plan
from .director_scenes import convert_plan_to_scenes
from .director_service import create_director_plan
from .input_handler import inspect_input
from .models import read_json, write_json
from .paths import ROOT, ProjectPaths, safe_project_name
from .progress import ProgressReporter
from .renderer import render
from .source_finder import find_sources
from .source_provider import LocalSourceProvider, PexelsSourceProvider
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
                        help="Replace an existing generated draft.mp4")
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
    if args.find_sources:
        if args.plan_only or args.video or args.voice or args.fixture_transcript:
            command_parser.error("--find-sources uses an existing project and cannot take source, fixture, or --plan-only arguments")
        if not args.source_provider:
            command_parser.error("--find-sources requires --source-provider local or pexels")
    elif args.source_provider:
        command_parser.error("--source-provider requires --find-sources")
    elif args.plan_only:
        if args.video or args.voice or args.fixture_transcript:
            command_parser.error("--plan-only uses an existing project and cannot take source or fixture arguments")
    elif not (args.video or args.voice):
        command_parser.error("one of --video or --voice is required unless --plan-only or --find-sources is used")
    if not 1 <= args.source_limit <= 20:
        command_parser.error("--source-limit must be between 1 and 20")
    try:
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
