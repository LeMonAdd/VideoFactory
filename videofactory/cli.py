"""VideoFactory command-line orchestration."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import SCHEMA_VERSION
from .asset_manager import discover_assets
from .audio import extract_narration
from .input_handler import inspect_input
from .models import read_json, write_json
from .paths import ROOT, ProjectPaths, safe_project_name
from .renderer import render
from .scene_planner import plan_scenes
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
    result = argparse.ArgumentParser(description="Local VideoFactory V1 editing pipeline")
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Talking-head video with narration")
    source.add_argument("--voice", type=Path, help="Standalone voiceover audio")
    result.add_argument("--project", required=True, help="Safe project name")
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


def run(args: argparse.Namespace) -> dict:
    name = safe_project_name(args.project)
    mode = "TALKING_HEAD" if args.video else "VOICEOVER"
    source, info, media_duration = inspect_input(args.video or args.voice, mode)
    paths = ProjectPaths(ROOT, name)
    if paths.draft.exists() and not args.overwrite_render:
        raise FileExistsError(f"Draft already exists: {paths.draft}; use --overwrite-render")
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
    }
    write_json(paths.project / "project.json", project)
    narration = extract_narration(source, paths.temp / "narration.wav")
    transcriber = (FixtureTranscriber(args.fixture_transcript) if args.fixture_transcript
                   else RealMLXTranscriber(paths.temp, args.whisper_model, args.language))
    transcript = transcriber.transcribe(narration, media_duration)
    write_json(paths.project / "transcript.json", transcript)
    sources = discover_assets(ROOT, mode, source if mode == "TALKING_HEAD" else None,
                              info if mode == "TALKING_HEAD" else None,
                              args.assets_dir.resolve() if args.assets_dir else None)
    write_json(paths.project / "sources.json", sources)
    scenes = plan_scenes(transcript, sources, mode)
    write_json(paths.project / "scenes.json", scenes)
    timeline = build_timeline(scenes, sources, source, mode, DEFAULT_SETTINGS)
    write_json(paths.project / "timeline.json", timeline)
    render(timeline, paths.draft, args.encoder, args.overwrite_render)
    return validate_output(paths.draft, DEFAULT_SETTINGS)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"VideoFactory error: {exc}", file=sys.stderr)
        return 1
    print(f"Rendered and validated: {result['path']}")
    print(f"{result['duration']:.3f}s, {result['width']}x{result['height']}, "
          f"{result['fps']:.3f} fps, {result['video_codec']}/{result['audio_codec']}")
    return 0
